"""Step 5 (generation) + critic pass for HW3 Part C.

Reads scripts/_support_plan.jsonl (grounded plan, no messages yet, produced
by scripts/plan_support_scenarios.py). For each scenario, makes one model
call to write the opening_message (+ followups, for multi-turn scenarios)
from the user-visible facts only -- never the expected/grounding block.
Then runs an independent critic call per conversation. Writes
scenarios/support_scenarios.jsonl with the internal-only fields stripped.

Resumable: reruns only scenarios missing from the intermediate output files.
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from observability.instrument import load_env  # noqa: E402

load_env()

import litellm  # noqa: E402

PLAN_PATH = REPO_ROOT / "scripts" / "_support_plan.jsonl"
GENERATED_PATH = REPO_ROOT / "scripts" / "_support_generated.jsonl"
CRITIQUED_PATH = REPO_ROOT / "scripts" / "_support_critiqued.jsonl"
FINAL_PATH = REPO_ROOT / "scenarios" / "support_scenarios.jsonl"

LITELLM_COURSE_MODELS = {
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "glm-5.2": "together_ai/zai-org/GLM-5.2",
}


def litellm_model_id() -> str:
    name = os.environ.get("CARTWHEEL_MODEL", "gpt-5.5")
    if name.startswith("gpt-"):
        return name
    return LITELLM_COURSE_MODELS.get(name, name)


MODEL = litellm_model_id()
WORKERS = 8


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def call_model(messages: list[dict]) -> str:
    resp = litellm.completion(model=MODEL, messages=messages, temperature=1.0, timeout=60)
    return resp.choices[0].message.content


GEN_SYSTEM = (
    "You simulate a real customer messaging a customer-support chat agent. "
    "Write naturally, the way a real person types into a chat window -- not "
    "polished customer-service prose. Never state facts you were not given. "
    "Never mention order numbers, exact dates, internal policy names, rule "
    "numbers, or anything about traces, tools, or prompts. Respond with a "
    "single JSON object and nothing else: "
    '{"opening_message": "...", "followups": ["...", ...]}. '
    "The followups list must have exactly the requested count, or be empty "
    "if none were requested. Each followup must make sense as a natural "
    "continuation without knowing what the agent said -- do not write a "
    "followup like 'yes, go ahead' that assumes a specific agent reply; "
    "instead add detail, a correction, or restate frustration."
)


def gen_prompt(plan: dict) -> str:
    tup = plan["tuple"]
    return (
        f"Role: {tup['role']}\n"
        f"Facts you know: {plan['_opening_facts']}\n"
        f"Language style to use: {tup['user_style']}\n"
        f"Number of user turns: {tup['turn_count']} "
        f"(opening message plus {plan['_followup_count']} followups)\n"
        f"Write the opening message and any followups now, as JSON."
    )


CRITIC_SYSTEM = (
    "You are a critic reviewing a simulated customer conversation for a "
    "synthetic evaluation dataset. Check for: (1) invented identifiers, "
    "amounts, dates, or entity names not present in the given facts; "
    "(2) a followup that only makes sense assuming a specific agent reply; "
    "(3) a templated/generic opener that would repeat across many "
    "conversations; (4) language a real user would not produce, such as "
    "quoting internal policy IDs or rule numbers; (5) any reference to "
    "tools, traces, prompts, or the grading/expected-answer metadata. "
    "If the conversation is fine, return it unchanged. If not, rewrite only "
    "what's necessary to fix the issue(s) -- preserve the facts, the "
    "assigned style, and the number of turns exactly. Respond with a single "
    'JSON object and nothing else: {"opening_message": "...", '
    '"followups": ["...", ...], "notes": "one short sentence, or empty '
    'string if no changes were needed"}.'
)


def critic_prompt(plan: dict, generated: dict) -> str:
    tup = plan["tuple"]
    convo = json.dumps(
        {"opening_message": generated["opening_message"], "followups": generated["followups"]}
    )
    return (
        f"Facts the user was given (must not be contradicted or exceeded): "
        f"{plan['_opening_facts']}\n"
        f"Assigned style: {tup['user_style']}\n"
        f"Generated conversation: {convo}\n"
        f"Review and return the JSON object."
    )


def generate_one(plan: dict) -> dict:
    raw = call_model([
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": gen_prompt(plan)},
    ])
    parsed = extract_json(raw)
    return {
        "id": plan["id"],
        "opening_message": parsed["opening_message"],
        "followups": parsed.get("followups") or [],
    }


def critique_one(plan: dict, generated: dict) -> dict:
    raw = call_model([
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": critic_prompt(plan, generated)},
    ])
    parsed = extract_json(raw)
    return {
        "id": plan["id"],
        "opening_message": parsed["opening_message"],
        "followups": parsed.get("followups") or [],
        "notes": parsed.get("notes", ""),
    }


def run_phase(plans: list[dict], done_ids: set[str], out_path: Path, worker) -> tuple[int, int]:
    todo = [p for p in plans if p["id"] not in done_ids]
    if not todo:
        return 0, 0
    ok, failed = 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, p): p for p in todo}
        for future in as_completed(futures):
            plan = futures[future]
            try:
                result = future.result()
                append_jsonl(out_path, result)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {plan['id']}: {exc}", file=sys.stderr)
                failed += 1
    return ok, failed


def main() -> None:
    plans = load_jsonl(PLAN_PATH)
    plans_by_id = {p["id"]: p for p in plans}
    print(f"Model: {MODEL}, plan size: {len(plans)}")

    generated = load_jsonl(GENERATED_PATH)
    generated_ids = {g["id"] for g in generated}
    print(f"Generation: {len(generated_ids)}/{len(plans)} already done")
    ok, failed = run_phase(plans, generated_ids, GENERATED_PATH, generate_one)
    print(f"Generation phase: +{ok} ok, {failed} failed")

    generated = load_jsonl(GENERATED_PATH)
    generated_by_id = {g["id"]: g for g in generated}
    missing = [p["id"] for p in plans if p["id"] not in generated_by_id]
    if missing:
        print(f"WARNING: {len(missing)} scenarios have no generated conversation yet: {missing[:10]}")

    critiqued = load_jsonl(CRITIQUED_PATH)
    critiqued_ids = {c["id"] for c in critiqued}
    print(f"Critic: {len(critiqued_ids)}/{len(generated)} already done")
    ok, failed = 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(critique_one, plans_by_id[g["id"]], g): g
            for g in generated if g["id"] not in critiqued_ids
        }
        for future in as_completed(futures):
            g = futures[future]
            try:
                result = future.result()
                append_jsonl(CRITIQUED_PATH, result)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"CRITIC FAILED {g['id']}: {exc}", file=sys.stderr)
                failed += 1
    print(f"Critic phase: +{ok} ok, {failed} failed")

    critiqued = load_jsonl(CRITIQUED_PATH)
    critiqued_by_id = {c["id"]: c for c in critiqued}
    missing_critique = [p["id"] for p in plans if p["id"] not in critiqued_by_id]
    if missing_critique:
        print(f"WARNING: {len(missing_critique)} scenarios never got a critic pass, will fall back to raw generation")

    final_records = []
    notes_count = 0
    for plan in plans:
        pid = plan["id"]
        source = critiqued_by_id.get(pid) or generated_by_id.get(pid)
        if source is None:
            print(f"MISSING entirely, excluded from final file: {pid}", file=sys.stderr)
            continue
        if source.get("notes"):
            notes_count += 1
        record = {
            "id": plan["id"],
            "scenario_group": plan["scenario_group"],
            "data_quality_case_id": plan["data_quality_case_id"],
            "tuple": plan["tuple"],
            "opening_message": source["opening_message"],
            "followups": source["followups"],
            "expected": plan["expected"],
        }
        final_records.append(record)

    FINAL_PATH.write_text("\n".join(json.dumps(r) for r in final_records) + "\n")
    print(f"Wrote {FINAL_PATH} ({len(final_records)} records, {notes_count} critic edits)")


if __name__ == "__main__":
    main()
