import argparse
import html
import json
from pathlib import Path
from typing import Any

STYLE = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; display: flex; }
nav { width: 260px; flex: none; height: 100vh; overflow-y: auto; border-right: 1px solid #8884; padding: 12px; box-sizing: border-box; }
nav a { display: block; padding: 6px 8px; border-radius: 6px; text-decoration: none; color: inherit; font-size: 13px; }
nav a:hover { background: #8882; }
nav .grp { font-size: 11px; text-transform: uppercase; opacity: 0.6; margin: 14px 0 4px 8px; }
main { flex: 1; padding: 24px 40px; max-width: 900px; }
.card { border: 1px solid #8884; border-radius: 10px; padding: 20px; margin-bottom: 40px; scroll-margin-top: 20px; }
.card h2 { margin-top: 0; }
.badges span { display: inline-block; font-size: 12px; background: #8882; border-radius: 999px; padding: 2px 10px; margin: 2px 4px 2px 0; }
.badge-challenge { background: #f5a1a133 !important; }
.badge-coverage { background: #a1c9f533 !important; }
.turn { border-left: 3px solid #8884; padding-left: 14px; margin: 14px 0; }
.turn .who { font-weight: 600; font-size: 12px; text-transform: uppercase; opacity: 0.6; }
.turn .user { border-left-color: #4a90d9; }
.expected { background: #8882; border-radius: 8px; padding: 14px; margin-top: 16px; }
.expected h3 { margin-top: 0; font-size: 13px; text-transform: uppercase; opacity: 0.7; }
pre { white-space: pre-wrap; word-break: break-word; font-family: inherit; margin: 4px 0; }
.tuple-table { font-size: 13px; border-collapse: collapse; margin-top: 8px; }
.tuple-table td { padding: 2px 10px 2px 0; vertical-align: top; opacity: 0.85; }
.tuple-table td:first-child { opacity: 0.55; white-space: nowrap; }
.scenario-id { font-family: ui-monospace, monospace; opacity: 0.6; font-size: 13px; }
"""


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get("id") or record.get("scenario_id")
            records[key] = record
    return records


def esc(value: Any) -> str:
    return html.escape(str(value))


def render_tuple(tuple_data: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in tuple_data.items()
    )
    return f'<table class="tuple-table">{rows}</table>'


def render_expected(expected: dict[str, Any]) -> str:
    lines = []
    for key in ("evaluation", "outcome", "criterion", "reason"):
        if key in expected:
            lines.append(f"<div><strong>{esc(key)}:</strong> {esc(expected[key])}</div>")
    source = expected.get("source")
    if source:
        lines.append(
            f"<div><strong>source:</strong> {esc(source.get('type'))} "
            f"&mdash; {esc(source.get('reference'))}</div>"
        )
    return f'<div class="expected"><h3>Expected</h3>{"".join(lines)}</div>'


def render_turns(turns: list[dict[str, str]]) -> str:
    parts = []
    for turn in turns:
        parts.append(
            f'<div class="turn user"><div class="who">User</div>'
            f'<pre>{esc(turn["user"])}</pre></div>'
        )
        parts.append(
            f'<div class="turn"><div class="who">Agent</div>'
            f'<pre>{esc(turn["agent"])}</pre></div>'
        )
    return "".join(parts)


def render_card(scenario_id: str, scenario: dict[str, Any], result: dict[str, Any]) -> str:
    group = scenario.get("scenario_group", result.get("scenario_group", "?"))
    badge_class = f"badge-{group}" if group in ("coverage", "challenge") else ""
    dq = scenario.get("data_quality_case_id")
    badges = [f'<span class="{badge_class}">{esc(group)}</span>']
    if result.get("model"):
        badges.append(f"<span>{esc(result['model'])}</span>")
    if result.get("status"):
        badges.append(f"<span>{esc(result['status'])}</span>")
    if result.get("duration_s") is not None:
        badges.append(f"<span>{result['duration_s']:.1f}s</span>")
    if dq:
        badges.append(f"<span>{esc(dq)}</span>")

    tuple_html = render_tuple(scenario.get("tuple", {})) if scenario else ""
    turns_html = render_turns(result.get("turns", []))
    expected_html = render_expected(result.get("expected") or scenario.get("expected", {}))

    return f"""
    <section class="card" id="{esc(scenario_id)}">
      <div class="scenario-id">{esc(scenario_id)}</div>
      <h2>{esc(scenario_id)}</h2>
      <div class="badges">{"".join(badges)}</div>
      {tuple_html}
      {turns_html}
      {expected_html}
    </section>
    """


def render_nav(ids: list[str], scenarios: dict[str, dict[str, Any]]) -> str:
    items = []
    last_group = None
    for scenario_id in ids:
        group = scenarios.get(scenario_id, {}).get("scenario_group", "")
        if group != last_group:
            items.append(f'<div class="grp">{esc(group or "ungrouped")}</div>')
            last_group = group
        items.append(f'<a href="#{esc(scenario_id)}">{esc(scenario_id)}</a>')
    return f"<nav>{''.join(items)}</nav>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="scenarios/pilot_scenarios.jsonl")
    parser.add_argument("--results", default="scenarios/pilot-results.jsonl")
    parser.add_argument("--output", default="scenarios/results/pilot_results.html")
    args = parser.parse_args()

    scenarios = load_jsonl(Path(args.scenarios))
    results = load_jsonl(Path(args.results))

    ids = sorted(results.keys() or scenarios.keys())
    cards = "".join(
        render_card(scenario_id, scenarios.get(scenario_id, {}), results.get(scenario_id, {}))
        for scenario_id in ids
    )
    nav = render_nav(ids, scenarios)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Pilot results</title><style>{STYLE}</style></head>"
        f"<body>{nav}<main>{cards}</main></body></html>"
    )
    print(f"Wrote {output_path} ({len(ids)} scenarios)")


if __name__ == "__main__":
    main()
