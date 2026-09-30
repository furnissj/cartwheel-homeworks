import json
import re
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seed.eligibility import is_refund_eligible, refund_needs_approval

DB_PATH = Path("data/cartwheel.db")
SCENARIOS_PATH = Path("scenarios/support_scenarios.jsonl")
# The seeded world runs on a fixed simulated "today" (seed/generate.py
# WORLD_ASOF), never the real wall-clock date, so grounding checks must use
# the same anchor the agent and the runner do.
TODAY = date(2026, 7, 1)
RETURN_WINDOW_DAYS = 30
REFUND_THRESHOLD_USD = 100.0


def parse_date(s):
    return date.fromisoformat(s) if s else None


def load_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_order(conn, order_id):
    row = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
    return dict(row) if row else None


def get_product(conn, product_id):
    row = conn.execute("SELECT * FROM products WHERE id=?", (product_id,)).fetchone()
    return dict(row) if row else None


def get_store(conn, store_id):
    row = conn.execute("SELECT * FROM stores WHERE id=?", (store_id,)).fetchone()
    return dict(row) if row else None


def get_user(conn, user_id):
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def main():
    conn = load_db()
    recs = [json.loads(l) for l in open(SCENARIOS_PATH)]
    findings = []

    for r in recs:
        sid = r["id"]
        t = r["tuple"]
        role = t.get("role")
        intent = t.get("intent")
        exp = r["expected"]
        msg = r["opening_message"] + " " + " ".join(r.get("followups") or [])

        order = get_order(conn, t["order_id"]) if t.get("order_id") else None
        product = get_product(conn, t["product_id"]) if t.get("product_id") else None

        # --- auth scoping check ---
        # Only flagged for objective evaluations, which assume the lookup
        # succeeds and compute a real fact. A human_judgment scenario whose
        # own criterion expects a permission denial (an intentional
        # auth-boundary challenge case) is fine as-is.
        if order is not None and exp.get("evaluation") == "objective":
            caller = get_user(conn, t.get("user_id"))
            if role == "shopper" and order["user_id"] != t.get("user_id"):
                findings.append((sid, "AUTH", f"shopper user_id={t.get('user_id')} does not own order {order['id']} (owner={order['user_id']})"))
            if role == "merchant":
                if caller is None or caller.get("store_id") != order["store_id"]:
                    findings.append((sid, "AUTH", f"merchant user_id={t.get('user_id')} store does not match order {order['id']} store={order['store_id']}"))

        # --- product store scoping (merchant) ---
        if product is not None and role == "merchant":
            caller = get_user(conn, t.get("user_id"))
            if caller is None or caller.get("store_id") != product["store_id"]:
                findings.append((sid, "AUTH", f"merchant user_id={t.get('user_id')} store does not match product {product['id']} store={product['store_id']}"))

        # --- ambiguous merchant/support order lookup with no order number given ---
        if intent == "order_status" and role in ("merchant", "support") and order is not None:
            has_number = bool(re.search(rf"\b{order['id']}\b", msg))
            if not has_number and exp.get("evaluation") == "objective":
                findings.append((sid, "AMBIGUOUS_LOOKUP", f"role={role} names no order number for order {order['id']}; objective evaluation not gradable"))

        # --- refund eligibility grounding ---
        # Skipped for data-quality-case scenarios: their outcome is a
        # deliberate escalation/refusal instruction (e.g. "do not compute a
        # deadline from a missing date"), not a plain eligible/denied claim.
        if intent == "refund" and order is not None and exp.get("evaluation") == "objective" and not r.get("data_quality_case_id"):
            store = get_store(conn, order["store_id"])
            window = (store["return_window_days_override"] if store else None) or RETURN_WINDOW_DAYS
            delivered = parse_date(order["delivered_at"])
            eligible = is_refund_eligible(
                status=order["status"], delivered_at=delivered, as_of=TODAY, return_window_days=window
            )
            outcome = exp.get("outcome", "")
            said_eligible = "denied" not in outcome.lower() and "ineligible" not in outcome.lower()
            if eligible != said_eligible:
                age = (TODAY - delivered).days if delivered else None
                findings.append((sid, "REFUND_ELIGIBILITY", f"order {order['id']} status={order['status']} delivered={order['delivered_at']} window={window}d age={age}d -> eligible={eligible}, but scenario says outcome={outcome!r}"))
            elif eligible and outcome in ("refund_auto_approved", "refund_queued_for_approval"):
                # Only scenarios whose outcome is literally one of these two
                # labels are making a checkable threshold-routing claim.
                # Boundary-day scenarios and data-quality-case scenarios use
                # other outcome labels deliberately scoped to a narrower
                # question (is it in the window / is the record usable at
                # all), not the full auto-approve-vs-queue routing.
                amount_usd = order["total_cents"] / 100
                needs_approval = refund_needs_approval(amount_usd, REFUND_THRESHOLD_USD)
                said_queued = outcome == "refund_queued_for_approval"
                if needs_approval != said_queued:
                    findings.append((sid, "REFUND_THRESHOLD", f"order {order['id']} amount=${amount_usd:.2f} needs_approval={needs_approval}, but scenario says outcome={outcome!r}"))

        # --- cancellation status grounding ---
        if intent == "cancellation" and order is not None and exp.get("evaluation") == "objective":
            cancellable = order["status"] == "placed"
            outcome = exp.get("outcome", "")
            said_cancellable = "denied" not in outcome.lower()
            if cancellable != said_cancellable:
                findings.append((sid, "CANCEL_STATUS", f"order {order['id']} status={order['status']} cancellable={cancellable}, but scenario says outcome={outcome!r}"))

        # --- turn/followup consistency ---
        n_followups = len(r.get("followups") or [])
        if t.get("turn_count") != n_followups + 1:
            findings.append((sid, "TURN_COUNT", f"turn_count={t.get('turn_count')} but {n_followups} followups"))

        # --- data quality case cross-check ---
        dq = r.get("data_quality_case_id")
        if dq:
            row = conn.execute(
                "SELECT entity_type, entity_id FROM data_quality_cases WHERE case_id=?", (dq,)
            ).fetchone()
            if row is None:
                findings.append((sid, "DQ_UNKNOWN", f"unknown data_quality_case_id {dq}"))
            else:
                entity_type, entity_id = row
                ref_id = t.get("order_id") if entity_type == "order" else t.get("product_id")
                if ref_id != entity_id:
                    findings.append((sid, "DQ_MISMATCH", f"case {dq} expects {entity_type} {entity_id}, scenario references {ref_id}"))

        # --- followup plausibility heuristic ---
        for i, f in enumerate(r.get("followups") or []):
            if re.match(r"^(yes|sure|ok(ay)?|go ahead|please (do|proceed)|sounds good)\b", f.strip(), re.I):
                findings.append((sid, "FOLLOWUP_ASSUMES_RESPONSE", f"followup {i+1} may assume a prior agent offer: {f!r}"))

    conn.close()

    by_type = {}
    for sid, kind, detail in findings:
        by_type.setdefault(kind, []).append((sid, detail))

    for kind, items in sorted(by_type.items()):
        print(f"\n=== {kind} ({len(items)}) ===")
        for sid, detail in items:
            print(f"  {sid}: {detail}")

    print(f"\nTotal findings: {len(findings)} across {len(set(f[0] for f in findings))} scenarios")


if __name__ == "__main__":
    main()
