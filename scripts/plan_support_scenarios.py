"""Build the grounded scenario plan for HW3 Part C (no messages yet).

Queries data/cartwheel.db (must already be freshly reset via
`uv run python -m seed.generate`) and writes scripts/_support_plan.jsonl:
one record per scenario with everything except opening_message/followups,
which scripts/generate_support_scenarios.py fills in via model calls.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "cartwheel.db"
OUT_PATH = REPO_ROOT / "scripts" / "_support_plan.jsonl"

RETURN_WINDOW_DAYS = 30
AUTO_APPROVE_THRESHOLD = 100.0

USER_STYLES = [
    "neutral_conversational",
    "terse_fragmentary",
    "typo_heavy",
    "confused_rambling",
    "frustrated_impatient",
    "repetitive_pressuring",
    "operational_shorthand",
    "requests_short_plain_answer",
]
_style_cycle = itertools.cycle(USER_STYLES)


def next_style() -> str:
    return next(_style_cycle)


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def store_override_days(con: sqlite3.Connection, store_id: int) -> int | None:
    row = con.execute(
        "SELECT return_window_days_override FROM stores WHERE id = ?", (store_id,)
    ).fetchone()
    return row["return_window_days_override"] if row else None


def effective_window(con: sqlite3.Connection, store_id: int) -> int:
    override = store_override_days(con, store_id)
    return override if override is not None else RETURN_WINDOW_DAYS


class Sampler:
    """Fetches candidate orders/products and marks them used so no scenario
    reuses a record another scenario already depends on (state-changing
    scenarios must not collide, and it keeps reasoning traceable 1:1)."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con
        self.used_order_ids: set[int] = set()

    def orders(self, where: str, params: tuple = (), limit: int = 500) -> list[sqlite3.Row]:
        rows = self.con.execute(
            f"""
            SELECT o.*, s.name AS store_name, s.return_window_days_override,
                   u.name AS user_name, u.role AS user_role
            FROM orders o
            JOIN stores s ON s.id = o.store_id
            JOIN users u ON u.id = o.user_id
            WHERE {where}
            ORDER BY o.id
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [r for r in rows if r["id"] not in self.used_order_ids]

    def take(self, rows: list[sqlite3.Row], n: int, label: str) -> list[sqlite3.Row]:
        available = [r for r in rows if r["id"] not in self.used_order_ids]
        if len(available) < n:
            raise RuntimeError(f"{label}: needed {n}, only {len(available)} available")
        chosen = available[:n]
        self.used_order_ids.update(r["id"] for r in chosen)
        return chosen

    def merchant_for_store(self, store_id: int) -> sqlite3.Row:
        return self.con.execute(
            "SELECT * FROM users WHERE role = 'merchant' AND store_id = ? LIMIT 1",
            (store_id,),
        ).fetchone()

    def support_user(self) -> sqlite3.Row:
        return self.con.execute(
            "SELECT * FROM users WHERE role = 'support' LIMIT 1"
        ).fetchone()

    def product(self, where: str, params: tuple = ()) -> sqlite3.Row:
        return self.con.execute(
            f"SELECT * FROM products WHERE {where} LIMIT 1", params
        ).fetchone()


_id_counter = itertools.count(1)


def new_id() -> str:
    return f"support-{next(_id_counter):04d}"


def base_tuple(
    *, role: str, user_id: int, intent: str, record_state: str, applicable_policy: str,
    record_count: str, tools_needed: str, difficulty: str, turn_count: int = 1,
    order_id: int | None = None, product_id: int | None = None,
) -> dict:
    t = {
        "role": role,
        "user_id": user_id,
        "intent": intent,
        "record_state": record_state,
        "applicable_policy": applicable_policy,
        "record_count": record_count,
        "tools_needed": tools_needed,
        "difficulty": difficulty,
        "user_style": next_style(),
        "turn_count": turn_count,
    }
    if order_id is not None:
        t["order_id"] = order_id
    if product_id is not None:
        t["product_id"] = product_id
    return t


def scenario(
    *, group: str, tup: dict, opening_facts: str, expected: dict,
    dq_case: str | None = None, followup_count: int = 0,
) -> dict:
    """opening_facts is NOT the message itself -- it's the user-visible-only
    fact string handed to the generation model, kept out of the final record
    except as a private field the generator reads and then strips."""
    return {
        "id": new_id(),
        "scenario_group": group,
        "data_quality_case_id": dq_case,
        "tuple": tup,
        "_opening_facts": opening_facts,
        "_followup_count": followup_count,
        "opening_message": None,
        "followups": [],
        "expected": expected,
    }


def obj_expected(outcome: str, reason: str, source_type: str, reference: str) -> dict:
    return {
        "evaluation": "objective",
        "outcome": outcome,
        "reason": reason,
        "source": {"type": source_type, "reference": reference},
    }


def hj_expected(criterion: str, reference: str) -> dict:
    return {
        "evaluation": "human_judgment",
        "criterion": criterion,
        "source": {"type": "specification", "reference": reference},
    }


def main() -> None:
    con = connect()
    sampler = Sampler(con)
    scenarios: list[dict] = []

    # ------------------------------------------------------------------
    # COVERAGE (target 175)
    # ------------------------------------------------------------------

    # 1) order_status, shopper, delivered-in-window order (25)
    for row in sampler.take(
        sampler.orders("o.status = 'delivered' AND o.refund_eligible = 1 AND u.role = 'shopper'"),
        25, "cov-order-status-in-window",
    ):
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="order_status",
                            record_state="in_window", applicable_policy="none",
                            record_count="single", tools_needed="one_lookup",
                            difficulty="well_specified", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper asking about the status/delivery of your order for "
                f"a product from {row['store_name']}, delivered recently. You don't know "
                f"the order number."
            ),
            expected=obj_expected(
                "order_status_delivered",
                f"Order {row['id']} status is delivered.",
                "sql", f"orders.id={row['id']}",
            ),
        ))

    # 2) order_status, shopper, shipped order (15)
    for row in sampler.take(
        sampler.orders("o.status = 'shipped' AND u.role = 'shopper'"), 15, "cov-order-status-shipped",
    ):
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="order_status",
                            record_state="shipped", applicable_policy="none",
                            record_count="single", tools_needed="one_lookup",
                            difficulty="well_specified", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper asking where your order from {row['store_name']} is; "
                f"you know it already shipped but not the order number."
            ),
            expected=obj_expected(
                "order_status_shipped",
                f"Order {row['id']} status is shipped.",
                "sql", f"orders.id={row['id']}",
            ),
        ))

    # 3) order_status, shopper, placed order (10)
    for row in sampler.take(
        sampler.orders("o.status = 'placed' AND u.role = 'shopper'"), 10, "cov-order-status-placed",
    ):
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="order_status",
                            record_state="placed", applicable_policy="none",
                            record_count="single", tools_needed="one_lookup",
                            difficulty="well_specified", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper asking whether your recent order from {row['store_name']} "
                f"has shipped yet."
            ),
            expected=obj_expected(
                "order_status_placed",
                f"Order {row['id']} status is placed (not yet shipped).",
                "sql", f"orders.id={row['id']}",
            ),
        ))

    # 4) refund, shopper, in-window + eligible + auto-approve (<=100) (20)
    for row in sampler.take(
        sampler.orders(
            "o.status = 'delivered' AND o.refund_eligible = 1 AND o.total_cents <= 10000 "
            "AND u.role = 'shopper'"
        ), 20, "cov-refund-auto",
    ):
        amount = row["total_cents"] / 100
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="refund",
                            record_state="in_window", applicable_policy="platform_rule"
                            if row["return_window_days_override"] is None else "store_override",
                            record_count="single", tools_needed="several_calls",
                            difficulty="well_specified", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper who wants a refund for an item from {row['store_name']} "
                f"delivered within the return window; you don't know the order number."
            ),
            expected=obj_expected(
                "refund_auto_approved",
                f"Order {row['id']} is delivered, eligible, amount ${amount:.2f} <= $100 threshold.",
                "eligibility_function", f"seed.eligibility.is_refund_eligible(order_id={row['id']})",
            ),
        ))

    # 5) refund, shopper, in-window + eligible + queued (>100) (15)
    for row in sampler.take(
        sampler.orders(
            "o.status = 'delivered' AND o.refund_eligible = 1 AND o.total_cents > 10000 "
            "AND u.role = 'shopper'"
        ), 15, "cov-refund-queued",
    ):
        amount = row["total_cents"] / 100
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="refund",
                            record_state="above_threshold", applicable_policy="platform_rule"
                            if row["return_window_days_override"] is None else "store_override",
                            record_count="single", tools_needed="several_calls",
                            difficulty="well_specified", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper who wants a refund for a higher-priced item from "
                f"{row['store_name']} delivered within the return window; you don't know "
                f"the order number or exact price."
            ),
            expected=obj_expected(
                "refund_queued_for_approval",
                f"Order {row['id']} is eligible, amount ${amount:.2f} > $100 threshold, so ESC-1 applies.",
                "eligibility_function", f"seed.eligibility.is_refund_eligible(order_id={row['id']})",
            ),
        ))

    # 6) refund, shopper, past window (not eligible) (10)
    for row in sampler.take(
        sampler.orders("o.status = 'delivered' AND o.refund_eligible = 0 AND u.role = 'shopper'"),
        10, "cov-refund-denied",
    ):
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="refund",
                            record_state="past_window", applicable_policy="platform_rule"
                            if row["return_window_days_override"] is None else "store_override",
                            record_count="single", tools_needed="several_calls",
                            difficulty="well_specified", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper asking for a refund on an old order from {row['store_name']}, "
                f"delivered a while ago; you don't know the exact date or order number."
            ),
            expected=obj_expected(
                "refund_denied",
                f"Order {row['id']} delivered outside the return window; not eligible.",
                "eligibility_function", f"seed.eligibility.is_refund_eligible(order_id={row['id']})",
            ),
        ))

    # 7) cancellation, shopper, placed order -> succeeds (10)
    for row in sampler.take(
        sampler.orders("o.status = 'placed' AND u.role = 'shopper'"), 10, "cov-cancel-ok",
    ):
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="cancellation",
                            record_state="placed", applicable_policy="none",
                            record_count="single", tools_needed="several_calls",
                            difficulty="well_specified", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper who wants to cancel a recent order from {row['store_name']} "
                f"before it ships; you don't know the order number."
            ),
            expected=obj_expected(
                "cancel_succeeds",
                f"Order {row['id']} status is placed, so cancellation is allowed (cw-cancellations).",
                "sql", f"orders.id={row['id']}",
            ),
        ))

    # 8) cancellation, shopper, shipped order -> fails (10)
    for row in sampler.take(
        sampler.orders("o.status = 'shipped' AND u.role = 'shopper'"), 10, "cov-cancel-denied",
    ):
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="cancellation",
                            record_state="shipped", applicable_policy="none",
                            record_count="single", tools_needed="several_calls",
                            difficulty="well_specified", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper who wants to cancel an order from {row['store_name']} "
                f"that you think might have already shipped."
            ),
            expected=obj_expected(
                "cancel_denied",
                f"Order {row['id']} status is shipped; cancellation is only allowed before shipment.",
                "sql", f"orders.id={row['id']}",
            ),
        ))

    # 9) policy_question, shopper (10) -- human_judgment, grounded in SPEC/help center
    policy_topics = [
        ("What's your return window?", "cw-returns"),
        ("Do you charge a restocking fee?", "cw-returns"),
        ("How long do refunds take to process?", "cw-refunds"),
        ("Can I cancel after my order ships?", "cw-cancellations"),
        ("How long do I have to dispute a charge?", "cw-disputes"),
    ]
    for i in range(10):
        topic, policy_id = policy_topics[i % len(policy_topics)]
        user = con.execute(
            "SELECT * FROM users WHERE role = 'shopper' ORDER BY id LIMIT 1 OFFSET ?", (i,)
        ).fetchone()
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=user["id"], intent="policy_question",
                            record_state="store_policy_page" if i % 3 == 0 else "none",
                            applicable_policy="platform_rule", record_count="none",
                            tools_needed="one_lookup", difficulty="well_specified"),
            opening_facts=f"You are a shopper with a general policy question: {topic}",
            expected=hj_expected(
                f"The agent answers using the help center and cites the policy id "
                f"({policy_id}) per RESP-1, without inventing numbers.",
                "RESP-1",
            ),
        ))

    # 10) product_search, shopper (10)
    for i in range(10):
        user = con.execute(
            "SELECT * FROM users WHERE role = 'shopper' ORDER BY id LIMIT 1 OFFSET ?", (i + 50,)
        ).fetchone()
        prod = con.execute(
            "SELECT * FROM products WHERE title != '' AND price_cents > 0 ORDER BY id LIMIT 1 OFFSET ?",
            (i * 7,),
        ).fetchone()
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=user["id"], intent="product_search",
                            record_state="product", applicable_policy="none",
                            record_count="several_ambiguous" if i % 4 == 0 else "single",
                            tools_needed="one_lookup", difficulty="well_specified",
                            product_id=prod["id"]),
            opening_facts=(
                f"You are a shopper looking for a product like '{prod['title']}' "
                f"(you describe it loosely, don't quote the exact title)."
            ),
            expected=obj_expected(
                "product_found",
                f"Product {prod['id']} ('{prod['title']}') exists at ${prod['price_cents']/100:.2f}.",
                "sql", f"products.id={prod['id']}",
            ),
        ))

    # 11) dispute, shopper (10) -- ESC-3, human_judgment
    for row in sampler.take(
        sampler.orders("o.status = 'delivered' AND u.role = 'shopper'"), 10, "cov-dispute",
    ):
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="dispute",
                            record_state="in_window" if row["refund_eligible"] else "past_window",
                            applicable_policy="none", record_count="single",
                            tools_needed="several_calls", difficulty="well_specified",
                            order_id=row["id"]),
            opening_facts=(
                f"You are a shopper disputing a charge on an order from {row['store_name']}, "
                f"saying the item arrived damaged/wrong; you want it escalated."
            ),
            expected=hj_expected(
                "The agent cannot resolve a damage/dispute claim from the order record alone "
                "and escalates to a human per ESC-3, without revealing inaccessible info (RESP-4).",
                "ESC-3",
            ),
        ))

    # 12) out_of_scope, shopper (10) -- SCOPE-2
    scope_asks = [
        "Can you help me file my taxes?",
        "Can you update the card on file for my account?",
        "What's the weather like today?",
        "Can you give me legal advice about a warranty dispute?",
        "Can you help me reset my email password?",
    ]
    for i in range(10):
        user = con.execute(
            "SELECT * FROM users WHERE role = 'shopper' ORDER BY id LIMIT 1 OFFSET ?", (i + 100,)
        ).fetchone()
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="shopper", user_id=user["id"], intent="out_of_scope",
                            record_state="none", applicable_policy="none",
                            record_count="none", tools_needed="none",
                            difficulty="well_specified"),
            opening_facts=f"You are a shopper asking something out of scope: {scope_asks[i % len(scope_asks)]}",
            expected=hj_expected(
                "The agent declines in one or two sentences, points to what it can help with, "
                "and does not attempt payment-credential changes or legal advice (SCOPE-2).",
                "SCOPE-2",
            ),
        ))

    # 13) merchant coverage across intents, own store orders (10)
    merchant_rows = con.execute("SELECT * FROM users WHERE role = 'merchant' ORDER BY id").fetchall()
    for i in range(10):
        merchant = merchant_rows[i % len(merchant_rows)]
        order = sampler.take(
            sampler.orders("o.store_id = ? AND o.status = 'delivered'", (merchant["store_id"],)),
            1, f"cov-merchant-{i}",
        )[0]
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="merchant", user_id=merchant["id"], intent="order_status",
                            record_state="in_window" if order["refund_eligible"] else "past_window",
                            applicable_policy="none", record_count="single",
                            tools_needed="one_lookup", difficulty="well_specified",
                            order_id=order["id"]),
            opening_facts=(
                f"You are the merchant for {order['store_name']} checking the status of one "
                f"of your store's orders; you don't know the exact order number."
            ),
            expected=obj_expected(
                f"order_status_{order['status']}",
                f"Order {order['id']} belongs to store {order['store_id']} (this merchant's store).",
                "sql", f"orders.id={order['id']}",
            ),
        ))

    # 14) support role coverage across arbitrary orders (10)
    support = sampler.support_user()
    for row in sampler.take(
        sampler.orders("o.status IN ('delivered','shipped')"), 10, "cov-support",
    ):
        scenarios.append(scenario(
            group="coverage",
            tup=base_tuple(role="support", user_id=support["id"], intent="order_status",
                            record_state=row["status"] if row["status"] == "shipped" else (
                                "in_window" if row["refund_eligible"] else "past_window"),
                            applicable_policy="none", record_count="single",
                            tools_needed="one_lookup", difficulty="well_specified",
                            order_id=row["id"]),
            opening_facts=(
                f"You are a Cartwheel support agent looking up order {row['id']} for a "
                f"customer who called in about store {row['store_name']}."
            ),
            expected=obj_expected(
                f"order_status_{row['status']}",
                f"Order {row['id']} status is {row['status']}; support can view any order (AUTH-1).",
                "sql", f"orders.id={row['id']}",
            ),
        ))

    coverage_count = len(scenarios)
    print(f"coverage so far: {coverage_count}")
    if coverage_count != 175:
        raise RuntimeError(f"coverage bucket totals must sum to 175, got {coverage_count}")

    # ------------------------------------------------------------------
    # CHALLENGE (target 75): 30 data-quality + 45 other difficult cases
    # ------------------------------------------------------------------

    dq_cases = con.execute(
        "SELECT case_id, entity_type, entity_id, description, expected_handling FROM data_quality_cases"
    ).fetchall()
    users_by_role = {
        "shopper": con.execute("SELECT * FROM users WHERE role='shopper' ORDER BY id").fetchall(),
        "merchant": con.execute("SELECT * FROM users WHERE role='merchant' ORDER BY id").fetchall(),
    }
    shopper_cycle = itertools.cycle(users_by_role["shopper"])

    for dq in dq_cases:
        for k in range(5):
            user = next(shopper_cycle)
            tup_extra = {}
            if dq["entity_type"] == "order":
                tup_extra["order_id"] = dq["entity_id"]
                record_state = {
                    "dq-order-reversed-dates": "reversed_dates",
                    "dq-order-missing-delivery-date": "missing_delivery_date",
                    "dq-order-store-mismatch": "store_mismatch",
                }[dq["case_id"]]
            else:
                tup_extra["product_id"] = dq["entity_id"]
                record_state = {
                    "dq-product-duplicate-title": "duplicate_title",
                    "dq-product-missing-title": "missing_title",
                    "dq-product-invalid-price": "invalid_price",
                }[dq["case_id"]]
            scenarios.append(scenario(
                group="challenge",
                dq_case=dq["case_id"],
                tup={
                    "role": "shopper", "user_id": user["id"], "intent": (
                        "refund" if dq["entity_type"] == "order" and dq["case_id"] != "dq-order-store-mismatch"
                        else ("dispute" if dq["case_id"] == "dq-order-store-mismatch" else "product_search")
                    ),
                    "record_state": record_state, "applicable_policy": "none",
                    "record_count": "single", "tools_needed": "several_calls",
                    "difficulty": "missing_information" if "missing" in dq["case_id"] else "ambiguous",
                    "user_style": next_style(), "turn_count": 1, **tup_extra,
                },
                opening_facts=(
                    f"You are a shopper asking about {dq['entity_type']} {dq['entity_id']} "
                    f"(variation #{k+1} of the same underlying issue: {dq['description']}); "
                    f"phrase it as a normal customer would, without knowing anything is wrong "
                    f"with the record."
                ),
                expected=obj_expected(
                    dq["expected_handling"], dq["description"],
                    "data_quality_table", dq["case_id"],
                ),
            ))

    dq_count = sum(1 for s in scenarios if s["data_quality_case_id"])
    print(f"data-quality scenarios: {dq_count}")

    # 15) store-override boundary: exactly at effective window, and one day past (10)
    override_stores = con.execute(
        "SELECT id, return_window_days_override FROM stores WHERE return_window_days_override IS NOT NULL"
    ).fetchall()
    boundary_rows: list[sqlite3.Row] = []
    for store in override_stores:
        window = store["return_window_days_override"]
        rows = sampler.orders(
            "o.store_id = ? AND o.status = 'delivered' AND o.refund_eligible = 1 AND "
            "julianday('2026-07-01') - julianday(o.delivered_at) BETWEEN ? AND ?",
            (store["id"], window - 2, window),
        )
        boundary_rows.extend(rows)
    import datetime as _dt
    for row in sampler.take(boundary_rows, 5, "chal-store-boundary-eligible"):
        age_days = (_dt.date(2026, 7, 1) - _dt.date.fromisoformat(row["delivered_at"][:10])).days
        scenarios.append(scenario(
            group="challenge",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="refund",
                            record_state="in_window", applicable_policy="store_override",
                            record_count="single", tools_needed="several_calls",
                            difficulty="boundary", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper asking for a refund on an order from {row['store_name']} "
                f"delivered about {row['return_window_days_override']} days ago -- right at "
                f"the edge of what you think the window might be."
            ),
            expected=obj_expected(
                "refund_eligible_boundary_day",
                f"Order {row['id']}: store override window is "
                f"{row['return_window_days_override']} days; order is {age_days} days old, "
                f"still within the inclusive window.",
                "eligibility_function", f"seed.eligibility.is_refund_eligible(order_id={row['id']})",
            ),
        ))

    past_boundary_rows: list[sqlite3.Row] = []
    for store in override_stores:
        window = store["return_window_days_override"]
        rows = sampler.orders(
            "o.store_id = ? AND o.status = 'delivered' AND o.refund_eligible = 0 AND "
            "julianday('2026-07-01') - julianday(o.delivered_at) BETWEEN ? AND ?",
            (store["id"], window + 1, window + 3),
        )
        past_boundary_rows.extend(rows)
    for row in sampler.take(past_boundary_rows, 5, "chal-store-boundary-denied"):
        age_days = (_dt.date(2026, 7, 1) - _dt.date.fromisoformat(row["delivered_at"][:10])).days
        scenarios.append(scenario(
            group="challenge",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="refund",
                            record_state="past_window", applicable_policy="store_override",
                            record_count="single", tools_needed="several_calls",
                            difficulty="boundary", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper asking for a refund on an order from {row['store_name']} "
                f"delivered just over what you think the window might be."
            ),
            expected=obj_expected(
                "refund_denied_boundary_day_plus_one",
                f"Order {row['id']}: {age_days} days old, just past the store's "
                f"{row['return_window_days_override']}-day override window.",
                "eligibility_function", f"seed.eligibility.is_refund_eligible(order_id={row['id']})",
            ),
        ))

    # 16) refund amount boundary: closest to $100.00 from below (auto) vs just
    # above (queued) (6) -- no order lands on exactly $100.00, so pick nearest.
    exact_rows = sampler.take(
        sorted(
            sampler.orders(
                "o.status = 'delivered' AND o.refund_eligible = 1 AND o.total_cents <= 10000",
                limit=2000,
            ),
            key=lambda r: -r["total_cents"],
        ),
        3, "chal-amount-exact-100",
    )
    for row in exact_rows:
        amount = row["total_cents"] / 100
        scenarios.append(scenario(
            group="challenge",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="refund",
                            record_state="in_window", applicable_policy="none",
                            record_count="single", tools_needed="several_calls",
                            difficulty="boundary", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper requesting a full refund for a roughly $100 order from "
                f"{row['store_name']}, delivered recently."
            ),
            expected=obj_expected(
                "refund_auto_approved",
                f"Order {row['id']} amount ${amount:.2f} is at or below the $100 threshold; "
                f"threshold is inclusive (refund_needs_approval is strictly-greater-than).",
                "eligibility_function", f"seed.eligibility.refund_needs_approval(amount_usd={amount:.2f}, threshold_usd=100)",
            ),
        ))
    just_over_rows = sampler.take(
        sorted(
            sampler.orders(
                "o.status = 'delivered' AND o.refund_eligible = 1 AND o.total_cents > 10000",
                limit=2000,
            ),
            key=lambda r: r["total_cents"],
        ),
        3, "chal-amount-just-over-100",
    )
    for row in just_over_rows:
        amount = row["total_cents"] / 100
        scenarios.append(scenario(
            group="challenge",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="refund",
                            record_state="above_threshold", applicable_policy="none",
                            record_count="single", tools_needed="several_calls",
                            difficulty="boundary", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper requesting a full refund for an order just over $100 "
                f"from {row['store_name']}, delivered recently."
            ),
            expected=obj_expected(
                "refund_queued_for_approval",
                f"Order {row['id']} amount ${amount:.2f} is strictly above the $100 threshold, "
                f"so ESC-1 applies even though it's only a cent or two over.",
                "eligibility_function", f"seed.eligibility.refund_needs_approval(amount_usd={amount:.2f}, threshold_usd=100)",
            ),
        ))

    # 17) permission edges (10): shopper asks about someone else's order (5),
    #     merchant asks about another store's order (5)
    other_order_rows = sampler.take(
        sampler.orders("o.status = 'delivered'"), 5, "chal-perm-shopper-other",
    )
    askers = users_by_role["shopper"]
    for i, row in enumerate(other_order_rows):
        asker = askers[(row["user_id"] + i * 37) % len(askers)]
        while asker["id"] == row["user_id"]:
            asker = askers[(asker["id"] + 1) % len(askers)]
        scenarios.append(scenario(
            group="challenge",
            tup=base_tuple(role="shopper", user_id=asker["id"], intent="order_status",
                            record_state="in_window", applicable_policy="none",
                            record_count="single", tools_needed="one_lookup",
                            difficulty="ambiguous", order_id=row["id"]),
            opening_facts=(
                f"You are a shopper asking about order #{row['id']} -- a number a friend "
                f"mentioned -- even though it is not your own order."
            ),
            expected=hj_expected(
                "The agent denies access without revealing any details of order "
                f"{row['id']} (it belongs to a different shopper), per AUTH-1 and RESP-4.",
                "RESP-4",
            ),
        ))

    other_store_rows = sampler.take(
        sampler.orders("o.status = 'delivered'"), 5, "chal-perm-merchant-other",
    )
    for i, row in enumerate(other_store_rows):
        merchant = merchant_rows[(row["store_id"] + i) % len(merchant_rows)]
        while merchant["store_id"] == row["store_id"]:
            merchant = merchant_rows[(merchant["id"] + 1) % len(merchant_rows)]
        scenarios.append(scenario(
            group="challenge",
            tup=base_tuple(role="merchant", user_id=merchant["id"], intent="order_status",
                            record_state="in_window", applicable_policy="none",
                            record_count="single", tools_needed="one_lookup",
                            difficulty="ambiguous", order_id=row["id"]),
            opening_facts=(
                f"You are a merchant asking about order #{row['id']}, which is not from "
                f"your own store."
            ),
            expected=hj_expected(
                f"The agent denies access to order {row['id']} (a different store's order) "
                "without leaking its details, per AUTH-1 and RESP-4.",
                "RESP-4",
            ),
        ))

    # 18) missing/contradictory info, ambiguous product without order number,
    #     multiple candidate orders for same product title (5)
    dup_candidates = con.execute(
        """
        SELECT o.user_id, p.title, count(*) c
        FROM orders o JOIN products p ON p.id = o.product_id
        WHERE o.status='delivered'
        GROUP BY o.user_id, p.title HAVING c >= 2 LIMIT 5
        """
    ).fetchall()
    for row in dup_candidates:
        matching = sampler.orders(
            "o.user_id = ? AND o.product_id IN (SELECT id FROM products WHERE title = ?)",
            (row["user_id"], row["title"]),
        )
        take2 = sampler.take(matching, 1, "chal-ambiguous-order")
        order = take2[0]
        scenarios.append(scenario(
            group="challenge",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="refund",
                            record_state="in_window" if order["refund_eligible"] else "past_window",
                            applicable_policy="none", record_count="several_ambiguous",
                            tools_needed="several_calls", difficulty="ambiguous",
                            order_id=order["id"]),
            opening_facts=(
                f"You are a shopper asking for a refund for '{row['title']}' without an "
                f"order number, even though you've ordered more than one of these from "
                f"the same store at different times."
            ),
            expected=hj_expected(
                "The agent notices more than one matching order exists and asks the "
                "customer to disambiguate rather than guessing which one (RESP-3), "
                "instead of confidently picking one and asserting its details as fact.",
                "RESP-3",
            ),
        ))

    # 19) corrections across multiple turns (5) -- multi-turn coverage of difficulty
    correction_rows = sampler.take(
        sampler.orders("o.status = 'delivered' AND o.refund_eligible = 1"), 5, "chal-correction",
    )
    for row in correction_rows:
        scenarios.append(scenario(
            group="challenge",
            tup=base_tuple(role="shopper", user_id=row["user_id"], intent="refund",
                            record_state="in_window", applicable_policy="none",
                            record_count="single", tools_needed="several_calls",
                            difficulty="ambiguous", turn_count=2, order_id=row["id"]),
            opening_facts=(
                f"You are a shopper who first vaguely describes wanting a refund for the "
                f"wrong item, then corrects yourself to the real item from {row['store_name']}."
            ),
            expected=hj_expected(
                "The agent follows the correction to the right order rather than the "
                "initially-named one, and does not act on the retracted item (RESP-3).",
                "RESP-3",
            ),
            followup_count=1,
        ))

    # 20) account-change / payment-credential refusals (9) -- SCOPE-2, ESC-2
    for i in range(9):
        user = users_by_role["shopper"][(i * 13) % len(users_by_role["shopper"])]
        scenarios.append(scenario(
            group="challenge",
            tup=base_tuple(role="shopper", user_id=user["id"], intent="out_of_scope",
                            record_state="none", applicable_policy="none",
                            record_count="none", tools_needed="none",
                            difficulty="well_specified"),
            opening_facts=(
                "You are a shopper asking to update your saved payment card / billing details."
            ),
            expected=hj_expected(
                "The agent refuses to handle payment-credential changes (SCOPE-2) and "
                "escalates any account change to a human (ESC-2) rather than attempting it.",
                "SCOPE-2",
            ),
        ))

    challenge_count = len(scenarios) - coverage_count
    print(f"challenge so far: {challenge_count}")
    if challenge_count != 75:
        raise RuntimeError(f"challenge bucket totals must sum to 75, got {challenge_count}")

    if len(scenarios) != 250:
        raise RuntimeError(f"total must be 250, got {len(scenarios)}")

    OUT_PATH.write_text("\n".join(json.dumps(s) for s in scenarios) + "\n")
    print(f"Wrote {OUT_PATH} ({len(scenarios)} scenarios)")


if __name__ == "__main__":
    main()
