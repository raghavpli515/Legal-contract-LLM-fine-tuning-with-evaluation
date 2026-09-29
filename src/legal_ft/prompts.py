"""Single source of truth for prompt templates.

Training, evaluation and the demo all build messages through these functions,
so the fine-tuned model is always queried in exactly the format it was trained on.
Messages are plain chat dicts; the tokenizer's chat template is applied later.
"""

from __future__ import annotations

import json

# The 41 CUAD v1 categories. Names are verified against CUADv1.json during data prep.
CUAD_CATEGORIES: tuple[str, ...] = (
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law",
    "Most Favored Nation", "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
    "Competitive Restriction Exception", "No-Solicit Of Employees", "Non-Disparagement",
    "Termination For Convenience", "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment", "Volume Restriction",
    "Ip Ownership Assignment", "Joint Ip Ownership", "License Grant",
    "Non-Transferable License", "Affiliate License-Licensor", "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License", "Irrevocable Or Perpetual License",
    "Source Code Escrow", "Post-Termination Services", "Audit Rights", "Uncapped Liability",
    "Cap On Liability", "Liquidated Damages", "Warranty Duration", "Insurance",
    "Covenant Not To Sue", "Third Party Beneficiary",
)

SYSTEM_PROMPT = (
    "You are a contract analysis assistant. You answer strictly from the contract text "
    "provided. If the text does not contain what is asked, say so. Never invent clauses "
    "or quote text that is not in the contract."
)

Message = dict[str, str]


def classification_labels(excluded: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Clause-type label set: CUAD categories minus the excluded metadata fields."""
    excluded_set = set(excluded)
    return [c for c in CUAD_CATEGORIES if c not in excluded_set]


def build_classification_messages(clause: str, labels: list[str]) -> list[Message]:
    label_list = "\n".join(f"- {label}" for label in labels)
    user = (
        "Classify the following contract clause into exactly one clause type.\n\n"
        f"Clause types:\n{label_list}\n\n"
        f"Clause:\n\"\"\"\n{clause}\n\"\"\"\n\n"
        "Answer with the clause type name only."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def classification_target(label: str) -> str:
    return label


def build_qa_messages(excerpt: str, category: str, description: str = "") -> list[Message]:
    detail = f" ({description})" if description else ""
    user = (
        f"Contract excerpt:\n\"\"\"\n{excerpt}\n\"\"\"\n\n"
        f"Question: Does this excerpt contain a \"{category}\" clause{detail}? "
        "If yes, quote the relevant text verbatim.\n\n"
        'Respond with JSON only: {"present": true|false, "evidence": "<verbatim quote>" | null}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def qa_target(present: bool, evidence: str | None) -> str:
    """Canonical QA completion. ``present`` comes first so its token probability
    can be read off as the model's confidence (see eval/generate.py)."""
    if not present:
        evidence = None
    return json.dumps({"present": present, "evidence": evidence}, ensure_ascii=False)
