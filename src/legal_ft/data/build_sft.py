"""Build chat-format JSONL for both tasks from CUAD, per contract-level split.

    python -m legal_ft.data.build_sft

Writes to data/processed/:
  train.jsonl, val.jsonl              mixed tasks, TRL prompt/completion format
  eval_classification.jsonl           frozen held-out eval sets (test split only),
  eval_qa.jsonl                       identical for the base and fine-tuned runs
  categories.json, splits.json, stats.json

Every record carries the fields scoring needs (gold labels / evidence, excerpt,
contract id) alongside ``prompt`` and ``completion``.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from legal_ft.config import REPO_ROOT, load_config, load_dotenv
from legal_ft.data.cuad import Contract, Span, load_contracts
from legal_ft.data.split import iterative_split
from legal_ft.data.windows import (
    Tokens,
    Window,
    evidence_in_window,
    positive_window,
    relation,
    tile,
)
from legal_ft.prompts import (
    build_classification_messages,
    build_qa_messages,
    classification_labels,
    classification_target,
    qa_target,
)

# Chat-template overhead (system/user/assistant markers) not counted by plain encoding.
TEMPLATE_TOKENS = 32


@dataclass
class QAPositive:
    contract_id: str
    category: str
    window: Window
    evidence: list[str]


def classification_items(
    contract: Contract, included: set[str], rarity: Counter, cfg: dict
) -> list[dict]:
    """One item per distinct gold span. Spans with several labels keep all of them as
    gold; the training target is the rarest (most specific) one."""
    by_range: dict[tuple[int, int], set[str]] = defaultdict(set)
    for span in contract.spans:
        if span.category in included:
            by_range[(span.start, span.end)].add(span.category)
    items = []
    for (start, end), labels in sorted(by_range.items()):
        clause = contract.text[start:end]
        if len(clause) < cfg["min_clause_chars"]:
            continue
        gold = sorted(labels, key=lambda c: (rarity[c], c))
        items.append({
            "task": "classification",
            "contract_id": contract.id,
            "clause": clause[: cfg["max_clause_chars"]],
            "labels": gold,
            "target_label": gold[0],
        })
    return items


def qa_positives(
    contract: Contract, tokens: Tokens, included: set[str], cfg: dict, rng: random.Random,
    skipped: Counter,
) -> list[QAPositive]:
    by_cat: dict[str, list[Span]] = defaultdict(list)
    for span in contract.spans:
        if span.category in included:
            by_cat[span.category].append(span)
    out, seen = [], set()
    for category, spans in sorted(by_cat.items()):
        for span in spans:
            window = positive_window(tokens, span, cfg["window_tokens"], rng)
            if window is None:
                skipped["span_longer_than_window"] += 1
                continue
            full, cut = evidence_in_window(spans, window)
            if cut:
                skipped["other_span_cut_by_window"] += 1
                continue
            evidence = list(dict.fromkeys(contract.quote(s) for s in full))
            if sum(map(len, evidence)) > cfg["max_evidence_chars"]:
                skipped["evidence_too_long"] += 1
                continue
            key = (category, tuple(evidence))
            if key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(key)
            out.append(QAPositive(contract.id, category, window, evidence))
    return out


def free_windows(contract: Contract, tiles: list[Window], category: str) -> list[Window]:
    """Tiles with no overlap at all with any span of ``category``."""
    spans = [s for s in contract.spans if s.category == category]
    return [w for w in tiles if all(relation(s, w) is None for s in spans)]


def matched_negative(
    pos: QAPositive, contracts: dict[str, Contract], tiles: dict[str, list[Window]],
    pool_ids: list[str], hard_ratio: float, rng: random.Random,
) -> dict | None:
    """A "present": false example for the same category as ``pos``."""
    hard = free_windows(contracts[pos.contract_id], tiles[pos.contract_id], pos.category)
    easy_ids = [c for c in pool_ids if pos.category not in contracts[c].categories()]
    use_hard = hard and (rng.random() < hard_ratio or not easy_ids)
    if use_hard:
        cid, window, neg_type = pos.contract_id, rng.choice(hard), "same_contract"
    elif easy_ids:
        cid = rng.choice(easy_ids)
        window, neg_type = rng.choice(tiles[cid]), "other_contract"
    else:
        return None
    return qa_record(contracts[cid], pos.category, window, [], neg_type)


def qa_record(
    contract: Contract, category: str, window: Window, evidence: list[str],
    neg_type: str | None,
) -> dict:
    excerpt = contract.text[window[0] : window[1]]
    assert all(q in excerpt for q in evidence), "gold evidence must be verbatim in the excerpt"
    return {
        "task": "qa",
        "contract_id": contract.id,
        "category": category,
        "window": list(window),
        "excerpt": excerpt,
        "present": bool(evidence),
        "evidence": evidence,
        "neg_type": neg_type,
    }


def balanced_sample(
    items: list, key: Callable, n: int, rng: random.Random
) -> list:
    """Round-robin over groups: as close to equal per group as the data allows."""
    groups: dict = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    out: list = []
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def attach_prompt(record: dict, labels: list[str], descriptions: dict[str, str]) -> dict:
    if record["task"] == "classification":
        prompt = build_classification_messages(record["clause"], labels)
        target = classification_target(record["target_label"])
    else:
        prompt = build_qa_messages(
            record["excerpt"], record["category"], descriptions.get(record["category"], "")
        )
        target = qa_target(record["present"], record["evidence"])
    record["prompt"] = prompt
    record["completion"] = [{"role": "assistant", "content": target}]
    return record


def count_tokens(record: dict, tokenizer) -> int:
    text = "".join(m["content"] for m in record["prompt"] + record["completion"])
    return len(tokenizer.encode(text, add_special_tokens=False).ids) + TEMPLATE_TOKENS


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    n = 0
    # newline=\n: identical bytes on Windows and Linux, so checksums are portable
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def build() -> dict:
    load_dotenv()
    from tokenizers import Tokenizer

    from legal_ft.data.download import fetch_cuad, sha256_of

    dcfg, ecfg = load_config("data"), load_config("eval")
    wcfg, ccfg, scfg = dcfg["windows"], dcfg["classification"], dcfg["sampling"]
    out_dir = REPO_ROOT / dcfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_pretrained(wcfg["tokenizer"])

    contracts_list, descriptions = load_contracts(fetch_cuad())
    contracts = {c.id: c for c in contracts_list}
    labels = classification_labels(dcfg["excluded_categories"])
    included = set(labels)
    rarity = Counter(s.category for c in contracts_list for s in c.spans)

    split_of = iterative_split(
        {cid: c.categories() & included for cid, c in contracts.items()},
        {k: dcfg["split"][k] for k in ("train", "val", "test")},
        dcfg["split"]["seed"],
    )

    rng = random.Random(scfg["seed"])
    skipped: Counter = Counter()
    cls_by_split: dict[str, list[dict]] = defaultdict(list)
    pos_by_split: dict[str, list[QAPositive]] = defaultdict(list)
    tiles: dict[str, list[Window]] = {}
    for cid in sorted(contracts):
        contract, split = contracts[cid], split_of[cid]
        tokens = Tokens.from_tokenizer(contract.text, tokenizer)  # discarded after this contract
        tiles[cid] = tile(tokens, wcfg["window_tokens"])
        cls_by_split[split] += classification_items(contract, included, rarity, ccfg)
        pos_by_split[split] += qa_positives(contract, tokens, included, wcfg, rng, skipped)

    def qa_set(split: str, n_pos: int, n_neg: int, r: random.Random) -> list[dict]:
        pool_ids = sorted(c for c in contracts if split_of[c] == split)
        positives = balanced_sample(pos_by_split[split], lambda p: p.category, n_pos, r)
        records = [
            qa_record(contracts[p.contract_id], p.category, p.window, p.evidence, None)
            for p in positives
        ]
        negatives = []
        for p in positives:
            neg = matched_negative(p, contracts, tiles, pool_ids, wcfg["hard_negative_ratio"], r)
            if neg is not None:
                negatives.append(neg)
        return records + negatives[:n_neg]

    def finalize(records: list[dict], split: str, r: random.Random) -> list[dict]:
        out = []
        for i, rec in enumerate(records):
            rec = attach_prompt(rec, labels, descriptions)
            rec["n_tokens"] = count_tokens(rec, tokenizer)
            if rec["n_tokens"] > scfg["max_seq_tokens"]:
                skipped[f"{split}_over_max_seq_tokens"] += 1
                continue
            rec["id"] = f"{split}-{rec['task']}-{i:05d}"
            rec["split"] = split
            out.append(rec)
        r.shuffle(out)
        return out

    by_label = lambda it: it["target_label"]
    files: dict[str, list[dict]] = {}
    for split in ("train", "val"):
        n = scfg[split]
        r = random.Random(f"{scfg['seed']}-{split}")
        records = balanced_sample(cls_by_split[split], by_label, n["classification"], r)
        records += qa_set(split, n["qa_positives"], n["qa_positives"], r)
        files[split] = finalize(records, split, r)

    es = ecfg["eval_set"]
    r = random.Random(es["seed"])
    n_pos = round(es["n_qa"] * es["qa_present_ratio"])
    files["eval_classification"] = finalize(
        balanced_sample(cls_by_split["test"], by_label, es["n_classification"], r), "test", r
    )
    files["eval_qa"] = finalize(qa_set("test", n_pos, es["n_qa"] - n_pos, r), "test", r)

    checksums = {}
    for name, records in files.items():
        write_jsonl(out_dir / f"{name}.jsonl", records)
        checksums[name] = sha256_of(out_dir / f"{name}.jsonl")
    expected = dcfg.get("output_sha256") or {}
    mismatched = sorted(k for k, v in expected.items() if checksums.get(k) != v)
    (out_dir / "categories.json").write_text(
        json.dumps({"labels": labels, "descriptions": descriptions}, indent=2), encoding="utf-8",
        newline="\n",
    )
    (out_dir / "splits.json").write_text(json.dumps(split_of, indent=2), encoding="utf-8", newline="\n")

    stats = {
        "contracts_per_split": Counter(split_of.values()),
        "pool_classification": {s: len(v) for s, v in cls_by_split.items()},
        "pool_qa_positives": {s: len(v) for s, v in pos_by_split.items()},
        "skipped": dict(skipped),
        "sha256": checksums,
        "sha256_mismatch": mismatched,
        "files": {
            name: {
                "n": len(recs),
                "tasks": Counter(r["task"] for r in recs),
                "qa_present": Counter(str(r["present"]) for r in recs if r["task"] == "qa"),
                "neg_types": Counter(r["neg_type"] for r in recs if r.get("neg_type")),
                "max_tokens": max((r["n_tokens"] for r in recs), default=0),
                "n_categories": len({r.get("category") or r.get("target_label") for r in recs}),
            }
            for name, recs in files.items()
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8", newline="\n")
    if mismatched:
        print(f"WARNING: built files differ from configs/data.yaml output_sha256: {mismatched}. "
              "Results would not be comparable to the reference build.")
    return stats


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
