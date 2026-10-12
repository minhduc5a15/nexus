"""Render one NEXUS SFT record and expose the assistant-only loss mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from scripts.audit_sft_data import DEFAULT_DATASET, audit_dataset, materialize_case
from scripts.eval_proposals import strict_json


def build_assistant_only_labels(record: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Use prefix tokenization so only the final assistant turn receives labels."""

    messages = record["messages"]
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("Record must end with one assistant target")
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tools=record["tools"],
        add_generation_prompt=True,
        tokenize=True,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tools=record["tools"],
        add_generation_prompt=False,
        tokenize=True,
    )
    if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
        raise ValueError("Tokenizer must return flat token ID lists")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Assistant target is not a suffix of the rendered prompt")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not labels or all(label == -100 for label in labels):
        raise ValueError("Assistant target has no supervised tokens")
    return {
        "input_ids": full_ids,
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "supervised_tokens": len(full_ids) - len(prompt_ids),
    }


def inspect_record(record: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    encoded = build_assistant_only_labels(record, tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    if len(tokens) != len(encoded["input_ids"]):
        raise ValueError("Tokenizer returned a mismatched token list")
    rows = [
        {
            "index": index,
            "loss": label != -100,
            "token_id": token_id,
            "token": token,
        }
        for index, (token_id, label, token) in enumerate(
            zip(encoded["input_ids"], encoded["labels"], tokens)
        )
    ]
    rendered = tokenizer.apply_chat_template(
        record["messages"],
        tools=record["tools"],
        add_generation_prompt=False,
        tokenize=False,
    )
    return {
        "id": record["id"],
        "prompt_tokens": encoded["prompt_tokens"],
        "supervised_tokens": encoded["supervised_tokens"],
        "rendered": rendered,
        "tokens": rows,
    }


def audit_selected_dataset(path: Path) -> dict[str, Any]:
    """Dispatch to the dataset-specific offline audit before tokenization."""

    document = strict_json(path.read_bytes())
    dataset_id = document.get("dataset_id")
    if dataset_id == "nexus-sft-sanity-v1":
        return audit_dataset(path)
    if dataset_id == "nexus-sft-pilot-v1":
        from scripts.audit_sft_pilot import audit_dataset as audit_pilot_dataset

        return audit_pilot_dataset(path)
    if dataset_id == "nexus-sft-candidate-v2":
        from scripts.audit_sft_candidate_v2 import audit_dataset as audit_candidate_dataset

        return audit_candidate_dataset(path)
    raise ValueError(f"Unsupported SFT dataset: {dataset_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--case", default="sft_create_delimited")
    parser.add_argument("--all-cases", action="store_true", help="Validate and summarize every record")
    parser.add_argument("--summary-only", action="store_true", help="Omit per-case token rows with --all-cases")
    parser.add_argument("--tokenizer", help="Local path or Hugging Face model ID")
    parser.add_argument("--revision", help="Tokenizer revision; defaults to the dataset lock")
    parser.add_argument("--all-tokens", action="store_true", help="Include prompt tokens as well as supervised tokens")
    args = parser.parse_args()

    audit_selected_dataset(args.dataset.resolve())
    document = strict_json(args.dataset.read_bytes())
    try:
        case = next(case for case in document["cases"] if case["id"] == args.case)
    except StopIteration:
        parser.error(f"Unknown case: {args.case}")
    if not args.tokenizer:
        print(json.dumps({
            "id": case["id"],
            "behavior": case["behavior"],
            "messages": case["messages"],
            "note": "Pass --tokenizer to render the locked Qwen chat template and token loss mask.",
        }, ensure_ascii=False, indent=2))
        return 0
    try:
        from transformers import AutoTokenizer
    except ImportError:
        parser.error("transformers is required for --tokenizer")
    revision = args.revision or document["base_model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=revision)
    if args.all_cases:
        rows = []
        for current in document["cases"]:
            record = materialize_case(
                current, prompt_version=document["contract"]["prompt_version"]
            )
            inspected = inspect_record(record, tokenizer)
            total = inspected["prompt_tokens"] + inspected["supervised_tokens"]
            if total > 2048:
                parser.error(f"{current['id']} exceeds 2048 tokens: {total}")
            rows.append({
                "id": current["id"],
                "behavior": current["behavior"],
                "prompt_tokens": inspected["prompt_tokens"],
                "supervised_tokens": inspected["supervised_tokens"],
                "total_tokens": total,
            })
        totals = [row["total_tokens"] for row in rows]
        prompt_totals = [row["prompt_tokens"] for row in rows]
        supervised_totals = [row["supervised_tokens"] for row in rows]
        maximum_total = max(totals)
        output = {
            "model": document["base_model"],
            "count": len(rows),
            "minimum_total_tokens": min(totals),
            "median_total_tokens": median(totals),
            "maximum_total_tokens": maximum_total,
            "maximum_total_case_ids": [row["id"] for row in rows if row["total_tokens"] == maximum_total],
            "prompt_tokens": {
                "minimum": min(prompt_totals),
                "median": median(prompt_totals),
                "maximum": max(prompt_totals),
            },
            "supervised_tokens": {
                "minimum": min(supervised_totals),
                "median": median(supervised_totals),
                "maximum": max(supervised_totals),
                "sum": sum(supervised_totals),
            },
            "assistant_only_mask": "verified_by_exact_token_prefix",
        }
        if not args.summary_only:
            output["cases"] = rows
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    record = materialize_case(case, prompt_version=document["contract"]["prompt_version"])
    result = inspect_record(record, tokenizer)
    if not args.all_tokens:
        result["tokens"] = [row for row in result["tokens"] if row["loss"]]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
