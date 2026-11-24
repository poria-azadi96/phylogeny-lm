#!/usr/bin/env python3
"""Evaluate a causal language model (optionally with a PEFT adapter) on the
synthetic phylogeny test set."""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TASKS = ("mrca", "sister", "monophyletic", "distance")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: invalid JSON: %s" % (path, line_number, exc))
    if not records:
        raise ValueError("No records found in %s" % path)
    return records


def first_nonempty_line(text):
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a phylogeny QA model.")
    parser.add_argument(
        "--model_name",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--adapter_dir",
        default=None,
        help="Optional PEFT adapter directory.",
    )
    parser.add_argument("--test_file", default="data/test.jsonl")
    parser.add_argument("--output_json", default="results/eval.json")
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args(argv)

    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    records = load_jsonl(args.test_file)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a pad or eos token")
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(device)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    predictions = []
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start:start + args.batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for record in batch_records
        ]
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generate_kwargs)
        generated_ids = output_ids[:, input_length:]
        completions = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        for record, completion in zip(batch_records, completions):
            prediction = first_nonempty_line(completion)
            expected = str(record["answer"]).strip()
            correct = prediction.strip().lower() == expected.lower()
            predictions.append(
                {
                    "id": record.get("id"),
                    "task": record["task"],
                    "tree": record.get("tree"),
                    "prompt": record["prompt"],
                    "expected": expected,
                    "completion": completion,
                    "prediction": prediction,
                    "correct": correct,
                }
            )

    total = len(predictions)
    correct_total = sum(1 for item in predictions if item["correct"])
    overall_accuracy = correct_total / total if total else 0.0

    per_task = {
        task: {"correct": 0, "total": 0, "accuracy": 0.0} for task in TASKS
    }
    for item in predictions:
        task = item["task"]
        if task not in per_task:
            per_task[task] = {"correct": 0, "total": 0, "accuracy": 0.0}
        per_task[task]["total"] += 1
        if item["correct"]:
            per_task[task]["correct"] += 1
    for stats in per_task.values():
        stats["accuracy"] = (
            stats["correct"] / stats["total"] if stats["total"] else 0.0
        )

    failures = [item for item in predictions if not item["correct"]]
    successes = [item for item in predictions if item["correct"]]
    samples = (failures + successes)[:10]

    metrics = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "test_file": args.test_file,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "overall": {
            "correct": correct_total,
            "total": total,
            "accuracy": overall_accuracy,
        },
        "per_task": per_task,
        "num_failures": len(failures),
        "samples": samples,
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("=" * 68)
    print("Evaluation summary")
    print("=" * 68)
    print("Model:       %s" % args.model_name)
    print("Adapter:     %s" % (args.adapter_dir or "none"))
    print("Test file:   %s" % args.test_file)
    print("Examples:    %d" % total)
    print("Correct:     %d" % correct_total)
    print("Accuracy:    %.2f%%" % (100.0 * overall_accuracy))
    print("-" * 68)
    print("%-18s %10s %8s %12s" % ("Task", "Correct", "Total", "Accuracy"))
    print("-" * 68)
    for task in TASKS:
        stats = per_task[task]
        print(
            "%-18s %10d %8d %11.2f%%"
            % (
                task,
                stats["correct"],
                stats["total"],
                100.0 * stats["accuracy"],
            )
        )
    print("-" * 68)
    print("Saved metrics and %d samples to %s" % (len(samples), args.output_json))


if __name__ == "__main__":
    main()
