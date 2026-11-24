#!/usr/bin/env python3
"""Build and evaluate a 48-case phylogeny stress test."""

import argparse
import json
import os
from collections import Counter

from phylogeny_utils import (
    make_prompt,
    parse_newick,
    recompute_answer,
    tree_summary,
)


CATEGORY_ORDER = (
    "real_biology_and_paraphyly",
    "adversarial_topology_swap",
    "ood_tree_sizes",
    "branch_lengths_and_formatting",
)
EXPECTED_CATEGORY_COUNTS = {
    "real_biology_and_paraphyly": 16,
    "adversarial_topology_swap": 12,
    "ood_tree_sizes": 12,
    "branch_lengths_and_formatting": 8,
}
TASKS = ("mrca", "sister", "monophyletic", "distance")


REAL_TREES = (
    (
        "Primates",
        "(((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;",
        (
            ("mrca", "What is the most recent common ancestor of Homo and Gorilla?"),
            ("sister", "What is the direct sister taxon of Homo?"),
            ("monophyletic", "Do Gorilla, Homo form a monophyletic clade?"),
            (
                "distance",
                "What is the topological edge distance between Homo and Pongo?",
            ),
        ),
    ),
    (
        "Carnivores",
        "(((Canis,Vulpes)Canidae,(Ursus,Ailuropoda)Ursidae)Caniformia,"
        "(Felis,Panthera)Felidae)Carnivora;",
        (
            ("mrca", "What is the most recent common ancestor of Canis and Ursus?"),
            ("sister", "What is the direct sister taxon of Canis?"),
            ("monophyletic", "Do Canis, Ursus form a monophyletic clade?"),
            (
                "distance",
                "What is the topological edge distance between Canis and Felis?",
            ),
        ),
    ),
    (
        "Vertebrates",
        "(((Gallus,Alligator)Archosauria,Anolis)Sauropsida,"
        "(Mus,Ornithorhynchus)Mammalia)Amniota;",
        (
            ("mrca", "What is the most recent common ancestor of Gallus and Anolis?"),
            ("sister", "What is the direct sister taxon of Mus?"),
            ("monophyletic", "Do Alligator, Anolis form a monophyletic clade?"),
            (
                "distance",
                "What is the topological edge distance between Gallus and Mus?",
            ),
        ),
    ),
    (
        "LandPlants",
        "((Marchantia,Physcomitrium)Bryophyta,(Selaginella,"
        "(Pinus,Ginkgo)Gymnospermae)Tracheophyta)LandPlants;",
        (
            ("mrca", "What is the most recent common ancestor of Pinus and Ginkgo?"),
            ("sister", "What is the direct sister taxon of Pinus?"),
            ("monophyletic", "Do Pinus, Selaginella form a monophyletic clade?"),
            ("monophyletic", "Do Pinus, Ginkgo form a monophyletic clade?"),
        ),
    ),
)

ADVERSARIAL_TREES = (
    (
        "AdversarialA",
        "(((Homo,Hylobates)MixA,(Pan,Gorilla)MixB)MixC,Mus)MixRoot;",
        (
            ("mrca", "What is the most recent common ancestor of Homo and Pan?"),
            ("sister", "What is the direct sister taxon of Homo?"),
            ("monophyletic", "Do Homo, Hylobates form a monophyletic clade?"),
            ("distance", "What is the topological edge distance between Homo and Mus?"),
        ),
    ),
    (
        "AdversarialB",
        "(((Mus,Gallus)MixD,(Alligator,Anolis)MixE)MixF,"
        "(Ornithorhynchus,Pongo)MixG)MixRoot2;",
        (
            ("mrca", "What is the most recent common ancestor of Mus and Anolis?"),
            ("sister", "What is the direct sister taxon of Mus?"),
            ("monophyletic", "Do Mus, Alligator form a monophyletic clade?"),
            (
                "distance",
                "What is the topological edge distance between Mus and Pongo?",
            ),
        ),
    ),
    (
        "AdversarialC",
        "(((Homo,Pan)MixH,(Gorilla,Pongo)MixI)MixJ,"
        "(Hylobates,(Alligator,Anolis)MixK)MixL)MixRoot3;",
        (
            ("mrca", "What is the most recent common ancestor of Homo and Gorilla?"),
            ("sister", "What is the direct sister taxon of Homo?"),
            ("monophyletic", "Do Homo, Gorilla form a monophyletic clade?"),
            (
                "distance",
                "What is the topological edge distance between Homo and Hylobates?",
            ),
        ),
    ),
)

OOD_TREES = (
    (
        "OOD4",
        "((Zephyria,Lunaris)Novaclade,(Terrax,Marina)Oceanidae)Gaiaformes;",
        (
            ("mrca", "What is the most recent common ancestor of Zephyria and Lunaris?"),
            ("sister", "What is the direct sister taxon of Zephyria?"),
            ("monophyletic", "Do Zephyria, Lunaris form a monophyletic clade?"),
            (
                "distance",
                "What is the topological edge distance between Zephyria and Terrax?",
            ),
        ),
    ),
    (
        "OOD8A",
        "(((Aurorix,Borealis)Skycladidae,(Cryonix,Delphis)Polaridae)Atmosphaeria,"
        "((Emberix,Flamara)Ignidae,(Glacius,Helionix)Cryonidae)Pyrotheria)Megalocladia;",
        (
            (
                "mrca",
                "What is the most recent common ancestor of Aurorix and Borealis?",
            ),
            ("sister", "What is the direct sister taxon of Aurorix?"),
            ("monophyletic", "Do Aurorix, Borealis form a monophyletic clade?"),
            (
                "distance",
                "What is the topological edge distance between Aurorix and Delphis?",
            ),
        ),
    ),
    (
        "OOD8B",
        "(((Nimbus,Opalix)Nebulidae,(Quasaris,Radianta)Stellaridae)Galaxiformes,"
        "((Sylvara,Terranox)Silvidae,(Umbrax,Vulcanis)Igniformia)Terraformia)Cosmoclada;",
        (
            ("mrca", "What is the most recent common ancestor of Nimbus and Opalix?"),
            ("sister", "What is the direct sister taxon of Nimbus?"),
            ("monophyletic", "Do Nimbus, Terranox form a monophyletic clade?"),
            ("distance", "What is the topological edge distance between Nimbus and Opalix?"),
        ),
    ),
)

BRANCH_LENGTH_TREES = (
    (
        "BranchPrimates",
        "( ( (Homo:0.006, Pan:1.2e-2) Hominini:0.03, Gorilla:1.5e-3 ) "
        "Homininae:0.01, (Pongo:0.004, Hylobates:0.005) Pongidae:0.02 ) "
        "Hominoidea:0.1 ;",
        (
            ("mrca", "What is the most recent common ancestor of Homo and Gorilla?"),
            ("sister", "What is the direct sister taxon of Homo?"),
            ("monophyletic", "Do Homo, Pan form a monophyletic clade?"),
            (
                "distance",
                "What is the topological edge distance between Homo and Pongo?",
            ),
        ),
    ),
    (
        "BranchCarnivores",
        "( ( (Canis:0.006, Vulpes:1.2e-2) Canidae:0.03, "
        "(Ursus:1.5e-3, Ailuropoda:0.004) Ursidae:0.02 ) Caniformia:0.01, "
        "(Felis:0.005, Panthera:0.006) Felidae:0.02 ) Carnivora:0.1 ;",
        (
            ("mrca", "What is the most recent common ancestor of Canis and Ursus?"),
            ("sister", "What is the direct sister taxon of Canis?"),
            ("monophyletic", "Do Canis, Ursus form a monophyletic clade?"),
            (
                "distance",
                "What is the topological edge distance between Canis and Felis?",
            ),
        ),
    ),
)


def _add_tree_cases(
    items,
    category,
    tree_name,
    raw_newick,
    cases,
    expect_branch_lengths=False,
):
    summary = tree_summary(raw_newick, auto_label_internal=True)
    normalized = summary["normalized_newick"]
    root = parse_newick(normalized)

    if expect_branch_lengths and not summary["metadata"]["branch_lengths_stripped"]:
        raise AssertionError(
            "%s: expected branch lengths to be stripped" % tree_name
        )

    for task, question in cases:
        expected = recompute_answer(task, question, root)
        if expected == "":
            raise AssertionError("%s: empty expected answer" % tree_name)
        items.append(
            {
                "category": category,
                "tree_name": tree_name,
                "raw_newick": raw_newick,
                "normalized_newick": normalized,
                "tree": normalized,
                "metadata": summary["metadata"],
                "leaves": summary["leaves"],
                "task": task,
                "question": question,
                "expected": expected,
                "prompt": make_prompt(normalized, task, question),
            }
        )


def build_items():
    items = []

    for tree_name, raw_newick, cases in REAL_TREES:
        _add_tree_cases(
            items,
            "real_biology_and_paraphyly",
            tree_name,
            raw_newick,
            cases,
        )
    for tree_name, raw_newick, cases in ADVERSARIAL_TREES:
        _add_tree_cases(
            items,
            "adversarial_topology_swap",
            tree_name,
            raw_newick,
            cases,
        )
    for tree_name, raw_newick, cases in OOD_TREES:
        _add_tree_cases(items, "ood_tree_sizes", tree_name, raw_newick, cases)
    for tree_name, raw_newick, cases in BRANCH_LENGTH_TREES:
        _add_tree_cases(
            items,
            "branch_lengths_and_formatting",
            tree_name,
            raw_newick,
            cases,
            expect_branch_lengths=True,
        )

    if len(items) != 48:
        raise AssertionError("Expected 48 stress-test cases, found %d" % len(items))

    category_counts = Counter(item["category"] for item in items)
    for category, expected_count in EXPECTED_CATEGORY_COUNTS.items():
        if category_counts[category] != expected_count:
            raise AssertionError(
                "%s: expected %d cases, found %d"
                % (category, expected_count, category_counts[category])
            )
    if set(category_counts) != set(CATEGORY_ORDER):
        raise AssertionError("Unexpected stress-test categories")

    for category in CATEGORY_ORDER:
        tasks = {item["task"] for item in items if item["category"] == category}
        if tasks != set(TASKS):
            raise AssertionError("%s does not cover all four tasks" % category)

    required_expectations = {
        "What is the most recent common ancestor of Homo and Gorilla?": "Homininae",
        "Do Gorilla, Homo form a monophyletic clade?": "No",
        "Do Canis, Ursus form a monophyletic clade?": "No",
        "Do Alligator, Anolis form a monophyletic clade?": "No",
        "Do Pinus, Selaginella form a monophyletic clade?": "No",
        "Do Pinus, Ginkgo form a monophyletic clade?": "Yes",
    }
    by_question = {item["question"]: item for item in items}
    for question, expected in required_expectations.items():
        if question not in by_question:
            raise AssertionError("Missing required stress case: %s" % question)
        actual = by_question[question]["expected"]
        if actual != expected:
            raise AssertionError(
                "Required case %r expected %r, got %r"
                % (question, expected, actual)
            )

    return items


def print_verification(items):
    category_counts = Counter(item["category"] for item in items)
    task_counts = Counter(item["task"] for item in items)
    branch_cases = [
        item
        for item in items
        if item["category"] == "branch_lengths_and_formatting"
    ]
    stripped = sum(
        1
        for item in branch_cases
        if item["metadata"]["branch_lengths_stripped"]
    )

    print("Verified %d stress-test cases." % len(items))
    for category in CATEGORY_ORDER:
        print("  %-34s %d" % (category, category_counts[category]))
    print(
        "Task counts: %s"
        % ", ".join("%s=%d" % (task, task_counts[task]) for task in TASKS)
    )
    print(
        "Branch-length cases normalized: %d/%d"
        % (stripped, len(branch_cases))
    )
    print("All prompts were built from normalized Newick strings.")


def _first_nonempty_line(text):
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _load_causal_lm(model_name, dtype):
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError as exc:
        if "dtype" not in str(exc).lower():
            raise
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def _generate_answers(model, tokenizer, prompts, device, max_new_tokens):
    import torch

    answers = []
    for prompt in prompts:
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            chat_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generate_kwargs)
        completion = tokenizer.decode(
            output_ids[0, input_length:],
            skip_special_tokens=True,
        )
        answers.append(_first_nonempty_line(completion))
    return answers


def _accuracy(records, model_key):
    correct = sum(1 for record in records if record[model_key + "_correct"])
    total = len(records)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def _group_accuracy(records, group_key, model_key):
    groups = {}
    for record in records:
        groups.setdefault(record[group_key], []).append(record)
    return {
        name: _accuracy(group_records, model_key)
        for name, group_records in groups.items()
    }


def _clipped(text, width):
    text = str(text)
    if len(text) <= width:
        return text
    return text[:max(0, width - 3)] + "..."


def evaluate_models(items, args):
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel

    if not os.path.isdir(args.adapter_dir):
        raise FileNotFoundError("LoRA adapter directory not found: %s" % args.adapter_dir)

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

    base_model = _load_causal_lm(args.model_name, dtype)
    base_model.to(device)
    base_model.eval()
    if hasattr(base_model, "generation_config"):
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    prompts = [item["prompt"] for item in items]
    base_predictions = None
    if not args.skip_base:
        base_predictions = _generate_answers(
            base_model, tokenizer, prompts, device, args.max_new_tokens
        )

    lora_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    lora_model.to(device)
    lora_model.eval()
    if hasattr(lora_model, "generation_config"):
        lora_model.generation_config.pad_token_id = tokenizer.pad_token_id
    lora_predictions = _generate_answers(
        lora_model, tokenizer, prompts, device, args.max_new_tokens
    )

    results = []
    for index, item in enumerate(items):
        lora_prediction = lora_predictions[index]
        result = dict(item)
        result["lora_prediction"] = lora_prediction
        result["lora_correct"] = (
            lora_prediction.strip().lower() == item["expected"].strip().lower()
        )
        if base_predictions is None:
            result["base_prediction"] = None
            result["base_correct"] = None
        else:
            base_prediction = base_predictions[index]
            result["base_prediction"] = base_prediction
            result["base_correct"] = (
                base_prediction.strip().lower()
                == item["expected"].strip().lower()
            )
        results.append(result)

    overall = {"lora": _accuracy(results, "lora")}
    if not args.skip_base:
        overall["base"] = _accuracy(results, "base")

    per_category = {}
    for category in CATEGORY_ORDER:
        category_records = [
            record for record in results if record["category"] == category
        ]
        per_category[category] = {
            "lora": _accuracy(category_records, "lora")
        }
        if not args.skip_base:
            per_category[category]["base"] = _accuracy(
                category_records, "base"
            )

    per_task = {}
    for task in TASKS:
        task_records = [record for record in results if record["task"] == task]
        per_task[task] = {"lora": _accuracy(task_records, "lora")}
        if not args.skip_base:
            per_task[task]["base"] = _accuracy(task_records, "base")

    metrics = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "max_new_tokens": args.max_new_tokens,
        "skip_base": args.skip_base,
        "overall": overall,
        "per_category": per_category,
        "per_task": per_task,
        "results": results,
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("=" * 90)
    print("Phylogeny stress-test summary")
    print("=" * 90)
    lora_overall = overall["lora"]
    print(
        "LoRA overall: %d/%d (%.2f%%)"
        % (
            lora_overall["correct"],
            lora_overall["total"],
            100.0 * lora_overall["accuracy"],
        )
    )
    if not args.skip_base:
        base_overall = overall["base"]
        print(
            "Base overall: %d/%d (%.2f%%)"
            % (
                base_overall["correct"],
                base_overall["total"],
                100.0 * base_overall["accuracy"],
            )
        )

    print("-" * 90)
    print("%-36s %-18s %-18s" % ("Category", "LoRA", "Base"))
    print("-" * 90)
    for category in CATEGORY_ORDER:
        category_metrics = per_category[category]
        lora_stats = category_metrics["lora"]
        lora_text = "%d/%d (%.2f%%)" % (
            lora_stats["correct"],
            lora_stats["total"],
            100.0 * lora_stats["accuracy"],
        )
        if args.skip_base:
            base_text = "skipped"
        else:
            base_stats = category_metrics["base"]
            base_text = "%d/%d (%.2f%%)" % (
                base_stats["correct"],
                base_stats["total"],
                100.0 * base_stats["accuracy"],
            )
        print("%-36s %-18s %-18s" % (category, lora_text, base_text))

    print("-" * 90)
    print("%-36s %-18s %-18s" % ("Task", "LoRA", "Base"))
    print("-" * 90)
    for task in TASKS:
        task_metrics = per_task[task]
        lora_stats = task_metrics["lora"]
        lora_text = "%d/%d (%.2f%%)" % (
            lora_stats["correct"],
            lora_stats["total"],
            100.0 * lora_stats["accuracy"],
        )
        if args.skip_base:
            base_text = "skipped"
        else:
            base_stats = task_metrics["base"]
            base_text = "%d/%d (%.2f%%)" % (
                base_stats["correct"],
                base_stats["total"],
                100.0 * base_stats["accuracy"],
            )
        print("%-36s %-18s %-18s" % (task, lora_text, base_text))

    print("-" * 90)
    for model_key in ("lora", "base"):
        if model_key == "base" and args.skip_base:
            continue
        failures = [
            result for result in results if not result[model_key + "_correct"]
        ]
        print("%s failures: %d" % (model_key.capitalize(), len(failures)))
        for result in failures:
            print(
                "  [%s] %s | %s | expected=%s | predicted=%s"
                % (
                    result["category"],
                    result["tree_name"],
                    result["task"],
                    result["expected"],
                    _clipped(result[model_key + "_prediction"], 60),
                )
            )
    print("Saved full stress-test results to %s" % args.output_json)
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify and evaluate a 48-case phylogeny stress test."
    )
    parser.add_argument(
        "--model_name",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument(
        "--adapter_dir",
        default="outputs/phylogeny-lora-v2",
    )
    parser.add_argument(
        "--output_json",
        default="results/stress_test_v2.json",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args(argv)

    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    items = build_items()

    if args.verify_only:
        print_verification(items)
        return

    evaluate_models(items, args)


if __name__ == "__main__":
    main()
