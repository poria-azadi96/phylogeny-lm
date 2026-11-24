#!/usr/bin/env python3
"""Evaluate base and LoRA models on unseen real biological phylogenies."""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_dataset import make_prompt, parse_newick, recompute_answer


TREES = (
    (
        "Primates",
        "(((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;",
    ),
    (
        "Carnivores",
        "(((Canis,Vulpes)Canidae,(Ursus,Ailuropoda)Ursidae)Caniformia,"
        "(Felis,Panthera)Felidae)Carnivora;",
    ),
    (
        "Vertebrates",
        "(((Gallus,Alligator)Archosauria,Anolis)Sauropsida,"
        "(Mus,Ornithorhynchus)Mammalia)Amniota;",
    ),
)

QUESTIONS = {
    "Primates": (
        ("mrca", "What is the most recent common ancestor of Homo and Gorilla?"),
        ("sister", "What is the direct sister taxon of Homo?"),
        ("monophyletic", "Do Homo, Pan form a monophyletic clade?"),
        (
            "distance",
            "What is the topological edge distance between Homo and Pongo?",
        ),
    ),
    "Carnivores": (
        ("mrca", "What is the most recent common ancestor of Canis and Ursus?"),
        ("sister", "What is the direct sister taxon of Canis?"),
        ("monophyletic", "Do Felis, Panthera form a monophyletic clade?"),
        (
            "distance",
            "What is the topological edge distance between Canis and Felis?",
        ),
    ),
    "Vertebrates": (
        ("mrca", "What is the most recent common ancestor of Gallus and Anolis?"),
        ("sister", "What is the direct sister taxon of Mus?"),
        (
            "monophyletic",
            "Do Mus, Ornithorhynchus form a monophyletic clade?",
        ),
        (
            "distance",
            "What is the topological edge distance between Gallus and Mus?",
        ),
    ),
}


def first_nonempty_line(text):
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def load_causal_lm(model_name, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError as exc:
        if "dtype" not in str(exc).lower():
            raise
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def build_items():
    items = []
    for tree_name, tree in TREES:
        root = parse_newick(tree)
        for task, question in QUESTIONS[tree_name]:
            expected = recompute_answer(task, question, root)
            items.append(
                {
                    "tree_name": tree_name,
                    "tree": tree,
                    "task": task,
                    "question": question,
                    "expected": expected,
                    "prompt": make_prompt(tree, task, question),
                }
            )
    if len(items) != 12:
        raise AssertionError("Expected exactly 12 unseen biological questions")
    return items


def generate_answers(model, tokenizer, prompts, device, max_new_tokens):
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
        answers.append(first_nonempty_line(completion))
    return answers


def clipped(text, width):
    text = str(text)
    if len(text) <= width:
        return text
    return text[:max(0, width - 3)] + "..."


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Demo base vs LoRA generalization on unseen biological trees."
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
        default="results/unseen_bio_demo.json",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args(argv)

    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isdir(args.adapter_dir):
        raise FileNotFoundError("LoRA adapter directory not found: %s" % args.adapter_dir)

    items = build_items()

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

    base_model = load_causal_lm(args.model_name, dtype)
    base_model.to(device)
    base_model.eval()
    if hasattr(base_model, "generation_config"):
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    prompts = [item["prompt"] for item in items]
    base_predictions = generate_answers(
        base_model, tokenizer, prompts, device, args.max_new_tokens
    )

    from peft import PeftModel

    lora_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    lora_model.to(device)
    lora_model.eval()
    if hasattr(lora_model, "generation_config"):
        lora_model.generation_config.pad_token_id = tokenizer.pad_token_id
    lora_predictions = generate_answers(
        lora_model, tokenizer, prompts, device, args.max_new_tokens
    )

    results = []
    for item, base_prediction, lora_prediction in zip(
        items, base_predictions, lora_predictions
    ):
        expected = item["expected"]
        lora_ok = lora_prediction.strip().lower() == expected.strip().lower()
        base_ok = base_prediction.strip().lower() == expected.strip().lower()
        results.append(
            {
                "tree": item["tree_name"],
                "newick": item["tree"],
                "task": item["task"],
                "question": item["question"],
                "prompt": item["prompt"],
                "expected": expected,
                "base_prediction": base_prediction,
                "lora_prediction": lora_prediction,
                "base_ok": base_ok,
                "lora_ok": lora_ok,
            }
        )

    total = len(results)
    base_correct = sum(1 for result in results if result["base_ok"])
    lora_correct = sum(1 for result in results if result["lora_ok"])

    output = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "max_new_tokens": args.max_new_tokens,
        "trees": [{"name": name, "newick": tree} for name, tree in TREES],
        "results": results,
        "overall": {
            "total": total,
            "base_correct": base_correct,
            "base_accuracy": base_correct / total,
            "lora_correct": lora_correct,
            "lora_accuracy": lora_correct / total,
        },
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("=" * 112)
    print("Unseen real biological phylogenies: Base vs LoRA")
    print("=" * 112)
    print(
        "%-12s %-14s %-14s %-22s %-22s %-8s"
        % ("Tree", "Task", "Expected", "Base", "LoRA", "LoRA_OK")
    )
    print("-" * 112)
    for result in results:
        print(
            "%-12s %-14s %-14s %-22s %-22s %-8s"
            % (
                result["tree"],
                result["task"],
                clipped(result["expected"], 14),
                clipped(result["base_prediction"], 22),
                clipped(result["lora_prediction"], 22),
                "yes" if result["lora_ok"] else "no",
            )
        )
    print("-" * 112)
    print(
        "Overall on %d unseen biological questions: Base %d/%d (%.2f%%), LoRA %d/%d (%.2f%%)"
        % (
            total,
            base_correct,
            total,
            100.0 * base_correct / total,
            lora_correct,
            total,
            100.0 * lora_correct / total,
        )
    )
    print("Saved full results to %s" % args.output_json)


if __name__ == "__main__":
    main()
