#!/usr/bin/env python3
"""Verify a real 16S rRNA K2P+Neighbor-Joining reconstruction and evaluate
Base vs. Fine-Tuned LoRA on tree questions plus an honest raw-FASTA control."""

import argparse
import json
import os
import re
from collections import Counter

from bio_phylo import parse_fasta, reconstruct_tree_from_fasta
from phylogeny_utils import (
    cherries,
    make_prompt,
    parse_newick,
    recompute_answer,
    solve_monophyletic,
)


BACTERIA = (
    "Escherichia_coli",
    "Pseudomonas_aeruginosa",
    "Bacillus_subtilis",
    "Staphylococcus_aureus",
)
ARCHAEA = (
    "Halobacterium_salinarum",
    "Methanocaldococcus_jannaschii",
    "Pyrococcus_furiosus",
    "Sulfolobus_acidocaldarius",
)
TASKS = ("mrca", "sister", "monophyletic", "distance")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify and evaluate 16S rRNA K2P+NJ phylogeny questions."
    )
    parser.add_argument(
        "--fasta",
        default="data/16s_rrna_archaea_bacteria.fasta",
    )
    parser.add_argument(
        "--model_name",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument(
        "--adapter_dir",
        default="model/phylogeny-lora",
    )
    parser.add_argument(
        "--output_json",
        default="results/16s_rrna_eval.json",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args(argv)

    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")
    return args


def _read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _print_distance_matrix(taxa, matrix):
    print("K2P distance matrix:")
    for index, taxon in enumerate(taxa):
        print("  %d = %s" % (index, taxon))
    print("      " + "".join("%9d" % index for index in range(len(taxa))))
    for index, row in enumerate(matrix):
        print("%5d " % index + "".join("%9.4f" % value for value in row))


def _biological_validation(root):
    checks = (
        ("Bacteria (4 taxa)", list(BACTERIA), "Yes"),
        ("Archaea (4 taxa)", list(ARCHAEA), "Yes"),
        (
            "Gammaproteobacteria (Escherichia_coli, Pseudomonas_aeruginosa)",
            ["Escherichia_coli", "Pseudomonas_aeruginosa"],
            "Yes",
        ),
        (
            "Bacillota (Bacillus_subtilis, Staphylococcus_aureus)",
            ["Bacillus_subtilis", "Staphylococcus_aureus"],
            "Yes",
        ),
    )
    results = []
    for name, taxa, expected in checks:
        actual = solve_monophyletic(root, taxa)
        results.append(
            {
                "name": name,
                "taxa": taxa,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )
    return {
        "checks": results,
        "all_passed": all(result["passed"] for result in results),
    }


def _verified_answer(task, question, normalized_newick):
    expected = recompute_answer(task, question, parse_newick(normalized_newick))
    repeated = recompute_answer(task, question, parse_newick(normalized_newick))
    if expected != repeated:
        raise AssertionError("Non-deterministic ground truth for %r" % question)
    if expected == "":
        raise AssertionError("Empty ground truth for %r" % question)
    return expected


def _question_record(index, task, category, question, expected, normalized):
    return {
        "id": "B%02d" % index,
        "task": task,
        "category": category,
        "question": question,
        "expected": expected,
        "normalized_newick": normalized,
        "prompt": make_prompt(normalized, task, question),
    }


def build_tree_questions(reconstruction):
    normalized = reconstruction["normalized_newick"]
    root = parse_newick(normalized)
    records = []

    mrca_pairs = (
        (
            "Escherichia_coli",
            "Pseudomonas_aeruginosa",
            "intra-domain",
        ),
        (
            "Bacillus_subtilis",
            "Staphylococcus_aureus",
            "intra-domain",
        ),
        (
            "Escherichia_coli",
            "Halobacterium_salinarum",
            "cross-domain",
        ),
        (
            "Bacillus_subtilis",
            "Sulfolobus_acidocaldarius",
            "cross-domain",
        ),
    )
    for taxon_a, taxon_b, category in mrca_pairs:
        question = (
            "What is the most recent common ancestor of %s and %s?"
            % (taxon_a, taxon_b)
        )
        expected = _verified_answer("mrca", question, normalized)
        records.append(
            _question_record(
                len(records) + 1,
                "mrca",
                category,
                question,
                expected,
                normalized,
            )
        )

    cherry_leaves = cherries(root)
    bacteria_cherries = [taxon for taxon in cherry_leaves if taxon in BACTERIA]
    archaea_cherries = [taxon for taxon in cherry_leaves if taxon in ARCHAEA]
    sister_subjects = bacteria_cherries[:2] + archaea_cherries[:2]
    if len(sister_subjects) < 4:
        sister_subjects = cherry_leaves[:4]
    if len(sister_subjects) < 4:
        raise ValueError(
            "Need at least 4 cherry leaves for sister questions, found %d"
            % len(cherry_leaves)
        )
    for taxon in sister_subjects:
        question = "What is the direct sister taxon of %s?" % taxon
        expected = _verified_answer("sister", question, normalized)
        records.append(
            _question_record(
                len(records) + 1,
                "sister",
                "cherry-leaf",
                question,
                expected,
                normalized,
            )
        )

    positive_candidates = (
        ["Escherichia_coli", "Pseudomonas_aeruginosa"],
        ["Bacillus_subtilis", "Staphylococcus_aureus"],
        list(BACTERIA),
        list(ARCHAEA),
    )
    positives = []
    for taxa in positive_candidates:
        question = "Do %s form a monophyletic clade?" % ", ".join(taxa)
        expected = _verified_answer("monophyletic", question, normalized)
        if expected == "Yes":
            positives.append((taxa, question, expected))
        if len(positives) == 2:
            break
    if len(positives) < 2:
        raise AssertionError("Could not find two intra-domain positive clades")

    paraphyly_traps = (
        ["Escherichia_coli", "Halobacterium_salinarum"],
        ["Bacillus_subtilis", "Sulfolobus_acidocaldarius"],
    )
    for taxa in paraphyly_traps:
        question = "Do %s form a monophyletic clade?" % ", ".join(taxa)
        expected = _verified_answer("monophyletic", question, normalized)
        if expected != "No":
            raise AssertionError(
                "Expected cross-domain paraphyly trap to be No: %r" % question
            )
        records.append(
            _question_record(
                len(records) + 1,
                "monophyletic",
                "cross-domain-paraphyly-trap",
                question,
                expected,
                normalized,
            )
        )
    for taxa, question, expected in positives:
        records.append(
            _question_record(
                len(records) + 1,
                "monophyletic",
                "intra-domain-positive",
                question,
                expected,
                normalized,
            )
        )

    distance_pairs = (
        (
            "Escherichia_coli",
            "Pseudomonas_aeruginosa",
            "intra-domain",
        ),
        (
            "Bacillus_subtilis",
            "Staphylococcus_aureus",
            "intra-domain",
        ),
        (
            "Escherichia_coli",
            "Halobacterium_salinarum",
            "cross-domain",
        ),
        (
            "Bacillus_subtilis",
            "Sulfolobus_acidocaldarius",
            "cross-domain",
        ),
    )
    for taxon_a, taxon_b, category in distance_pairs:
        question = (
            "What is the topological edge distance between %s and %s?"
            % (taxon_a, taxon_b)
        )
        expected = _verified_answer("distance", question, normalized)
        records.append(
            _question_record(
                len(records) + 1,
                "distance",
                category,
                question,
                expected,
                normalized,
            )
        )

    if len(records) != 16:
        raise AssertionError("Expected 16 tree questions, found %d" % len(records))
    task_counts = Counter(record["task"] for record in records)
    for task in TASKS:
        if task_counts[task] != 4:
            raise AssertionError(
                "%s: expected 4 questions, found %d"
                % (task, task_counts[task])
            )
    return records


def _snippet(sequence, length=240):
    return sequence[: min(length, len(sequence))]


def _sequence_block(taxa, sequence_by_id, length=240):
    blocks = []
    for taxon in taxa:
        blocks.append(">%s\n%s" % (taxon, _snippet(sequence_by_id[taxon], length)))
    return "\n".join(blocks)


def build_raw_fasta_questions(reconstruction, fasta_text):
    records_by_id = {
        record["id"]: record["sequence"] for record in parse_fasta(fasta_text)
    }
    labels = reconstruction["taxa"]
    index = {label: position for position, label in enumerate(labels)}
    distances = reconstruction["distance_matrix"]
    normalized = reconstruction["normalized_newick"]
    root = parse_newick(normalized)

    def k2p(taxon_a, taxon_b):
        return distances[index[taxon_a]][index[taxon_b]]

    questions = []

    bacteria_pairs = [
        (a, b)
        for first, a in enumerate(BACTERIA)
        for b in BACTERIA[first + 1:]
    ]
    closest_pair = min(bacteria_pairs, key=lambda pair: (k2p(*pair), pair))
    expected_pair = "%s and %s" % closest_pair
    prompt = (
        "Raw 16S rRNA sequence snippets (240 bp each, no phylogenetic tree "
        "provided):\n%s\nQuestion: Which two species are most closely related? "
        "Answer with exactly one pair in the form 'Species_a and Species_b'."
        % _sequence_block(BACTERIA, records_by_id)
    )
    questions.append(
        {
            "id": "C1",
            "kind": "closest-pair",
            "taxa": list(BACTERIA),
            "question": prompt,
            "expected": expected_pair,
            "prompt": prompt,
        }
    )

    positive_pair = ["Escherichia_coli", "Pseudomonas_aeruginosa"]
    positive_question = "Do %s form a monophyletic clade?" % ", ".join(positive_pair)
    positive_expected = recompute_answer(
        "monophyletic", positive_question, root
    )
    if positive_expected != "Yes":
        raise AssertionError("Raw-control positive monophyly expected Yes")
    prompt = (
        "Raw 16S rRNA sequence snippets (240 bp each, no phylogenetic tree "
        "provided):\n%s\nQuestion: Do %s form a monophyletic clade? "
        "Answer Yes or No."
        % (_sequence_block(positive_pair, records_by_id), " and ".join(positive_pair))
    )
    questions.append(
        {
            "id": "C2",
            "kind": "raw-monophyly-positive",
            "taxa": positive_pair,
            "question": prompt,
            "expected": positive_expected,
            "prompt": prompt,
        }
    )

    trap_pair = ["Escherichia_coli", "Halobacterium_salinarum"]
    trap_question = "Do %s form a monophyletic clade?" % ", ".join(trap_pair)
    trap_expected = recompute_answer("monophyletic", trap_question, root)
    if trap_expected != "No":
        raise AssertionError("Raw-control paraphyly trap expected No")
    prompt = (
        "Raw 16S rRNA sequence snippets (240 bp each, no phylogenetic tree "
        "provided):\n%s\nQuestion: Do %s form a monophyletic clade? "
        "Answer Yes or No."
        % (_sequence_block(trap_pair, records_by_id), " and ".join(trap_pair))
    )
    questions.append(
        {
            "id": "C3",
            "kind": "raw-monophyly-paraphyly-trap",
            "taxa": trap_pair,
            "question": prompt,
            "expected": trap_expected,
            "prompt": prompt,
        }
    )

    subject = "Bacillus_subtilis"
    options = ["Staphylococcus_aureus", "Escherichia_coli"]
    closest_option = min(options, key=lambda taxon: (k2p(subject, taxon), taxon))
    option_taxa = [subject] + options
    prompt = (
        "Raw 16S rRNA sequence snippets (240 bp each, no phylogenetic tree "
        "provided):\n%s\nQuestion: Which species is the closest relative of %s: "
        "%s or %s? Answer with exactly one taxon name."
        % (
            _sequence_block(option_taxa, records_by_id),
            subject,
            options[0],
            options[1],
        )
    )
    questions.append(
        {
            "id": "C4",
            "kind": "closest-relative",
            "taxa": option_taxa,
            "question": prompt,
            "expected": closest_option,
            "prompt": prompt,
        }
    )

    if len(questions) != 4:
        raise AssertionError("Expected 4 raw-FASTA control questions")
    return questions


def _answers_match(prediction, expected, strict=True):
    if strict:
        return str(prediction).strip().lower() == str(expected).strip().lower()
    text_pred = re.sub(r"[^a-z0-9]+", " ", str(prediction).strip().lower())
    text_exp = re.sub(r"[^a-z0-9]+", " ", str(expected).strip().lower())
    return " ".join(text_pred.split()) == " ".join(text_exp.split())


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
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        )


def _load_models(args):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    if not os.path.isdir(args.adapter_dir):
        raise FileNotFoundError(
            "LoRA adapter directory not found: %s" % args.adapter_dir
        )

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
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, device


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


def _accuracy(records, prediction_key):
    correct = sum(
        1 for record in records if record[prediction_key + "_correct"]
    )
    total = len(records)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def _score_predictions(questions, predictions, prediction_key, strict=True):
    results = []
    for question, prediction in zip(questions, predictions):
        expected = question["expected"]
        results.append(
            {
                "id": question["id"],
                "task": question.get("task", question.get("kind")),
                "category": question.get("category", question.get("kind")),
                "question": question["question"],
                "expected": expected,
                "prediction": prediction,
                prediction_key + "_correct": _answers_match(
                    prediction, expected, strict=strict
                ),
            }
        )
    return results


def _per_task(results, prediction_key):
    grouped = {}
    for result in results:
        grouped.setdefault(result["task"], []).append(result)
    return {
        task: _accuracy(grouped[task], prediction_key)
        for task in sorted(grouped)
    }


def _clipped(text, width):
    text = str(text)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def evaluate_step_b(model, tokenizer, questions, device, max_new_tokens):
    prompts = [question["prompt"] for question in questions]

    with model.disable_adapter():
        base_predictions = _generate_answers(
            model, tokenizer, prompts, device, max_new_tokens
        )
    lora_predictions = _generate_answers(
        model, tokenizer, prompts, device, max_new_tokens
    )

    base_results = _score_predictions(
        questions, base_predictions, "base", strict=True
    )
    lora_results = _score_predictions(
        questions, lora_predictions, "lora", strict=True
    )

    return {
        "base": {
            "overall": _accuracy(base_results, "base"),
            "per_task": _per_task(base_results, "base"),
            "results": base_results,
        },
        "lora": {
            "overall": _accuracy(lora_results, "lora"),
            "per_task": _per_task(lora_results, "lora"),
            "results": lora_results,
        },
    }


def evaluate_step_c(model, tokenizer, questions, device, max_new_tokens):
    prompts = [question["prompt"] for question in questions]
    predictions = _generate_answers(
        model, tokenizer, prompts, device, max_new_tokens
    )
    results = _score_predictions(questions, predictions, "lora", strict=False)
    return {
        "overall": _accuracy(results, "lora"),
        "results": results,
    }


def print_step_b(questions, evaluation):
    print("=" * 90)
    print("Step B - 16 real 16S rRNA gene-tree questions")
    print("=" * 90)
    print(
        "Task counts: %s"
        % ", ".join(
            "%s=%d"
            % (task, sum(1 for question in questions if question["task"] == task))
            for task in TASKS
        )
    )
    for question in questions:
        print(
            "  [%s] %s -> %s"
            % (question["task"], question["question"], question["expected"])
        )
    if evaluation is None:
        print("All 16 ground truths verified with phylogeny_utils.")
        return

    for model_key in ("base", "lora"):
        stats = evaluation[model_key]["overall"]
        print(
            "%s overall: %d/%d (%.2f%%)"
            % (
                model_key.capitalize(),
                stats["correct"],
                stats["total"],
                100.0 * stats["accuracy"],
            )
        )
    print("-" * 90)
    print("%-18s %-18s %-18s" % ("Task", "Base", "LoRA"))
    print("-" * 90)
    for task in TASKS:
        cells = []
        for model_key in ("base", "lora"):
            stats = evaluation[model_key]["per_task"][task]
            cells.append(
                "%d/%d (%.2f%%)"
                % (
                    stats["correct"],
                    stats["total"],
                    100.0 * stats["accuracy"],
                )
            )
        print("%-18s %-18s %-18s" % (task, cells[0], cells[1]))
    print("-" * 90)
    for model_key in ("base", "lora"):
        failures = [
            result
            for result in evaluation[model_key]["results"]
            if not result[model_key + "_correct"]
        ]
        print("%s failures: %d" % (model_key.capitalize(), len(failures)))
        for result in failures:
            print(
                "  [%s] %s | expected=%s | predicted=%s"
                % (
                    result["task"],
                    result["id"],
                    result["expected"],
                    _clipped(result["prediction"], 60),
                )
            )


def print_step_c(questions, evaluation):
    print("=" * 90)
    print("Step C - Honest raw-FASTA control (240 bp snippets, no tree)")
    print("=" * 90)
    for question in questions:
        print("  [%s] expected=%s" % (question["kind"], question["expected"]))
    if evaluation is None:
        print("Raw-FASTA control prompts built; model scoring skipped.")
        return
    stats = evaluation["overall"]
    print(
        "LoRA raw-FASTA score: %d/%d (%.2f%%)"
        % (
            stats["correct"],
            stats["total"],
            100.0 * stats["accuracy"],
        )
    )
    for result in evaluation["results"]:
        print(
            "  [%s] expected=%s | predicted=%s | %s"
            % (
                result["id"],
                result["expected"],
                _clipped(result["prediction"], 60),
                "PASS" if result["lora_correct"] else "FAIL",
            )
        )


def _save_json(path, payload):
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main(argv=None):
    args = parse_args(argv)
    fasta_text = _read_text(args.fasta)
    reconstruction = reconstruct_tree_from_fasta(fasta_text)

    print("=" * 90)
    print("Step A - Biological Reconstruction Validation")
    print("=" * 90)
    print("FASTA: %s" % args.fasta)
    print("Taxa: %d" % len(reconstruction["taxa"]))
    for taxon in reconstruction["taxa"]:
        print("  %-34s %d bp" % (taxon, reconstruction["lengths"][taxon]))
    _print_distance_matrix(
        reconstruction["taxa"], reconstruction["distance_matrix"]
    )
    print("Weighted Newick:   %s" % reconstruction["weighted_newick"])
    print("Normalized Newick: %s" % reconstruction["normalized_newick"])

    root = parse_newick(reconstruction["normalized_newick"])
    validation = _biological_validation(root)
    print("Biological validation check:")
    for check in validation["checks"]:
        print(
            "  %-70s expected=%s actual=%s [%s]"
            % (
                check["name"],
                check["expected"],
                check["actual"],
                "PASS" if check["passed"] else "FAIL",
            )
        )
    if not validation["all_passed"]:
        raise RuntimeError(
            "Biological validation failed; reconstructed tree does not recover "
            "the expected Bacteria/Archaea and phylum clades."
        )

    tree_questions = build_tree_questions(reconstruction)
    raw_questions = build_raw_fasta_questions(reconstruction, fasta_text)

    step_a = dict(reconstruction)
    step_a["biological_validation"] = validation
    payload = {
        "fasta": args.fasta,
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "max_new_tokens": args.max_new_tokens,
        "verify_only": args.verify_only,
        "step_a": step_a,
        "step_b": {"questions": tree_questions, "evaluation": None},
        "step_c": {"questions": raw_questions, "evaluation": None},
    }

    if args.verify_only:
        print_step_b(tree_questions, None)
        print_step_c(raw_questions, None)
        _save_json(args.output_json, payload)
        print("Saved verified reconstruction and questions to %s" % args.output_json)
        return

    model, tokenizer, device = _load_models(args)
    step_b_evaluation = evaluate_step_b(
        model,
        tokenizer,
        tree_questions,
        device,
        args.max_new_tokens,
    )
    step_c_evaluation = evaluate_step_c(
        model,
        tokenizer,
        raw_questions,
        device,
        args.max_new_tokens,
    )

    payload["step_b"]["evaluation"] = step_b_evaluation
    payload["step_c"]["evaluation"] = step_c_evaluation
    print_step_b(tree_questions, step_b_evaluation)
    print_step_c(raw_questions, step_c_evaluation)
    _save_json(args.output_json, payload)
    print("Saved full evaluation to %s" % args.output_json)


if __name__ == "__main__":
    main()
