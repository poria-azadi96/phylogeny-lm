#!/usr/bin/env python3
"""Train a LoRA adapter for the synthetic phylogeny QA task."""

import argparse
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


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


def _to_id_list(encoded):
    """Normalize apply_chat_template output to a flat list of token IDs."""
    if isinstance(encoded, dict) and "input_ids" in encoded:
        ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
    else:
        ids = encoded

    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    if isinstance(ids, tuple):
        ids = list(ids)

    if isinstance(ids, list):
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("Expected a single tokenized conversation")
            ids = ids[0]
        return [int(token_id) for token_id in ids]

    raise TypeError("Unsupported apply_chat_template output: %r" % type(encoded))


def build_tokenized_example(tokenizer, record, eos_token_id):
    user_messages = [{"role": "user", "content": record["prompt"]}]
    full_messages = user_messages + [
        {"role": "assistant", "content": record["answer"]}
    ]

    prompt_ids = _to_id_list(
        tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = _to_id_list(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )

    if not full_ids or full_ids[-1] != eos_token_id:
        full_ids.append(eos_token_id)

    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt prefix mismatch for record %r" % record.get("id")
        )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if len(labels) != len(full_ids):
        raise ValueError("Label length mismatch for record %r" % record.get("id"))

    return {
        "input_ids": full_ids,
        "labels": labels,
    }


class PhylogenyDataset(Dataset):
    def __init__(self, records, tokenizer, eos_token_id):
        self.examples = [
            build_tokenized_example(tokenizer, record, eos_token_id)
            for record in records
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def make_collate_fn(pad_token_id):
    def collate(batch):
        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            ids = item["input_ids"]
            item_labels = item["labels"]
            padding = max_length - len(ids)
            input_ids.append(ids + [pad_token_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(item_labels + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            token_count = (batch["labels"][:, 1:] != -100).sum().item()
            if token_count:
                total_loss += loss.item() * token_count
                total_tokens += token_count

    model.train()
    return total_loss / total_tokens if total_tokens else 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a phylogeny LoRA adapter.")
    parser.add_argument(
        "--model_name",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--val_file", default="data/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/phylogeny-lora")
    parser.add_argument("--metrics_json", default="results/train_metrics.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.lora_r <= 0:
        parser.error("--lora_r must be positive")
    if args.lora_alpha <= 0:
        parser.error("--lora_alpha must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a pad or eos token")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an eos token")
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
    )
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable_percentage = (
        100.0 * trainable_parameters / total_parameters
        if total_parameters
        else 0.0
    )

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)
    train_dataset = PhylogenyDataset(
        train_records, tokenizer, tokenizer.eos_token_id
    )
    val_dataset = PhylogenyDataset(val_records, tokenizer, tokenizer.eos_token_id)

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps)) if total_steps else 0
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch in train_loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            token_count = (batch["labels"][:, 1:] != -100).sum().item()
            if token_count:
                epoch_loss += loss.item() * token_count
                epoch_tokens += token_count

        train_loss = epoch_loss / epoch_tokens if epoch_tokens else 0.0
        val_loss = evaluate_loss(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(
            "Epoch %d/%d - train_loss: %.6f - val_loss: %.6f"
            % (epoch, args.epochs, train_loss, val_loss)
        )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "model_name": args.model_name,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "model_dtype": str(model_dtype),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "train_batches": len(train_loader),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "lora": {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": 0.05,
            "target_modules": TARGET_MODULES,
        },
        "parameter_stats": {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "trainable_percentage": trainable_percentage,
        },
        "history": history,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
    }

    metrics_dir = os.path.dirname(os.path.abspath(args.metrics_json))
    os.makedirs(metrics_dir, exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        "Saved LoRA adapter and tokenizer to %s" % args.output_dir
    )
    print("Saved training metrics to %s" % args.metrics_json)


if __name__ == "__main__":
    main()
