#!/usr/bin/env python3
"""Stdlib HTTP inference server for the fine-tuned phylogeny QA adapter."""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from phylogeny_utils import (
    NewickError,
    make_prompt,
    parse_newick,
    recompute_answer,
    tree_summary,
)


TASKS = ("mrca", "sister", "monophyletic", "distance")
MAX_BODY_BYTES = 1_000_000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_INDEX = os.path.join(BASE_DIR, "static", "index.html")
MODEL_LOCK = threading.Lock()

PRESET_SPECS = (
    {
        "id": "primates",
        "name": "Primates",
        "description": "Great-ape phylogeny with Homo and Pan forming a cherry.",
        "newick": (
            "(((Homo,Pan)Hominini,Gorilla)Homininae,"
            "(Pongo,Hylobates)Pongidae)Hominoidea;"
        ),
        "task": "mrca",
        "taxa": ("Homo", "Gorilla"),
    },
    {
        "id": "carnivores",
        "name": "Carnivores",
        "description": "Carnivoran families split into Caniformia and Felidae.",
        "newick": (
            "(((Canis,Vulpes)Canidae,(Ursus,Ailuropoda)Ursidae)Caniformia,"
            "(Felis,Panthera)Felidae)Carnivora;"
        ),
        "task": "mrca",
        "taxa": ("Canis", "Ursus"),
    },
    {
        "id": "vertebrates-paraphyly-trap",
        "name": "Vertebrates Paraphyly Trap",
        "description": (
            "Alligator and Anolis are not a clade: Anolis falls outside "
            "Archosauria."
        ),
        "newick": (
            "(((Gallus,Alligator)Archosauria,Anolis)Sauropsida,"
            "(Mus,Ornithorhynchus)Mammalia)Amniota;"
        ),
        "task": "monophyletic",
        "taxa": ("Alligator", "Anolis"),
    },
    {
        "id": "land-plants",
        "name": "Land Plants",
        "description": (
            "Bryophytes branch off first; Pinus and Ginkgo form Gymnospermae."
        ),
        "newick": (
            "((Marchantia,Physcomitrium)Bryophyta,(Selaginella,"
            "(Pinus,Ginkgo)Gymnospermae)Tracheophyta)LandPlants;"
        ),
        "task": "mrca",
        "taxa": ("Pinus", "Ginkgo"),
    },
    {
        "id": "adversarial-counterfactual-primates",
        "name": "Adversarial Counterfactual Primates",
        "description": (
            "Counterfactual topology where Homo's sister is Hylobates, not Pan."
        ),
        "newick": "(((Homo,Hylobates)MixA,(Pan,Gorilla)MixB)MixC,Mus)MixRoot;",
        "task": "sister",
        "taxa": ("Homo",),
    },
    {
        "id": "branch-length-primates",
        "name": "Branch-Length Primates",
        "description": (
            "Real primate topology with branch lengths; lengths are stripped "
            "and the topology is preserved."
        ),
        "newick": (
            "( ( (Homo:0.006, Pan:1.2e-2) Hominini:0.03, Gorilla:1.5e-3 ) "
            "Homininae:0.01, (Pongo:0.004, Hylobates:0.005) Pongidae:0.02 ) "
            "Hominoidea:0.1 ;"
        ),
        "task": "distance",
        "taxa": ("Homo", "Pongo"),
    },
)


def build_presets():
    presets = []
    for spec in PRESET_SPECS:
        summary = tree_summary(spec["newick"], auto_label_internal=True)
        presets.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "description": spec["description"],
                "raw_newick": spec["newick"],
                "normalized_newick": summary["normalized_newick"],
                "metadata": summary["metadata"],
                "leaves": summary["leaves"],
                "internal_clades": summary["internal_clades"],
                "cherry_leaves": summary["cherry_leaves"],
                "tree_dict": summary["tree_dict"],
                "suggested_task": spec["task"],
                "suggested_taxa": list(spec["taxa"]),
                "suggested": {
                    "task": spec["task"],
                    "taxa": list(spec["taxa"]),
                },
            }
        )
    return presets


def first_nonempty_line(text):
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def answer_matches(prediction, ground_truth):
    if prediction is None:
        return None
    return str(prediction).strip().lower() == str(ground_truth).strip().lower()


def load_causal_lm(model_name, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError as exc:
        if "dtype" not in str(exc).lower():
            raise
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def generate_prediction(model, tokenizer, prompt, device, max_new_tokens):
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
    return first_nonempty_line(completion)


def generate_base_prediction(model, tokenizer, prompt, device, max_new_tokens):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return generate_prediction(
                model, tokenizer, prompt, device, max_new_tokens
            )
    model.disable_adapter_layers()
    try:
        return generate_prediction(
            model, tokenizer, prompt, device, max_new_tokens
        )
    finally:
        model.enable_adapter_layers()


def normalize_taxa(raw_taxa):
    if isinstance(raw_taxa, str):
        raw_taxa = [part.strip() for part in raw_taxa.split(",")]
    if not isinstance(raw_taxa, list):
        raise ValueError("'taxa' must be a list of leaf names")
    taxa = []
    for item in raw_taxa:
        if not isinstance(item, str):
            raise ValueError("Taxon names must be strings")
        name = item.strip()
        if not name:
            raise ValueError("Taxon names must not be empty")
        taxa.append(name)
    return taxa


def prepare_question(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    newick = payload.get("newick")
    summary = tree_summary(newick, auto_label_internal=True)

    task = payload.get("task")
    if not isinstance(task, str):
        raise ValueError("'task' must be one of: %s" % ", ".join(TASKS))
    task = task.strip().lower()
    if task not in TASKS:
        raise ValueError("'task' must be one of: %s" % ", ".join(TASKS))

    taxa = normalize_taxa(payload.get("taxa"))
    if not taxa:
        raise ValueError("At least one taxon is required")
    if len(set(taxa)) != len(taxa):
        raise ValueError("Taxa must be distinct")

    leaf_set = set(summary["leaves"])
    unknown = [name for name in taxa if name not in leaf_set]
    if unknown:
        raise ValueError("Unknown taxa: %s" % ", ".join(unknown))

    if task in ("mrca", "distance"):
        if len(taxa) != 2:
            raise ValueError("'%s' requires exactly two distinct leaves" % task)
        if task == "mrca":
            question = (
                "What is the most recent common ancestor of %s and %s?"
                % (taxa[0], taxa[1])
            )
        else:
            question = (
                "What is the topological edge distance between %s and %s?"
                % (taxa[0], taxa[1])
            )
    elif task == "sister":
        if len(taxa) != 1:
            raise ValueError("'sister' requires exactly one leaf")
        if taxa[0] not in set(summary["cherry_leaves"]):
            raise ValueError(
                "Sister task requires a leaf inside a cherry; choose one of: %s"
                % ", ".join(summary["cherry_leaves"])
            )
        question = "What is the direct sister taxon of %s?" % taxa[0]
    else:
        if len(taxa) not in (2, 3):
            raise ValueError("'monophyletic' requires two or three distinct leaves")
        taxa = sorted(taxa)
        question = "Do %s form a monophyletic clade?" % ", ".join(taxa)

    return {
        "summary": summary,
        "task": task,
        "taxa": taxa,
        "question": question,
    }


class PhylogenyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write(
            "%s - %s\n" % (self.log_date_time_string(), fmt % args)
        )

    def _send_bytes(self, body, status=200, content_type="text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, status=status, content_type="application/json; charset=utf-8")

    def _read_json(self):
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Missing Content-Length header")
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise ValueError("Invalid Content-Length header")
        if length <= 0:
            raise ValueError("Request body is empty")
        if length > MAX_BODY_BYTES:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Request body must be valid JSON")
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_index()
            return
        if path == "/api/health":
            state = self.server.state
            self._send_json(
                {
                    "status": "ok",
                    "model_name": state["model_name"],
                    "adapter_dir": state["adapter_dir"],
                    "device": state["device"],
                    "base_enabled": state["base_enabled"],
                }
            )
            return
        if path == "/api/presets":
            self._send_json(self.server.state["presets"])
            return
        if path == "/favicon.ico":
            self._send_bytes(b"", status=204)
            return
        if path in ("/api/parse", "/api/predict"):
            self._send_json({"error": "Use POST for this endpoint"}, status=405)
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/parse", "/api/predict"):
            if path in ("/api/health", "/api/presets"):
                self._send_json({"error": "Use GET for this endpoint"}, status=405)
            else:
                self._send_json({"error": "Not found"}, status=404)
            return

        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/parse":
            self._handle_parse(payload)
        else:
            self._handle_predict(payload)

    def _serve_index(self):
        try:
            with open(STATIC_INDEX, "rb") as handle:
                body = handle.read()
        except OSError:
            self._send_bytes(b"static/index.html not found", status=500)
            return
        self._send_bytes(body, status=200, content_type="text/html; charset=utf-8")

    def _handle_parse(self, payload):
        try:
            newick = payload.get("newick")
            summary = tree_summary(newick, auto_label_internal=True)
        except (NewickError, ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(summary)

    def _handle_predict(self, payload):
        try:
            prepared = prepare_question(payload)
        except (NewickError, ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        state = self.server.state
        summary = prepared["summary"]
        task = prepared["task"]
        taxa = prepared["taxa"]
        question = prepared["question"]
        normalized = summary["normalized_newick"]
        prompt = make_prompt(normalized, task, question)

        try:
            root = parse_newick(normalized)
            ground_truth = str(recompute_answer(task, question, root))
        except (NewickError, ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        try:
            with MODEL_LOCK:
                started = time.perf_counter()
                lora_prediction = generate_prediction(
                    state["model"],
                    state["tokenizer"],
                    prompt,
                    state["device"],
                    state["max_new_tokens"],
                )
                lora_ms = round((time.perf_counter() - started) * 1000.0, 2)

                if state["base_enabled"]:
                    started = time.perf_counter()
                    base_prediction = generate_base_prediction(
                        state["model"],
                        state["tokenizer"],
                        prompt,
                        state["device"],
                        state["max_new_tokens"],
                    )
                    base_ms = round((time.perf_counter() - started) * 1000.0, 2)
                else:
                    base_prediction = None
                    base_ms = None
        except Exception as exc:
            sys.stderr.write(traceback.format_exc())
            self._send_json(
                {"error": "Inference failed: %s" % exc}, status=500
            )
            return

        self._send_json(
            {
                "normalized_newick": normalized,
                "metadata": summary["metadata"],
                "tree_dict": summary["tree_dict"],
                "task": task,
                "taxa": taxa,
                "question": question,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "lora_prediction": lora_prediction,
                "lora_correct": bool(answer_matches(lora_prediction, ground_truth)),
                "lora_ms": lora_ms,
                "base_prediction": base_prediction,
                "base_correct": answer_matches(base_prediction, ground_truth),
                "base_ms": base_ms,
            }
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Serve the fine-tuned phylogeny QA adapter over HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--model_name",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument("--adapter_dir", default="model/phylogeny-lora")
    parser.add_argument("--no_base", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args(argv)

    if not (0 <= args.port <= 65535):
        parser.error("--port must be between 0 and 65535")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")
    return args


def load_state(args):
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
        raise SystemExit("Tokenizer must define a pad or eos token")

    base_model = load_causal_lm(args.model_name, dtype)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": str(device),
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "base_enabled": not args.no_base,
        "max_new_tokens": args.max_new_tokens,
        "presets": build_presets(),
    }


def main(argv=None):
    args = parse_args(argv)
    print("Loading adapter from %s ..." % args.adapter_dir, flush=True)
    state = load_state(args)
    server = ThreadingHTTPServer((args.host, args.port), PhylogenyHandler)
    server.state = state
    print("Device: %s" % state["device"], flush=True)
    print(
        "Base predictions: %s" % ("enabled" if state["base_enabled"] else "disabled"),
        flush=True,
    )
    print(
        "Serving on http://%s:%d (Ctrl+C to stop)" % (args.host, args.port),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
