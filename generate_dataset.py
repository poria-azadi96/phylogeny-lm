#!/usr/bin/env python3
"""Generate and verify a synthetic phylogeny question-answering dataset.

Only the Python 3 standard library is used.
"""

import argparse
import itertools
import json
import os
import random
import re
import sys


TASKS = ("mrca", "sister", "monophyletic", "distance")
REQUIRED_FIELDS = ("id", "split", "task", "tree", "prompt", "answer")
SPLITS = ("train", "val", "test")

TRAIN_NAMES = (
    "Aeris", "Boreon", "Caelus", "Dravix", "Elysia", "Ferron", "Galena",
    "Helion", "Iridia", "Jovian", "Kaelix", "Lumina", "Mireon", "Nivalis",
    "Orinth", "Pellis", "Quorix", "Ravena", "Sorell", "Tauron", "Ulvan",
    "Vesper", "Wrenna", "Xanthos", "Yalara", "Zephyra", "Asterion",
    "Brivia", "Corvus", "Diantha", "Eburon", "Fenara",
)

VAL_NAMES = (
    "Halcyon", "Ignis", "Junara", "Koreth", "Lyrion", "Morwen", "Nereus",
    "Octavia", "Pyra", "Quilla", "Rhodon", "Sylvan", "Thalia", "Umbria",
    "Vorcan", "Wyrena", "Xerion", "Yvaine", "Zorvan", "Amberis", "Brontes",
    "Cypria", "Dorian", "Elara", "Faelan", "Glaucus", "Hesper", "Ismena",
    "Jorund", "Kalypso", "Lethos", "Marnix",
)

TEST_NAMES = (
    "Nyx", "Oberon", "Pallus", "Quirin", "Rhea", "Solon", "Tethys",
    "Uraeus", "Vespera", "Wystan", "Xylo", "Ymir", "Zelena", "Achilles",
    "Bellona", "Cassian", "Daphne", "Endymion", "Faunus", "Grendel",
    "Hesperia", "Icarus", "Jasper", "Kalliope", "Lycaon", "Morpheus",
    "Naiad", "Orpheus", "Perses", "Rheia", "Silvanus", "Triton",
)

NAME_POOLS = {
    "train": TRAIN_NAMES,
    "val": VAL_NAMES,
    "test": TEST_NAMES,
}

SYNTHETIC_CLADE_LABELS = (
    "CladeA", "GroupB", "LineageC", "BranchD", "NodeE", "TaxonCladeF",
    "SubcladeG", "StemH", "CladeI", "GroupJ", "LineageK", "BranchL",
    "NodeM", "TaxonCladeN", "SubcladeO", "StemP", "CladeQ", "GroupR",
    "LineageS", "BranchT", "NodeU", "TaxonCladeV", "SubcladeW", "StemX",
)

NAME_PATTERN = r"[A-Za-z][A-Za-z0-9_]*"
MRCA_RE = re.compile(
    r"What is the most recent common ancestor of (" + NAME_PATTERN + r") and ("
    + NAME_PATTERN + r")\?"
)
SISTER_RE = re.compile(r"What is the direct sister taxon of (" + NAME_PATTERN + r")\?")
MONO_RE = re.compile(r"Do (.+) form a monophyletic clade\?")
DISTANCE_RE = re.compile(
    r"What is the topological edge distance between (" + NAME_PATTERN + r") and ("
    + NAME_PATTERN + r")\?"
)


class NewickError(ValueError):
    """Raised when a Newick string cannot be parsed or is not a valid tree."""


class Node:
    __slots__ = ("name", "label", "children", "parent")

    def __init__(self, name=None, label=None, children=None):
        self.name = name
        self.label = label
        self.children = list(children) if children else []
        self.parent = None
        for child in self.children:
            child.parent = self

    @property
    def is_leaf(self):
        return not self.children


def parse_newick(newick_str):
    """Parse a rooted binary Newick tree with labelled internal nodes.

    This is an independent recursive-descent parser. It rejects branch
    lengths, missing internal labels, non-binary internal nodes, duplicate
    leaf names, duplicate clade labels, and malformed/trailing input.
    """
    if not isinstance(newick_str, str):
        raise NewickError("Newick input must be a string")

    text = newick_str.strip()
    if not text:
        raise NewickError("Newick input is empty")
    if not text.endswith(";"):
        raise NewickError("Newick input must end with ';'")

    pos = 0
    length = len(text)

    def skip_ws():
        nonlocal pos
        while pos < length and text[pos].isspace():
            pos += 1

    def read_token():
        nonlocal pos
        skip_ws()
        start = pos
        while (
            pos < length
            and text[pos] not in "(),;:"
            and not text[pos].isspace()
        ):
            pos += 1
        if start == pos:
            raise NewickError("Expected a taxon or clade label at position %d" % pos)
        return text[start:pos]

    def parse_subtree():
        nonlocal pos
        skip_ws()
        if pos >= length:
            raise NewickError("Unexpected end of Newick input")

        if text[pos] == "(":
            pos += 1
            left = parse_subtree()
            skip_ws()
            if pos >= length or text[pos] != ",":
                raise NewickError("Expected ',' between children at position %d" % pos)
            pos += 1
            right = parse_subtree()
            skip_ws()
            if pos >= length or text[pos] != ")":
                raise NewickError("Expected ')' at position %d" % pos)
            pos += 1
            label = read_token()
            return Node(label=label, children=(left, right))

        if text[pos] in ",);:":
            raise NewickError("Expected a taxon name at position %d" % pos)
        name = read_token()
        return Node(name=name)

    root = parse_subtree()
    skip_ws()
    if pos >= length or text[pos] != ";":
        raise NewickError("Expected terminating ';' at position %d" % pos)
    pos += 1
    skip_ws()
    if pos != length:
        raise NewickError("Unexpected trailing content after ';'")

    leaf_names = []
    internal_labels = []

    def validate(node):
        if node.is_leaf:
            if not node.name:
                raise NewickError("Leaf without a taxon name")
            leaf_names.append(node.name)
        else:
            if len(node.children) != 2:
                raise NewickError("Internal nodes must have exactly two children")
            if not node.label:
                raise NewickError("Internal node without a clade label")
            internal_labels.append(node.label)
            for child in node.children:
                validate(child)

    validate(root)

    if not internal_labels:
        raise NewickError("Tree must contain at least one internal node")
    if len(set(leaf_names)) != len(leaf_names):
        raise NewickError("Duplicate taxon names are not allowed")
    if len(set(internal_labels)) != len(internal_labels):
        raise NewickError("Duplicate clade labels are not allowed")

    return root


def all_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def all_internal_nodes(root):
    return [node for node in all_nodes(root) if not node.is_leaf]


def all_leaves(root):
    return [node for node in all_nodes(root) if node.is_leaf]


def clade_names(node):
    if node.is_leaf:
        return frozenset((node.name,))
    names = set()
    for child in node.children:
        names.update(clade_names(child))
    return frozenset(names)


def leaf_index(root):
    return {leaf.name: leaf for leaf in all_leaves(root)}


def lca(a, b):
    ancestors = set()
    node = a
    while node is not None:
        ancestors.add(node)
        node = node.parent
    node = b
    while node not in ancestors:
        if node is None:
            raise ValueError("Nodes are not in the same tree")
        node = node.parent
    return node


def node_depth(node):
    depth = 0
    while node.parent is not None:
        depth += 1
        node = node.parent
    return depth


def edge_distance(a, b):
    return node_depth(a) + node_depth(b) - 2 * node_depth(lca(a, b))


def render_newick(node):
    if node.is_leaf:
        return node.name
    children = ",".join(render_newick(child) for child in node.children)
    return "(" + children + ")" + node.label


def build_random_tree(names, rng):
    """Build an unlabelled random rooted binary tree over the given names."""
    if len(names) == 1:
        return Node(name=names[0])
    split = rng.randint(1, len(names) - 1)
    left = build_random_tree(names[:split], rng)
    right = build_random_tree(names[split:], rng)
    return Node(children=(left, right))


def assign_clade_labels(root, mixed=False, rng=None):
    internal_nodes = []

    def visit(node):
        if node.is_leaf:
            return
        for child in node.children:
            visit(child)
        internal_nodes.append(node)

    visit(root)

    if mixed:
        if rng is None:
            raise ValueError("rng is required for mixed clade labels")
        labels = rng.sample(SYNTHETIC_CLADE_LABELS, len(internal_nodes))
        for node, label in zip(internal_nodes, labels):
            node.label = label
    else:
        for index, node in enumerate(internal_nodes, start=1):
            node.label = "C%d" % index

    return root


def make_question_and_answer(rng, root, task, force_positive=None):
    leaves = leaf_index(root)
    names = sorted(leaves)
    internal = all_internal_nodes(root)
    clade_sets = {clade_names(node) for node in internal}

    if task == "mrca":
        a_name, b_name = rng.sample(names, 2)
        ancestor = lca(leaves[a_name], leaves[b_name])
        if ancestor.is_leaf:
            raise RuntimeError("MRCA of distinct leaves cannot be a leaf")
        question = "What is the most recent common ancestor of %s and %s?" % (
            a_name,
            b_name,
        )
        return question, ancestor.label

    if task == "sister":
        candidates = []
        for node in internal:
            left, right = node.children
            if left.is_leaf and right.is_leaf:
                candidates.append((left.name, right.name))
                candidates.append((right.name, left.name))
        if not candidates:
            raise RuntimeError("Every binary tree with multiple leaves must have a cherry")
        a_name, b_name = rng.choice(candidates)
        question = "What is the direct sister taxon of %s?" % a_name
        return question, b_name

    if task == "monophyletic":
        positive = rng.random() < 0.5 if force_positive is None else force_positive
        if positive:
            candidates = [
                sorted(s)
                for s in clade_sets
                if len(s) in (2, 3)
            ]
            if not candidates:
                raise RuntimeError("Tree has no 2- or 3-leaf internal clade")
            chosen = rng.choice(candidates)
        else:
            candidates = []
            for size in (2, 3):
                for combo in itertools.combinations(names, size):
                    if frozenset(combo) not in clade_sets:
                        candidates.append(list(combo))
            if not candidates:
                raise RuntimeError("Tree has no non-monophyletic 2- or 3-leaf set")
            chosen = rng.choice(candidates)

        chosen = sorted(chosen)
        answer = "Yes" if frozenset(chosen) in clade_sets else "No"
        question = "Do %s form a monophyletic clade?" % ", ".join(chosen)
        return question, answer

    if task == "distance":
        a_name, b_name = rng.sample(names, 2)
        distance = edge_distance(leaves[a_name], leaves[b_name])
        question = "What is the topological edge distance between %s and %s?" % (
            a_name,
            b_name,
        )
        return question, str(distance)

    raise ValueError("Unknown task: %r" % (task,))


def make_prompt(tree, task, question):
    return "Tree: %s\nTask (%s): %s\nAnswer with only the exact value." % (
        tree,
        task,
        question,
    )


def generate_record(
    rng,
    split,
    pool,
    task,
    record_id,
    force_positive=None,
    mixed_clade_labels=False,
):
    leaf_count = rng.randint(5, 7)
    taxon_names = rng.sample(list(pool), leaf_count)
    root = build_random_tree(taxon_names, rng)
    assign_clade_labels(root, mixed=mixed_clade_labels, rng=rng)
    tree = render_newick(root) + ";"

    question, answer = make_question_and_answer(
        rng, root, task, force_positive=force_positive
    )
    return {
        "id": record_id,
        "split": split,
        "task": task,
        "tree": tree,
        "prompt": make_prompt(tree, task, question),
        "answer": answer,
    }


def balanced_task_counts(total):
    base, remainder = divmod(total, len(TASKS))
    return [base + (1 if i < remainder else 0) for i in range(len(TASKS))]


def generate_split(split, total, rng, mixed_clade_labels=False):
    counts = balanced_task_counts(total)
    tasks = []
    for task, count in zip(TASKS, counts):
        tasks.extend([task] * count)
    rng.shuffle(tasks)

    mixed_flags = [False] * total
    if mixed_clade_labels and total:
        mixed_count = int(round(total * 0.3))
        mixed_flags = [True] * mixed_count + [False] * (total - mixed_count)
        rng.shuffle(mixed_flags)

    records = []
    positive_index = 0
    for index, task in enumerate(tasks, start=1):
        force_positive = None
        if task == "monophyletic":
            force_positive = positive_index % 2 == 0
            positive_index += 1
        record_id = "%s-%06d" % (split, index)
        records.append(
            generate_record(
                rng,
                split,
                NAME_POOLS[split],
                task,
                record_id,
                force_positive=force_positive,
                mixed_clade_labels=mixed_flags[index - 1],
            )
        )
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def extract_question(record):
    task = record["task"]
    prompt = record["prompt"]
    parts = prompt.split("\n")
    if len(parts) != 3:
        raise AssertionError("%s: prompt must have exactly three lines" % record["id"])
    prefix = "Tree: "
    if not parts[0].startswith(prefix):
        raise AssertionError("%s: prompt has invalid tree line" % record["id"])
    if parts[0][len(prefix):] != record["tree"]:
        raise AssertionError("%s: prompt tree does not match record tree" % record["id"])
    task_prefix = "Task (%s): " % task
    if not parts[1].startswith(task_prefix):
        raise AssertionError("%s: prompt has invalid task line" % record["id"])
    if parts[2] != "Answer with only the exact value.":
        raise AssertionError("%s: prompt has invalid answer instruction" % record["id"])
    return parts[1][len(task_prefix):]


def recompute_answer(task, question, root):
    leaves = leaf_index(root)
    internal = all_internal_nodes(root)
    clade_sets = {clade_names(node) for node in internal}

    if task == "mrca":
        match = MRCA_RE.fullmatch(question)
        if not match:
            raise AssertionError("Invalid mrca question: %r" % question)
        a_name, b_name = match.groups()
        if a_name == b_name:
            raise AssertionError("MRCA question uses the same leaf twice")
        if a_name not in leaves or b_name not in leaves:
            raise AssertionError("MRCA question uses an unknown leaf")
        ancestor = lca(leaves[a_name], leaves[b_name])
        if ancestor.is_leaf:
            raise AssertionError("MRCA of distinct leaves is a leaf")
        return ancestor.label

    if task == "sister":
        match = SISTER_RE.fullmatch(question)
        if not match:
            raise AssertionError("Invalid sister question: %r" % question)
        a_name = match.group(1)
        if a_name not in leaves:
            raise AssertionError("Sister question uses an unknown leaf")
        parent = leaves[a_name].parent
        if parent is None:
            raise AssertionError("Sister question leaf has no parent")
        siblings = [child for child in parent.children if child is not leaves[a_name]]
        if len(siblings) != 1 or not siblings[0].is_leaf:
            raise AssertionError("Sister question is not about a leaf cherry")
        return siblings[0].name

    if task == "monophyletic":
        match = MONO_RE.fullmatch(question)
        if not match:
            raise AssertionError("Invalid monophyletic question: %r" % question)
        raw = match.group(1)
        taxa = raw.split(", ")
        if len(taxa) not in (2, 3):
            raise AssertionError("Monophyletic question must use 2 or 3 leaves")
        if len(set(taxa)) != len(taxa):
            raise AssertionError("Monophyletic question has duplicate leaves")
        if taxa != sorted(taxa):
            raise AssertionError("Monophyletic question leaves are not sorted")
        if any(name not in leaves for name in taxa):
            raise AssertionError("Monophyletic question uses an unknown leaf")
        return "Yes" if frozenset(taxa) in clade_sets else "No"

    if task == "distance":
        match = DISTANCE_RE.fullmatch(question)
        if not match:
            raise AssertionError("Invalid distance question: %r" % question)
        a_name, b_name = match.groups()
        if a_name == b_name:
            raise AssertionError("Distance question uses the same leaf twice")
        if a_name not in leaves or b_name not in leaves:
            raise AssertionError("Distance question uses an unknown leaf")
        return str(edge_distance(leaves[a_name], leaves[b_name]))

    raise AssertionError("Unknown task: %r" % (task,))


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise AssertionError("%s:%d: blank line" % (path, line_number))
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError("%s:%d: invalid JSON: %s" % (path, line_number, exc))
            records.append(record)
    return records


def verify_records(
    records,
    split,
    expected_size,
    mixed_clade_labels=False,
    allowed_taxa=None,
    seen_ids=None,
):
    if len(records) != expected_size:
        raise AssertionError(
            "%s: expected %d records, found %d"
            % (split, expected_size, len(records))
        )
    if seen_ids is None:
        seen_ids = set()

    task_counts = {task: 0 for task in TASKS}
    taxa = set()
    trees = set()
    prompts = set()
    standard_label_trees = 0
    mixed_label_trees = 0
    synthetic_labels = set(SYNTHETIC_CLADE_LABELS)

    for record in records:
        if not isinstance(record, dict):
            raise AssertionError("%s: record is not an object" % split)
        if set(record) != set(REQUIRED_FIELDS):
            raise AssertionError(
                "%s: record fields are %r, expected %r"
                % (split, sorted(record), sorted(REQUIRED_FIELDS))
            )
        if record["split"] != split:
            raise AssertionError("%s: record split is %r" % (split, record["split"]))
        if record["task"] not in TASKS:
            raise AssertionError("%s: unknown task %r" % (split, record["task"]))
        if record["id"] in seen_ids:
            raise AssertionError("Duplicate record id: %r" % record["id"])
        seen_ids.add(record["id"])

        root = parse_newick(record["tree"])
        internal = all_internal_nodes(root)
        labels = {node.label for node in internal}
        expected_labels = {"C%d" % i for i in range(1, len(internal) + 1)}
        if mixed_clade_labels:
            if labels == expected_labels:
                standard_label_trees += 1
            elif labels.issubset(synthetic_labels):
                mixed_label_trees += 1
            else:
                raise AssertionError(
                    "%s: clade labels %r are neither C1..C%d nor synthetic labels"
                    % (record["id"], labels, len(internal))
                )
        elif labels != expected_labels:
            raise AssertionError(
                "%s: clade labels %r are not C1..C%d"
                % (record["id"], labels, len(internal))
            )

        leaf_names = {leaf.name for leaf in all_leaves(root)}
        if not 5 <= len(leaf_names) <= 7:
            raise AssertionError("%s: tree does not have 5 to 7 leaves" % record["id"])
        if allowed_taxa is not None and not leaf_names.issubset(allowed_taxa):
            raise AssertionError(
                "%s: tree uses taxa outside the allowed pool: %r"
                % (record["id"], sorted(leaf_names - allowed_taxa))
            )

        question = extract_question(record)
        expected_answer = recompute_answer(record["task"], question, root)
        if record["answer"] != expected_answer:
            raise AssertionError(
                "%s: answer %r != recomputed %r"
                % (record["id"], record["answer"], expected_answer)
            )
        if record["task"] == "monophyletic" and record["answer"] not in ("Yes", "No"):
            raise AssertionError("%s: monophyletic answer is not Yes/No" % record["id"])

        task_counts[record["task"]] += 1
        taxa.update(leaf_names)
        trees.add(record["tree"])
        prompts.add(record["prompt"])

    counts = list(task_counts.values())
    if max(counts) - min(counts) > 1:
        raise AssertionError("%s: task counts are not balanced: %r" % (split, task_counts))

    if mixed_clade_labels and expected_size >= 20:
        mixed_fraction = mixed_label_trees / float(expected_size)
        if not 0.2 <= mixed_fraction <= 0.4:
            raise AssertionError(
                "%s: mixed-label fraction %.3f is not near 30%%"
                % (split, mixed_fraction)
            )

    message = (
        "%s: %d records, task counts %s, %d taxa, %d unique trees"
        % (split, len(records), task_counts, len(taxa), len(trees))
    )
    if mixed_clade_labels:
        message += ", %d standard-label trees, %d mixed-label trees" % (
            standard_label_trees,
            mixed_label_trees,
        )
    print(message)

    return {
        "taxa": taxa,
        "trees": trees,
        "prompts": prompts,
        "task_counts": task_counts,
        "standard_label_trees": standard_label_trees,
        "mixed_label_trees": mixed_label_trees,
    }


def verify_dataset(out_dir, expected_sizes, mixed_clade_labels=False):
    split_data = {}
    seen_ids = set()

    for split in SPLITS:
        path = os.path.join(out_dir, split + ".jsonl")
        if not os.path.exists(path):
            raise AssertionError("Missing dataset file: %s" % path)
        records = load_jsonl(path)
        split_data[split] = verify_records(
            records,
            split,
            expected_sizes[split],
            mixed_clade_labels=mixed_clade_labels,
            allowed_taxa=set(NAME_POOLS[split]),
            seen_ids=seen_ids,
        )

    for first, second in itertools.combinations(SPLITS, 2):
        for label, key in (
            ("taxon names", "taxa"),
            ("Newick trees", "trees"),
            ("prompts", "prompts"),
        ):
            overlap = split_data[first][key] & split_data[second][key]
            if overlap:
                raise AssertionError(
                    "%s/%s %s overlap: %r"
                    % (first, second, label, sorted(overlap)[:5])
                )

    print("Verification passed: all answers correct, task counts balanced, splits disjoint.")


def collect_record_sets(records):
    taxa = set()
    trees = set()
    prompts = set()
    for record in records:
        root = parse_newick(record["tree"])
        taxa.update(leaf.name for leaf in all_leaves(root))
        trees.add(record["tree"])
        prompts.add(record["prompt"])
    return {"taxa": taxa, "trees": trees, "prompts": prompts}


def verify_train_only_file(path, expected_size, mixed_clade_labels, reference_paths):
    records = load_jsonl(path)
    data = verify_records(
        records,
        "train",
        expected_size,
        mixed_clade_labels=mixed_clade_labels,
        allowed_taxa=set(TRAIN_NAMES),
    )

    for reference_path in reference_paths:
        if not os.path.exists(reference_path):
            print("Reference file %s not found; skipping overlap check." % reference_path)
            continue
        reference_data = collect_record_sets(load_jsonl(reference_path))
        for label, key in (
            ("taxon names", "taxa"),
            ("Newick trees", "trees"),
            ("prompts", "prompts"),
        ):
            overlap = data[key] & reference_data[key]
            if overlap:
                raise AssertionError(
                    "%s %s overlap with %s: %r"
                    % (path, label, reference_path, sorted(overlap)[:5])
                )
        print("No taxon/tree/prompt overlap between %s and %s." % (path, reference_path))

    print(
        "Verification passed for %s: all answers correct, tasks balanced, references disjoint."
        % path
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and verify a synthetic phylogeny QA dataset."
    )
    parser.add_argument("--train_size", type=int, default=600)
    parser.add_argument("--val_size", type=int, default=80)
    parser.add_argument("--test_size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument(
        "--train_only_out",
        default=None,
        help="Write only a new training split to this path; do not touch data/*.jsonl.",
    )
    parser.add_argument(
        "--mixed_clade_labels",
        action="store_true",
        help="Use synthetic non-C clade labels for about 30%% of generated trees.",
    )
    parser.add_argument("--verify", dest="verify", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args(argv)

    if args.train_size < 0 or args.val_size < 0 or args.test_size < 0:
        parser.error("dataset sizes must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }

    pools = [set(NAME_POOLS[split]) for split in SPLITS]
    if any(len(pool) < 24 for pool in pools):
        raise AssertionError("Each split needs at least 24 taxon names")
    for first, second in itertools.combinations(SPLITS, 2):
        if NAME_POOLS[first] and set(NAME_POOLS[first]) & set(NAME_POOLS[second]):
            raise AssertionError("Taxon name pools for %s and %s overlap" % (first, second))

    if args.train_only_out:
        output_path = os.path.abspath(args.train_only_out)
        protected_paths = {
            os.path.abspath(os.path.join(args.out_dir, split + ".jsonl"))
            for split in SPLITS
        }
        if output_path in protected_paths:
            raise AssertionError(
                "--train_only_out must not overwrite an existing split file"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rng = random.Random(args.seed)
        records = generate_split(
            "train",
            args.train_size,
            rng,
            mixed_clade_labels=args.mixed_clade_labels,
        )
        write_jsonl(output_path, records)
        print("Wrote %d records to %s" % (len(records), output_path))

        if args.verify:
            reference_paths = [
                os.path.join(args.out_dir, "val.jsonl"),
                os.path.join(args.out_dir, "test.jsonl"),
            ]
            verify_train_only_file(
                output_path,
                args.train_size,
                args.mixed_clade_labels,
                reference_paths,
            )
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    for split in SPLITS:
        records = generate_split(
            split,
            sizes[split],
            rng,
            mixed_clade_labels=args.mixed_clade_labels,
        )
        path = os.path.join(args.out_dir, split + ".jsonl")
        write_jsonl(path, records)
        print("Wrote %d records to %s" % (len(records), path))

    if args.verify:
        verify_dataset(
            args.out_dir,
            sizes,
            mixed_clade_labels=args.mixed_clade_labels,
        )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("Verification failed: %s" % exc, file=sys.stderr)
        sys.exit(1)
