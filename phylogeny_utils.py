#!/usr/bin/env python3
"""Reusable Newick parsing, normalization, and exact phylogeny solvers."""

import re


NAME_PATTERN = r"[A-Za-z][A-Za-z0-9_]*"
BRANCH_LENGTH_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
MRCA_RE = re.compile(
    r"What is the most recent common ancestor of (" + NAME_PATTERN + r") and ("
    + NAME_PATTERN + r")\?"
)
SISTER_RE = re.compile(
    r"What is the direct sister taxon of (" + NAME_PATTERN + r")\?"
)
MONO_RE = re.compile(r"Do (.+) form a monophyletic clade\?")
DISTANCE_RE = re.compile(
    r"What is the topological edge distance between (" + NAME_PATTERN + r") and ("
    + NAME_PATTERN + r")\?"
)


class NewickError(ValueError):
    """Raised when a Newick string is malformed or invalid as a tree."""


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


def _without_whitespace(raw_newick):
    if not isinstance(raw_newick, str):
        raise NewickError("Newick input must be a string")
    text = raw_newick.strip()
    if not text:
        raise NewickError("Newick input is empty")
    if not text.endswith(";"):
        raise NewickError("Newick input must end with ';'")
    return "".join(text.split())


class _NewickParser:
    def __init__(self, text, allow_branch_lengths):
        self.text = text
        self.pos = 0
        self.length = len(text)
        self.allow_branch_lengths = allow_branch_lengths
        self.branch_lengths_stripped = False

    def parse(self):
        root = self._parse_subtree()
        if self.pos >= self.length or self.text[self.pos] != ";":
            raise NewickError("Expected terminating ';' at position %d" % self.pos)
        self.pos += 1
        if self.pos != self.length:
            raise NewickError("Unexpected trailing content after ';'")
        return root

    def _parse_subtree(self):
        if self.pos >= self.length:
            raise NewickError("Unexpected end of Newick input")

        if self.text[self.pos] == "(":
            self.pos += 1
            left = self._parse_subtree()
            if self.pos >= self.length or self.text[self.pos] != ",":
                raise NewickError("Expected ',' between children at position %d" % self.pos)
            self.pos += 1
            right = self._parse_subtree()
            if self.pos < self.length and self.text[self.pos] == ",":
                raise NewickError(
                    "Internal nodes must have exactly two children at position %d"
                    % self.pos
                )
            if self.pos >= self.length or self.text[self.pos] != ")":
                raise NewickError("Expected ')' at position %d" % self.pos)
            self.pos += 1
            label = self._read_optional_label()
            self._consume_optional_branch_length()
            return Node(label=label, children=(left, right))

        if self.text[self.pos] in ",);:":
            raise NewickError("Expected a taxon name at position %d" % self.pos)
        name = self._read_token()
        self._consume_optional_branch_length()
        return Node(name=name)

    def _read_optional_label(self):
        if self.pos >= self.length or self.text[self.pos] in ",);:":
            return None
        return self._read_token()

    def _read_token(self):
        start = self.pos
        while (
            self.pos < self.length
            and self.text[self.pos] not in "(),;:"
        ):
            self.pos += 1
        if start == self.pos:
            raise NewickError("Expected a taxon or clade label at position %d" % self.pos)
        return self.text[start:self.pos]

    def _consume_optional_branch_length(self):
        if self.pos >= self.length or self.text[self.pos] != ":":
            return
        if not self.allow_branch_lengths:
            raise NewickError("Branch lengths are not allowed in normalized Newick")
        self.pos += 1
        start = self.pos
        while (
            self.pos < self.length
            and self.text[self.pos] not in "(),;:"
        ):
            self.pos += 1
        if start == self.pos:
            raise NewickError("Empty branch length at position %d" % self.pos)
        token = self.text[start:self.pos]
        if BRANCH_LENGTH_RE.fullmatch(token) is None:
            raise NewickError("Invalid branch length %r" % token)
        self.branch_lengths_stripped = True


def _validate_structure(root, require_labels):
    leaf_names = []
    internal_labels = []
    internal_count = 0

    def walk(node):
        nonlocal internal_count
        if node.is_leaf:
            if not node.name:
                raise NewickError("Leaf without a taxon name")
            leaf_names.append(node.name)
            return
        if len(node.children) != 2:
            raise NewickError("Internal nodes must have exactly two children")
        internal_count += 1
        if node.label:
            internal_labels.append(node.label)
        elif require_labels:
            raise NewickError("Internal node without a clade label")
        for child in node.children:
            walk(child)

    walk(root)
    if len(set(leaf_names)) != len(leaf_names):
        raise NewickError("Duplicate taxon names are not allowed")
    if len(set(internal_labels)) != len(internal_labels):
        raise NewickError("Duplicate clade labels are not allowed")
    return leaf_names, internal_labels, internal_count


def _internal_nodes_postorder(root):
    nodes = []

    def walk(node):
        if node.is_leaf:
            return
        for child in node.children:
            walk(child)
        nodes.append(node)

    walk(root)
    return nodes


def _render_newick(node):
    if node.is_leaf:
        return node.name
    if not node.label:
        raise NewickError("Cannot render an unlabeled internal node")
    children = ",".join(_render_newick(child) for child in node.children)
    return "(" + children + ")" + node.label


def normalize_newick(raw_newick, auto_label_internal=True):
    """Normalize a Newick tree and return (newick, metadata)."""
    text = _without_whitespace(raw_newick)
    parser = _NewickParser(text, allow_branch_lengths=True)
    root = parser.parse()

    _, existing_labels, internal_count = _validate_structure(
        root, require_labels=False
    )
    if len(set(existing_labels)) != len(existing_labels):
        raise NewickError("Duplicate clade labels are not allowed")

    missing_labels = [node for node in _internal_nodes_postorder(root) if not node.label]
    if missing_labels and not auto_label_internal:
        raise NewickError("Internal node without a clade label")

    auto_labeled_nodes = []
    used_labels = set(existing_labels)
    next_label = 1
    for node in _internal_nodes_postorder(root):
        if node.label:
            continue
        while True:
            candidate = "N%d" % next_label
            next_label += 1
            if candidate not in used_labels:
                break
        node.label = candidate
        used_labels.add(candidate)
        auto_labeled_nodes.append(candidate)

    normalized = _render_newick(root) + ";"
    metadata = {
        "branch_lengths_stripped": parser.branch_lengths_stripped,
        "auto_labeled_nodes": auto_labeled_nodes,
        "leaf_count": len(_all_leaves(root)),
        "internal_count": internal_count,
    }
    return normalized, metadata


def parse_newick(newick_str):
    """Strictly parse normalized binary Newick with labeled internal nodes."""
    text = _without_whitespace(newick_str)
    parser = _NewickParser(text, allow_branch_lengths=False)
    root = parser.parse()
    _validate_structure(root, require_labels=True)
    return root


def all_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _all_leaves(root):
    return [node for node in all_nodes(root) if node.is_leaf]


def _all_internal_nodes(root):
    return [node for node in all_nodes(root) if not node.is_leaf]


def _leaf_index(root):
    return {leaf.name: leaf for leaf in _all_leaves(root)}


def _lca(a, b):
    ancestors = set()
    node = a
    while node is not None:
        ancestors.add(node)
        node = node.parent
    node = b
    while node is not None and node not in ancestors:
        node = node.parent
    if node is None:
        raise ValueError("Nodes are not in the same tree")
    return node


def _node_depth(node):
    depth = 0
    while node.parent is not None:
        depth += 1
        node = node.parent
    return depth


def _clade_names(node):
    if node.is_leaf:
        return frozenset((node.name,))
    names = set()
    for child in node.children:
        names.update(_clade_names(child))
    return frozenset(names)


def solve_mrca(root, taxon_a, taxon_b):
    leaves = _leaf_index(root)
    if taxon_a not in leaves or taxon_b not in leaves:
        raise ValueError("Both taxa must exist in the tree")
    if taxon_a == taxon_b:
        raise ValueError("MRCA requires two distinct taxa")
    ancestor = _lca(leaves[taxon_a], leaves[taxon_b])
    if ancestor.is_leaf:
        raise ValueError("MRCA of distinct leaves cannot be a leaf")
    return ancestor.label


def solve_sister(root, taxon_a):
    leaves = _leaf_index(root)
    if taxon_a not in leaves:
        raise ValueError("Taxon does not exist in the tree")
    leaf = leaves[taxon_a]
    parent = leaf.parent
    if parent is None:
        raise ValueError("Taxon has no parent")
    siblings = [child for child in parent.children if child is not leaf]
    if len(siblings) != 1 or not siblings[0].is_leaf:
        raise ValueError("Direct sibling is not a leaf")
    return siblings[0].name


def cherries(root):
    result = []
    for node in _all_internal_nodes(root):
        if len(node.children) == 2 and all(child.is_leaf for child in node.children):
            result.extend(child.name for child in node.children)
    return sorted(result)


def solve_monophyletic(root, taxa):
    leaves = _leaf_index(root)
    taxa = list(taxa)
    if not taxa:
        raise ValueError("At least one taxon is required")
    if len(set(taxa)) != len(taxa):
        raise ValueError("Taxa must be distinct")
    if any(taxon not in leaves for taxon in taxa):
        raise ValueError("All taxa must exist in the tree")
    clade_sets = {_clade_names(node) for node in _all_internal_nodes(root)}
    return "Yes" if frozenset(taxa) in clade_sets else "No"


def solve_distance(root, taxon_a, taxon_b):
    leaves = _leaf_index(root)
    if taxon_a not in leaves or taxon_b not in leaves:
        raise ValueError("Both taxa must exist in the tree")
    if taxon_a == taxon_b:
        raise ValueError("Distance requires two distinct taxa")
    a = leaves[taxon_a]
    b = leaves[taxon_b]
    return _node_depth(a) + _node_depth(b) - 2 * _node_depth(_lca(a, b))


def recompute_answer(task, question, root):
    if task == "mrca":
        match = MRCA_RE.fullmatch(question)
        if not match:
            raise ValueError("Invalid mrca question: %r" % question)
        return str(solve_mrca(root, match.group(1), match.group(2)))

    if task == "sister":
        match = SISTER_RE.fullmatch(question)
        if not match:
            raise ValueError("Invalid sister question: %r" % question)
        return str(solve_sister(root, match.group(1)))

    if task == "monophyletic":
        match = MONO_RE.fullmatch(question)
        if not match:
            raise ValueError("Invalid monophyletic question: %r" % question)
        taxa = [name.strip() for name in match.group(1).split(",")]
        return str(solve_monophyletic(root, taxa))

    if task == "distance":
        match = DISTANCE_RE.fullmatch(question)
        if not match:
            raise ValueError("Invalid distance question: %r" % question)
        return str(solve_distance(root, match.group(1), match.group(2)))

    raise ValueError("Unknown task: %r" % (task,))


def make_prompt(tree, task, question):
    return "Tree: %s\nTask (%s): %s\nAnswer with only the exact value." % (
        tree,
        task,
        question,
    )


def tree_to_dict(node):
    return {
        "id": node.label if node.label is not None else node.name,
        "name": node.name,
        "label": node.label,
        "is_leaf": node.is_leaf,
        "children": [tree_to_dict(child) for child in node.children],
    }


def tree_summary(raw_newick, auto_label_internal=True):
    normalized, metadata = normalize_newick(
        raw_newick, auto_label_internal=auto_label_internal
    )
    root = parse_newick(normalized)
    leaves = sorted(leaf.name for leaf in _all_leaves(root))
    internal_clades = sorted(node.label for node in _all_internal_nodes(root))
    return {
        "normalized_newick": normalized,
        "metadata": metadata,
        "leaves": leaves,
        "internal_clades": internal_clades,
        "cherry_leaves": cherries(root),
        "tree_dict": tree_to_dict(root),
    }
