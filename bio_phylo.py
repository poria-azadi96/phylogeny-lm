#!/usr/bin/env python3
"""FASTA parsing, Kimura 2-parameter distances, and Neighbor-Joining trees."""

import math
import re

from Bio.Align import PairwiseAligner

from phylogeny_utils import normalize_newick, tree_summary


TAXON_PATTERN = r"[A-Za-z][A-Za-z0-9_]*"
TAXON_RE = re.compile(TAXON_PATTERN)
VALID_NUCLEOTIDES = set("ACGTURYKMSWBDHVN")
CANONICAL_NUCLEOTIDES = set("ACGT")
TRANSITION_PAIRS = (frozenset(("A", "G")), frozenset(("C", "T")))


def _clean_taxon_id(header):
    stripped = header.strip()
    if not stripped:
        raise ValueError("FASTA header is empty")
    token = stripped.split()[0]
    token = token.split("|")[0]
    token = token.replace("-", "_").replace(".", "_")
    token = re.sub(r"[^A-Za-z0-9_]", "_", token)
    token = token.strip("_")
    if not token:
        raise ValueError("Could not extract a taxon identifier from %r" % header)
    if token[0].isdigit():
        token = "Taxon_" + token
    if TAXON_RE.fullmatch(token) is None:
        raise ValueError("Invalid taxon identifier %r from header %r" % (token, header))
    return token


def _validate_sequence(sequence, taxon_id):
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError("Sequence for %s is empty" % taxon_id)
    invalid = sorted(set(cleaned) - VALID_NUCLEOTIDES)
    if invalid:
        raise ValueError(
            "Sequence for %s contains invalid nucleotide letters: %s"
            % (taxon_id, "".join(invalid))
        )
    return cleaned.replace("U", "T")


def parse_fasta(fasta_text):
    """Parse multi-FASTA text into normalized records."""
    if not isinstance(fasta_text, str):
        raise ValueError("FASTA input must be a string")

    records = []
    header = None
    chunks = []

    def flush():
        if header is None:
            return
        taxon_id = _clean_taxon_id(header)
        sequence = _validate_sequence("".join(chunks), taxon_id)
        records.append(
            {"id": taxon_id, "header": header.strip(), "sequence": sequence}
        )

    for raw_line in fasta_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            chunks = []
        else:
            if header is None:
                raise ValueError("Found sequence data before the first FASTA header")
            chunks.append(line)
    flush()

    if len(records) < 3:
        raise ValueError("At least 3 FASTA sequences are required")

    seen = set()
    for record in records:
        if record["id"] in seen:
            raise ValueError("Duplicate taxon identifier %r" % record["id"])
        seen.add(record["id"])

    return records


def _normalize_sequence(sequence):
    return "".join(str(sequence).split()).upper().replace("U", "T")


def _kimura_distance(transitions, transversions, n_sites):
    if n_sites <= 0:
        return 0.0
    p = transitions / n_sites
    q = transversions / n_sites
    x = 1.0 - 2.0 * p - q
    y = 1.0 - 2.0 * q
    if x <= 1e-6 or y <= 1e-6:
        p_distance = (transitions + transversions) / n_sites
        jc_argument = 1.0 - (4.0 / 3.0) * p_distance
        if jc_argument <= 1e-6:
            distance = 2.5
        else:
            distance = -0.75 * math.log(jc_argument)
    else:
        distance = -0.5 * math.log(x) - 0.25 * math.log(y)
    return min(2.5, max(0.0, distance))


def compute_k2p_distance(seq_a, seq_b):
    """Return (Kimura 2-parameter distance, percent identity, aligned sites)."""
    seq_a = _normalize_sequence(seq_a)
    seq_b = _normalize_sequence(seq_b)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -1.0

    alignment = aligner.align(seq_a, seq_b)[0]

    n_sites = 0
    matches = 0
    transitions = 0
    transversions = 0

    for (a_start, a_end), (b_start, b_end) in zip(*alignment.aligned):
        block_a = seq_a[a_start:a_end]
        block_b = seq_b[b_start:b_end]
        for base_a, base_b in zip(block_a, block_b):
            if base_a not in CANONICAL_NUCLEOTIDES:
                continue
            if base_b not in CANONICAL_NUCLEOTIDES:
                continue
            n_sites += 1
            if base_a == base_b:
                matches += 1
            elif frozenset((base_a, base_b)) in TRANSITION_PAIRS:
                transitions += 1
            else:
                transversions += 1

    if n_sites == 0:
        return (0.0, 0.0, 0)

    distance = _kimura_distance(transitions, transversions, n_sites)
    identity = 100.0 * matches / n_sites
    return (round(distance, 6), round(identity, 2), n_sites)


def neighbor_joining(labels, dist_matrix):
    """Build an unrooted tree with exact Saitou-Nei Neighbor-Joining."""
    labels = list(labels)
    n = len(labels)
    if n < 3:
        raise ValueError("Neighbor-Joining requires at least 3 taxa")
    if len(dist_matrix) != n or any(len(row) != n for row in dist_matrix):
        raise ValueError("Distance matrix must be square and match the labels")

    used_names = set(labels)
    internal_counter = 0

    def new_internal_name():
        nonlocal internal_counter
        while True:
            internal_counter += 1
            candidate = "NJ_%d" % internal_counter
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate

    distances = {label: {} for label in labels}
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            distances[label_i][label_j] = float(dist_matrix[i][j])

    adjacency = {label: {} for label in labels}
    active = list(labels)

    while len(active) > 2:
        m = len(active)
        row_sums = {
            a: sum(distances[a][b] for b in active if b != a) for a in active
        }

        best = None
        for first in range(m):
            for second in range(first + 1, m):
                a = active[first]
                b = active[second]
                q_value = (m - 2) * distances[a][b] - row_sums[a] - row_sums[b]
                if (
                    best is None
                    or q_value < best[0] - 1e-12
                    or (
                        abs(q_value - best[0]) <= 1e-12
                        and (a, b) < (best[1], best[2])
                    )
                ):
                    best = (q_value, a, b)

        _, a, b = best
        d_ab = distances[a][b]
        limb_a = 0.5 * d_ab + (row_sums[a] - row_sums[b]) / (2.0 * (m - 2))
        limb_b = d_ab - limb_a
        limb_a = max(0.0, limb_a)
        limb_b = max(0.0, limb_b)

        internal = new_internal_name()
        adjacency[internal] = {a: limb_a, b: limb_b}
        adjacency[a][internal] = limb_a
        adjacency[b][internal] = limb_b

        remaining = [node for node in active if node not in (a, b)]
        distances[internal] = {internal: 0.0}
        for other in remaining:
            d_internal_other = 0.5 * (
                distances[a][other] + distances[b][other] - d_ab
            )
            distances[internal][other] = d_internal_other
            distances[other][internal] = d_internal_other

        active = remaining + [internal]

    a, b = active
    final_length = max(0.0, float(distances[a][b]))
    adjacency[a][b] = final_length
    adjacency[b][a] = final_length
    return adjacency


class _RootedNode:
    __slots__ = ("name", "label", "length", "children")

    def __init__(self, name=None, label=None, length=0.0, children=None):
        self.name = name
        self.label = label
        self.length = float(length) if length is not None else 0.0
        self.children = list(children) if children else []

    @property
    def is_leaf(self):
        return not self.children


def _diameter(adjacency, labels):
    leaf_set = set(labels)
    best = None
    for start in sorted(leaf_set):
        stack = [(start, None, 0.0, (start,))]
        while stack:
            node, parent, distance, path = stack.pop()
            if node in leaf_set and node != start:
                candidate = (distance, tuple(sorted((start, node))), path)
                if best is None or candidate > best:
                    best = candidate
            for neighbor in sorted(adjacency[node]):
                if neighbor == parent:
                    continue
                stack.append(
                    (
                        neighbor,
                        node,
                        distance + float(adjacency[node][neighbor]),
                        path + (neighbor,),
                    )
                )
    if best is None:
        raise ValueError("Could not find two leaf tips in the tree")
    return best[0], best[2]


def _format_length(value):
    value = max(0.0, float(value))
    text = ("%.6f" % value).rstrip("0").rstrip(".")
    if text in ("", "-0"):
        return "0"
    return text


def _render_weighted_newick(node, is_root=False):
    if node.is_leaf:
        return "%s:%s" % (node.name, _format_length(node.length))
    body = "(" + ",".join(
        _render_weighted_newick(child) for child in node.children
    ) + ")" + node.label
    if is_root:
        return body
    return body + ":" + _format_length(node.length)


def midpoint_root(adjacency, labels):
    """Root an unrooted tree at the midpoint of its weighted leaf diameter."""
    labels = list(labels)
    if not labels:
        raise ValueError("At least one label is required")
    leaf_set = set(labels)
    missing = leaf_set - set(adjacency)
    if missing:
        raise ValueError("Adjacency is missing labels: %s" % sorted(missing))

    total_length, path = _diameter(adjacency, labels)
    midpoint = total_length / 2.0 if total_length > 0 else 0.0

    split_index = None
    cumulative = 0.0
    for index in range(len(path) - 1):
        u = path[index]
        v = path[index + 1]
        edge_length = float(adjacency[u][v])
        if edge_length > 0 and cumulative <= midpoint <= cumulative + edge_length + 1e-12:
            split_index = index
            distance_from_u = min(max(midpoint - cumulative, 0.0), edge_length)
            break
        cumulative += edge_length

    if split_index is None:
        split_index = len(path) - 2
        u = path[split_index]
        v = path[split_index + 1]
        edge_length = float(adjacency[u][v])
        distance_from_u = edge_length / 2.0

    u = path[split_index]
    v = path[split_index + 1]
    edge_length = float(adjacency[u][v])
    distance_from_u = min(max(distance_from_u, 0.0), edge_length)
    distance_from_v = max(0.0, edge_length - distance_from_u)

    def build(node, parent, incoming_length):
        if node in leaf_set:
            return _RootedNode(name=node, length=incoming_length)
        children = []
        for neighbor in sorted(adjacency[node]):
            if neighbor == parent:
                continue
            children.append(
                build(neighbor, node, float(adjacency[node][neighbor]))
            )
        if len(children) != 2:
            raise ValueError(
                "Unrooted tree is not binary at internal node %s" % node
            )
        return _RootedNode(length=incoming_length, children=children)

    left = build(u, v, distance_from_u)
    right = build(v, u, distance_from_v)
    root = _RootedNode(children=[left, right])

    counter = 0

    def assign_labels(node, is_root):
        nonlocal counter
        if node.is_leaf:
            return
        for child in node.children:
            assign_labels(child, False)
        if is_root:
            node.label = "Root"
        else:
            counter += 1
            node.label = "Clade%d" % counter

    assign_labels(root, True)
    return root


def reconstruct_tree_from_fasta(fasta_text):
    """Run FASTA -> K2P -> Neighbor-Joining -> midpoint rooting."""
    records = parse_fasta(fasta_text)
    labels = [record["id"] for record in records]
    sequences = [record["sequence"] for record in records]
    count = len(labels)

    distance_matrix = [[0.0] * count for _ in range(count)]
    identity_matrix = [[0.0] * count for _ in range(count)]
    aligned_sites_matrix = [[0] * count for _ in range(count)]

    for i in range(count):
        for j in range(i, count):
            if i == j:
                distance = 0.0
                identity = 100.0
                sites = sum(1 for base in sequences[i] if base in CANONICAL_NUCLEOTIDES)
            else:
                distance, identity, sites = compute_k2p_distance(
                    sequences[i], sequences[j]
                )
            distance_matrix[i][j] = distance_matrix[j][i] = distance
            identity_matrix[i][j] = identity_matrix[j][i] = identity
            aligned_sites_matrix[i][j] = aligned_sites_matrix[j][i] = sites

    adjacency = neighbor_joining(labels, distance_matrix)
    rooted = midpoint_root(adjacency, labels)
    weighted_newick = _render_weighted_newick(rooted, is_root=True) + ";"
    normalized_newick, _ = normalize_newick(weighted_newick)

    return {
        "taxa": labels,
        "lengths": {
            record["id"]: len(record["sequence"]) for record in records
        },
        "distance_matrix": distance_matrix,
        "identity_matrix": identity_matrix,
        "aligned_sites_matrix": aligned_sites_matrix,
        "weighted_newick": weighted_newick,
        "normalized_newick": normalized_newick,
        "tree_summary": tree_summary(normalized_newick),
    }
