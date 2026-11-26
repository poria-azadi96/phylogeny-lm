/* PhyloScope Offline Bioinformatics & Newick Engine (Zero-dependency ES6) */
(function (global) {
  "use strict";

  function cleanTaxonName(raw) {
    return String(raw || "Taxon")
      .trim()
      .split(/\s+/)[0]
      .replace(/[^A-Za-z0-9_.-]/g, "_")
      .replace(/^_+|_+$/g, "") || "Taxon";
  }

  function parseFasta(text) {
    const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
    const records = [];
    let currentId = null;
    let seqParts = [];
    const seen = new Set();

    function flush() {
      if (currentId === null) return;
      const seq = seqParts.join("").replace(/[^A-Za-z]/g, "").toUpperCase();
      if (!seq.length) {
        throw new Error("Empty sequence for taxon: " + currentId);
      }
      let name = cleanTaxonName(currentId);
      if (seen.has(name)) {
        let idx = 2;
        while (seen.has(name + "_" + idx)) idx++;
        name = name + "_" + idx;
      }
      seen.add(name);
      records.push({ id: name, sequence: seq });
    }

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      if (line.startsWith(">")) {
        flush();
        currentId = line.slice(1).trim();
        seqParts = [];
      } else {
        if (currentId === null) {
          throw new Error("FASTA must start with a '>' header line.");
        }
        seqParts.push(line);
      }
    }
    flush();
    if (records.length < 4) {
      throw new Error("At least 4 FASTA sequences are required for Neighbor-Joining (found " + records.length + ").");
    }
    return records;
  }

  // Banded Needleman-Wunsch global alignment (fast enough for 16S rRNA ~1500bp in JS)
  function globalAlign(seqA, seqB) {
    const n = seqA.length;
    const m = seqB.length;
    const match = 2;
    const mismatch = -1;
    const gap = -2;

    // Band width covers length difference + generous margin for 16S indels
    const band = Math.max(120, Math.abs(n - m) + 90);
    const width = m + 1;
    const NEG_INF = -1000000000;

    // Use flat Int32Array for speed
    const dp = new Int32Array((n + 1) * width);
    const tb = new Uint8Array((n + 1) * width); // 1=diag, 2=up(gap in B), 3=left(gap in A)
    dp.fill(NEG_INF);

    dp[0] = 0;
    for (let j = 1; j <= Math.min(m, band); j++) {
      dp[j] = j * gap;
      tb[j] = 3;
    }

    for (let i = 1; i <= n; i++) {
      const row = i * width;
      const prevRow = (i - 1) * width;
      const jMin = Math.max(0, i - band);
      const jMax = Math.min(m, i + band);
      if (jMin === 0) {
        dp[row] = i * gap;
        tb[row] = 2;
      }
      const chA = seqA.charCodeAt(i - 1);
      for (let j = Math.max(1, jMin); j <= jMax; j++) {
        const chB = seqB.charCodeAt(j - 1);
        const sDiag = dp[prevRow + j - 1] + (chA === chB ? match : mismatch);
        const sUp = dp[prevRow + j] + gap;
        const sLeft = dp[row + j - 1] + gap;
        if (sDiag >= sUp && sDiag >= sLeft) {
          dp[row + j] = sDiag;
          tb[row + j] = 1;
        } else if (sUp >= sLeft) {
          dp[row + j] = sUp;
          tb[row + j] = 2;
        } else {
          dp[row + j] = sLeft;
          tb[row + j] = 3;
        }
      }
    }

    const outA = [];
    const outB = [];
    let i = n;
    let j = m;
    while (i > 0 || j > 0) {
      const dir = tb[i * width + j];
      if (dir === 1 || (i > 0 && j > 0 && dir === 0)) {
        outA.push(seqA[i - 1]);
        outB.push(seqB[j - 1]);
        i--;
        j--;
      } else if (dir === 2 || j === 0) {
        outA.push(seqA[i - 1]);
        outB.push("-");
        i--;
      } else {
        outA.push("-");
        outB.push(seqB[j - 1]);
        j--;
      }
    }
    return [outA.reverse().join(""), outB.reverse().join("")];
  }

  function k2pDistance(alA, alB) {
    let valid = 0;
    let transitions = 0;
    let transversions = 0;
    const isValid = { A: 1, C: 1, G: 1, T: 1, U: 1 };
    const isPurine = { A: 1, G: 1 };

    for (let i = 0; i < alA.length; i++) {
      let a = alA[i];
      let b = alB[i];
      if (a === "U") a = "T";
      if (b === "U") b = "T";
      if (!isValid[a] || !isValid[b]) continue;
      valid++;
      if (a !== b) {
        if (Boolean(isPurine[a]) === Boolean(isPurine[b])) {
          transitions++;
        } else {
          transversions++;
        }
      }
    }
    if (valid === 0) return 0.75;
    const P = transitions / valid;
    const Q = transversions / valid;
    const term1 = 1.0 - 2.0 * P - Q;
    const term2 = 1.0 - 2.0 * Q;
    if (term1 <= 1e-9 || term2 <= 1e-9) {
      const pDist = Math.min(0.74, (transitions + transversions) / valid);
      return -0.75 * Math.log(Math.max(1e-6, 1.0 - (4.0 / 3.0) * pDist));
    }
    const d = -0.5 * Math.log(term1 * Math.sqrt(term2));
    return d < 0 ? 0.0 : d;
  }

  function computeDistanceMatrix(records) {
    const n = records.length;
    const mat = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const [alA, alB] = globalAlign(records[i].sequence, records[j].sequence);
        const d = k2pDistance(alA, alB);
        mat[i][j] = d;
        mat[j][i] = d;
      }
    }
    return mat;
  }

  function neighborJoining(taxa, distMat) {
    let active = taxa.map((t, idx) => ({
      id: idx,
      label: t,
      isLeaf: true,
      adj: [] // { to, len }
    }));
    const nodes = active.slice();
    let d = distMat.map((row) => row.slice());
    let nextId = taxa.length;

    while (active.length > 2) {
      const n = active.length;
      const r = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += d[i][j];
        r[i] = sum;
      }

      let bestI = 0;
      let bestJ = 1;
      let bestQ = Infinity;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const q = (n - 2) * d[i][j] - r[i] - r[j];
          if (q < bestQ) {
            bestQ = q;
            bestI = i;
            bestJ = j;
          }
        }
      }

      const dij = d[bestI][bestJ];
      const delta = (r[bestI] - r[bestJ]) / (2 * (n - 2));
      const lenI = Math.max(1e-6, 0.5 * dij + delta);
      const lenJ = Math.max(1e-6, 0.5 * dij - delta);

      const u = { id: nextId++, label: null, isLeaf: false, adj: [] };
      nodes.push(u);
      const nodeI = active[bestI];
      const nodeJ = active[bestJ];
      u.adj.push({ to: nodeI, len: lenI });
      nodeI.adj.push({ to: u, len: lenI });
      u.adj.push({ to: nodeJ, len: lenJ });
      nodeJ.adj.push({ to: u, len: lenJ });

      const newActive = [u];
      const remainingIdx = [];
      for (let k = 0; k < n; k++) {
        if (k !== bestI && k !== bestJ) {
          newActive.push(active[k]);
          remainingIdx.push(k);
        }
      }

      const m = newActive.length;
      const newD = Array.from({ length: m }, () => new Array(m).fill(0));
      for (let a = 0; a < remainingIdx.length; a++) {
        const k = remainingIdx[a];
        const duk = Math.max(0, 0.5 * (d[bestI][k] + d[bestJ][k] - dij));
        newD[0][a + 1] = duk;
        newD[a + 1][0] = duk;
      }
      for (let a = 0; a < remainingIdx.length; a++) {
        for (let b = a + 1; b < remainingIdx.length; b++) {
          const val = d[remainingIdx[a]][remainingIdx[b]];
          newD[a + 1][b + 1] = val;
          newD[b + 1][a + 1] = val;
        }
      }
      active = newActive;
      d = newD;
    }

    // Connect last 2 active nodes
    const lastLen = Math.max(1e-6, d[0][1]);
    active[0].adj.push({ to: active[1], len: lastLen });
    active[1].adj.push({ to: active[0], len: lastLen });
    return nodes;
  }

  function midpointRootToNewick(nodes) {
    const leaves = nodes.filter((n) => n.isLeaf);
    function bfs(start) {
      const dist = new Map();
      const parent = new Map();
      const parentLen = new Map();
      dist.set(start.id, 0);
      const q = [start];
      let farNode = start;
      while (q.length) {
        const cur = q.shift();
        const cd = dist.get(cur.id);
        if (cur.isLeaf && cd > dist.get(farNode.id)) farNode = cur;
        for (const e of cur.adj) {
          if (!dist.has(e.to.id)) {
            dist.set(e.to.id, cd + e.len);
            parent.set(e.to.id, cur);
            parentLen.set(e.to.id, e.len);
            q.push(e.to);
          }
        }
      }
      return { farNode, dist, parent, parentLen };
    }

    const r1 = bfs(leaves[0]);
    const r2 = bfs(r1.farNode);
    const total = r2.dist.get(r2.farNode.id);
    const half = total / 2.0;

    let cur = r2.farNode;
    let edgeU = null;
    let edgeV = null;
    let edgeLen = 0;
    let lenFromU = 0;
    while (r2.parent.has(cur.id)) {
      const p = r2.parent.get(cur.id);
      const l = r2.parentLen.get(cur.id);
      const dCur = r2.dist.get(cur.id);
      const dP = r2.dist.get(p.id);
      if (dP <= half && half <= dCur) {
        edgeU = p;
        edgeV = cur;
        edgeLen = l;
        lenFromU = half - dP;
        break;
      }
      cur = p;
    }
    if (!edgeU) {
      edgeU = leaves[0];
      edgeV = leaves[0].adj[0].to;
      edgeLen = leaves[0].adj[0].len;
      lenFromU = edgeLen / 2.0;
    }

    function buildSubtree(node, blockedId, branchLen) {
      if (node.isLeaf) {
        return { label: node.label, isLeaf: true, len: branchLen, children: [] };
      }
      const kids = [];
      for (const e of node.adj) {
        if (e.to.id !== blockedId) {
          kids.push(buildSubtree(e.to, node.id, e.len));
        }
      }
      kids.sort((a, b) => firstLeaf(a).localeCompare(firstLeaf(b)));
      return { label: null, isLeaf: false, len: branchLen, children: kids };
    }

    function firstLeaf(t) {
      return t.isLeaf ? t.label : firstLeaf(t.children[0]);
    }

    const leftTree = buildSubtree(edgeU, edgeV.id, Math.max(1e-6, lenFromU));
    const rightTree = buildSubtree(edgeV, edgeU.id, Math.max(1e-6, edgeLen - lenFromU));
    const rootChildren = [leftTree, rightTree].sort((a, b) => firstLeaf(a).localeCompare(firstLeaf(b)));
    const root = { label: "Root", isLeaf: false, len: null, children: rootChildren };

    // Assign Clade1..CladeK post-order for internal nodes except Root
    let cladeIdx = 1;
    function labelInternal(node, isRoot) {
      if (node.isLeaf) return;
      for (const c of node.children) labelInternal(c, false);
      if (!isRoot) node.label = "Clade" + cladeIdx++;
    }
    labelInternal(root, true);

    function toNewick(node, withLengths) {
      if (node.isLeaf) {
        return withLengths && node.len !== null ? `${node.label}:${node.len.toFixed(6)}` : node.label;
      }
      const inner = node.children.map((c) => toNewick(c, withLengths)).join(",");
      const lbl = node.label || "";
      return withLengths && node.len !== null ? `(${inner})${lbl}:${node.len.toFixed(6)}` : `(${inner})${lbl}`;
    }

    return {
      weightedNewick: toNewick(root, true) + ";",
      normalizedNewick: toNewick(root, false) + ";"
    };
  }

  function reconstructFromFasta(fastaText) {
    const records = parseFasta(fastaText);
    const taxa = records.map((r) => r.id);
    const matrix = computeDistanceMatrix(records);
    const nodes = neighborJoining(taxa, matrix);
    const trees = midpointRootToNewick(nodes);
    return {
      taxa,
      lengths: records.map((r) => r.sequence.length),
      distance_matrix: matrix,
      weighted_newick: trees.weightedNewick,
      normalized_newick: trees.normalizedNewick
    };
  }

  // --- Rooted Binary Newick Parser + Exact Solvers (mirrors phylogeny_utils.py) ---
  function parseNewick(raw) {
    const s = String(raw || "").trim();
    if (!s.endsWith(";")) throw new Error("Newick string must end with ';'.");
    let pos = 0;

    function skipWs() {
      while (pos < s.length && /\s/.test(s[pos])) pos++;
    }

    function readLabel() {
      skipWs();
      let start = pos;
      while (pos < s.length && !/[\s(),:;]/.test(s[pos])) pos++;
      return s.slice(start, pos);
    }

    function skipBranchLen() {
      skipWs();
      if (s[pos] === ":") {
        pos++;
        skipWs();
        while (pos < s.length && !/[\s(),;]/.test(s[pos])) pos++;
      }
    }

    function parseNode() {
      skipWs();
      if (s[pos] === "(") {
        pos++;
        const children = [];
        while (true) {
          children.push(parseNode());
          skipWs();
          if (s[pos] === ",") {
            pos++;
            continue;
          }
          if (s[pos] === ")") {
            pos++;
            break;
          }
          throw new Error("Expected ',' or ')' at position " + pos);
        }
        if (children.length !== 2) {
          throw new Error("Only strictly binary trees are supported (found node with " + children.length + " children).");
        }
        const label = readLabel();
        skipBranchLen();
        return { isLeaf: false, label: label || null, children };
      } else {
        const name = readLabel();
        if (!name) throw new Error("Expected leaf taxon name at position " + pos);
        skipBranchLen();
        return { isLeaf: true, name, label: name, children: [] };
      }
    }

    const root = parseNode();
    skipWs();
    if (s[pos] !== ";") throw new Error("Expected ';' at position " + pos);
    pos++;
    skipWs();
    if (pos !== s.length) throw new Error("Unexpected trailing characters after ';'.");

    // Auto-label unlabeled internal nodes post-order
    let cladeNum = 1;
    const usedLabels = new Set();
    function collect(n) {
      if (n.isLeaf) {
        if (usedLabels.has(n.label)) throw new Error("Duplicate taxon name: " + n.label);
        usedLabels.add(n.label);
      } else {
        n.children.forEach(collect);
        if (n.label) usedLabels.add(n.label);
      }
    }
    collect(root);

    const autoLabeled = [];
    function assignLabels(n, isRoot) {
      if (n.isLeaf) return;
      n.children.forEach((c) => assignLabels(c, false));
      if (!n.label) {
        if (isRoot && !usedLabels.has("Root")) {
          n.label = "Root";
        } else {
          while (usedLabels.has("Clade" + cladeNum)) cladeNum++;
          n.label = "Clade" + cladeNum++;
        }
        usedLabels.add(n.label);
        autoLabeled.push(n.label);
      }
    }
    assignLabels(root, true);

    function toNorm(n) {
      if (n.isLeaf) return n.label;
      return `(${toNorm(n.children[0])},${toNorm(n.children[1])})${n.label}`;
    }

    return {
      root,
      normalizedNewick: toNorm(root) + ";",
      autoLabeled
    };
  }

  function analyzeTree(newickStr) {
    const parsed = parseNewick(newickStr);
    const parent = new Map();
    const depth = new Map();
    const nodeByLabel = new Map();
    const leaves = [];
    const internal = [];

    function walk(n, p, d) {
      parent.set(n.label, p);
      depth.set(n.label, d);
      nodeByLabel.set(n.label, n);
      if (n.isLeaf) leaves.push(n.label);
      else {
        internal.push(n.label);
        n.children.forEach((c) => walk(c, n, d + 1));
      }
    }
    walk(parsed.root, null, 0);

    function leafSet(n) {
      if (n.isLeaf) return [n.label];
      return leafSet(n.children[0]).concat(leafSet(n.children[1]));
    }

    const sisterEligible = [];
    for (const l of leaves) {
      const p = parent.get(l);
      if (p && p.children[0].isLeaf && p.children[1].isLeaf) {
        sisterEligible.push(l);
      }
    }

    function solve(task, taxa) {
      if (task === "mrca") {
        let a = taxa[0], b = taxa[1];
        while (depth.get(a) > depth.get(b)) a = parent.get(a).label;
        while (depth.get(b) > depth.get(a)) b = parent.get(b).label;
        while (a !== b) {
          a = parent.get(a).label;
          b = parent.get(b).label;
        }
        return a;
      }
      if (task === "sister") {
        const p = parent.get(taxa[0]);
        if (!p || !p.children[0].isLeaf || !p.children[1].isLeaf) {
          throw new Error(taxa[0] + " is not in a cherry (both siblings must be leaves).");
        }
        return p.children[0].label === taxa[0] ? p.children[1].label : p.children[0].label;
      }
      if (task === "monophyletic") {
        // LCA of all selected taxa; monophyletic iff leafSet(LCA).length === taxa.length
        let lca = taxa[0];
        for (let i = 1; i < taxa.length; i++) {
          lca = solve("mrca", [lca, taxa[i]]);
        }
        const cladeLeaves = leafSet(nodeByLabel.get(lca));
        return cladeLeaves.length === taxa.length ? "Yes" : "No";
      }
      if (task === "distance") {
        const lca = solve("mrca", taxa);
        const d = depth.get(taxa[0]) + depth.get(taxa[1]) - 2 * depth.get(lca);
        return String(d);
      }
      throw new Error("Unknown task: " + task);
    }

    function buildQuestion(task, taxa) {
      if (task === "mrca") {
        return `What is the most recent common ancestor of ${taxa[0]} and ${taxa[1]}?`;
      }
      if (task === "sister") {
        return `What is the direct sister taxon of ${taxa[0]}?`;
      }
      if (task === "monophyletic") {
        const sorted = taxa.slice().sort();
        return `Do ${sorted.join(", ")} form a monophyletic clade?`;
      }
      if (task === "distance") {
        return `What is the topological edge distance between ${taxa[0]} and ${taxa[1]}?`;
      }
      throw new Error("Unknown task: " + task);
    }

    function buildPrompt(task, taxa) {
      const q = buildQuestion(task, taxa);
      return {
        question: q,
        prompt: `Tree: ${parsed.normalizedNewick}\nTask (${task}): ${q}\nAnswer with only the exact value.`
      };
    }

    return {
      root: parsed.root,
      normalizedNewick: parsed.normalizedNewick,
      autoLabeled: parsed.autoLabeled,
      leaves,
      internal,
      sisterEligible,
      solve,
      buildPrompt
    };
  }

  global.PhyloEngine = {
    parseFasta,
    reconstructFromFasta,
    parseNewick,
    analyzeTree
  };
})(window);
