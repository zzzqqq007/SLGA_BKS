#!/usr/bin/env python3
"""
CMSTPTW Solution Certificate Verifier — compatibility edition

This checker is designed for the certificate files already produced by the
experiments.  It normalizes legacy certificate schemas before verification.

Key convention
--------------
The MANUSCRIPT uses n = number of customers, with nodes V={0,...,n}.
Legacy certificates may instead store n = number of parent-array entries
(nodes including depot).  Therefore the verifier does NOT blindly interpret
the raw certificate field "n".  The parent-array length is the primary source:

    n_nodes     = len(parent)
    n_customers = len(parent) - 1

If a raw n field exists:
  * n == len(parent)     -> interpreted as node count (legacy schema)
  * n + 1 == len(parent) -> interpreted as customer count
  * otherwise            -> reported as a schema warning; parent length wins

The original Solomon/GH instance rows are selected by explicit node IDs
0..n_customers, so a 150-customer case uses 151 rows including depot 0.

The paper's benchmark model is unit-demand and uses unrounded Euclidean
distance as both edge cost and travel time.  The demand/service columns in
the original routing files are parsed for audit but intentionally ignored.

BKS convention
--------------
The manuscript counts a "substantive" BKS improvement only when the relative
improvement is at least 0.01 percent.  The checker therefore reports both:
  * strict improvement: recomputed_cost < BKS (within numerical tolerance)
  * substantive improvement: 100*(BKS-cost)/BKS >= 0.01 by default

Usage from the package directory (Python 3.9 or later, standard library only):
    python verify_solution_compatible.py --cert-dir solutions --instances PATH_TO_INSTANCES

Certificate directories are searched recursively, including the 100/150/200
subdirectories. Supply --bks PATH_TO_REFERENCE_JSON to also check improvements
against a fixed reference table. Original benchmark instances are required.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def is_int_not_bool(x):
    return isinstance(x, int) and not isinstance(x, bool)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def first_present(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return k, d[k]
    return None, None


def numeric_alias(d: dict, keys):
    """Return first finite numeric alias; booleans are deliberately ignored."""
    for k in keys:
        if k in d and is_number(d[k]):
            return k, float(d[k])
        if k in d and isinstance(d[k], str):
            try:
                v = float(d[k].strip())
                if math.isfinite(v):
                    return k, v
            except ValueError:
                pass
    return None, None


# ---------------------------------------------------------------------------
# Certificate schema normalization
# ---------------------------------------------------------------------------

PARENT_KEYS = ("parent", "parents", "parent_array", "parentArray")
INSTANCE_KEYS = (
    "instance_file", "instance", "instance_name", "instanceName",
    "benchmark_instance", "case", "name",
)
Q_KEYS = ("Q", "q", "capacity", "capacity_limit", "capacityLimit")
COST_KEYS = (
    # Prefer fields most likely to contain full-precision claimed values.
    "claimed_cost", "verified_cost", "objective", "objective_value",
    "best_cost", "cost", "verified_new_bks", "new_bks",
)


def build_instance_index(instances_dir: str):
    by_stem = {}
    for root, _dirs, files in os.walk(instances_dir):
        for fn in files:
            path = os.path.join(root, fn)
            stem = os.path.splitext(fn)[0].lower()
            by_stem.setdefault(stem, []).append(path)
    return by_stem


def infer_instance_from_context(cert: dict, cert_path: str, instance_index):
    """
    Resolve the instance even for legacy certificates without instance_file.

    Priority:
      1) known instance-field aliases;
      2) an exact/substring match between an available instance-file stem and
         the certificate filename / string metadata.

    The fallback accepts a result only when the best stem is unique.
    """
    key, raw = first_present(cert, INSTANCE_KEYS)
    if isinstance(raw, str) and raw.strip():
        return raw.strip(), key, None

    haystacks = [os.path.splitext(os.path.basename(cert_path))[0].lower()]
    for v in cert.values():
        if isinstance(v, str):
            haystacks.append(v.lower())
    combined = " ".join(haystacks)

    candidates = []
    for stem, paths in instance_index.items():
        # Boundary-aware first; plain substring as fallback for names such as C1_2_1.
        pattern = r"(?<![a-z0-9])" + re.escape(stem) + r"(?![a-z0-9])"
        if re.search(pattern, combined, flags=re.I) or stem in combined:
            for p in paths:
                candidates.append((len(stem), stem, p))

    if not candidates:
        return None, None, "instance identifier missing and could not be inferred from certificate context"

    # Longest matching stem is safest; require uniqueness at that length.
    candidates.sort(reverse=True)
    best_len = candidates[0][0]
    best = [(stem, p) for L, stem, p in candidates if L == best_len]
    unique_paths = sorted(set(p for _stem, p in best))
    if len(unique_paths) != 1:
        return None, None, (
            "instance identifier missing and filename inference is ambiguous: "
            + ", ".join(unique_paths[:5])
        )

    return unique_paths[0], "inferred_from_certificate_context", None


def normalize_certificate(cert: dict, cert_path: str, instances_dir: str, instance_index):
    errors = []
    warnings = []

    # Parent array -----------------------------------------------------------
    parent_key, parent = first_present(cert, PARENT_KEYS)
    if parent_key is None:
        errors.append(f"certificate has no parent array; tried aliases {PARENT_KEYS}")
        return None, errors, warnings
    if not isinstance(parent, list):
        errors.append(f"{parent_key} must be a list, got {type(parent).__name__}")
        return None, errors, warnings
    if len(parent) < 2:
        errors.append(f"parent array is too short: {len(parent)}")
        return None, errors, warnings

    n_nodes = len(parent)
    n_customers = n_nodes - 1

    # Raw n semantics --------------------------------------------------------
    raw_n_key, raw_n = first_present(
        cert, ("n", "n_nodes", "node_count", "num_nodes", "n_customers", "customer_count")
    )
    n_semantics = "inferred from parent length"

    if raw_n is not None:
        try:
            raw_n_int = int(raw_n)
        except (TypeError, ValueError):
            warnings.append(f"{raw_n_key}={raw_n!r} is not integer-like; ignored")
        else:
            if raw_n_key in ("n_nodes", "node_count", "num_nodes"):
                expected = n_nodes
                n_semantics = f"{raw_n_key}=node count"
                if raw_n_int != expected:
                    warnings.append(
                        f"{raw_n_key}={raw_n_int} but len(parent)={n_nodes}; parent length used"
                    )
            elif raw_n_key in ("n_customers", "customer_count"):
                expected = n_customers
                n_semantics = f"{raw_n_key}=customer count"
                if raw_n_int != expected:
                    warnings.append(
                        f"{raw_n_key}={raw_n_int} but len(parent)-1={n_customers}; parent length used"
                    )
            else:  # ambiguous legacy key "n"
                if raw_n_int == n_nodes:
                    n_semantics = "legacy n interpreted as node count"
                elif raw_n_int == n_customers:
                    n_semantics = "n interpreted as customer count"
                else:
                    warnings.append(
                        f"raw n={raw_n_int} matches neither len(parent)={n_nodes} "
                        f"nor len(parent)-1={n_customers}; parent length used"
                    )

    # Q ---------------------------------------------------------------------
    q_key, q_raw = first_present(cert, Q_KEYS)
    if q_key is None:
        errors.append(f"certificate has no capacity Q; tried aliases {Q_KEYS}")
        return None, errors, warnings
    try:
        Q = float(q_raw)
    except (TypeError, ValueError):
        errors.append(f"{q_key} must be numeric, got {q_raw!r}")
        return None, errors, warnings
    if not math.isfinite(Q) or Q <= 0:
        errors.append(f"Q must be finite and positive, got {Q!r}")
        return None, errors, warnings

    # Instance ---------------------------------------------------------------
    instance_token, instance_source, infer_error = infer_instance_from_context(
        cert, cert_path, instance_index
    )
    if infer_error:
        errors.append(infer_error)
        return None, errors, warnings

    # Resolve token/path to actual file.
    if os.path.isfile(str(instance_token)):
        instance_path = str(instance_token)
        instance_name = os.path.basename(instance_path)
    else:
        instance_path = find_instance_file(str(instance_token), instances_dir, instance_index)
        if instance_path is None:
            errors.append(
                f"could not resolve instance {instance_token!r} under {instances_dir}"
            )
            return None, errors, warnings
        instance_name = os.path.basename(instance_path)

    # Optional claimed objective --------------------------------------------
    cost_key, claimed_cost = numeric_alias(cert, COST_KEYS)

    # If multiple recognized numeric cost fields exist and disagree, warn.
    seen_costs = []
    for k in COST_KEYS:
        if k in cert:
            _, v = numeric_alias({k: cert[k]}, (k,))
            if v is not None:
                seen_costs.append((k, v))
    if len(seen_costs) > 1:
        values = [v for _k, v in seen_costs]
        if max(values) - min(values) > 1e-6:
            warnings.append(
                "recognized cost fields disagree: "
                + ", ".join(f"{k}={v:.12g}" for k, v in seen_costs)
                + f"; using {cost_key}"
            )

    normalized = {
        "parent": parent,
        "parent_source": parent_key,
        "n_nodes": n_nodes,
        "n_customers": n_customers,
        "raw_n": raw_n,
        "raw_n_source": raw_n_key,
        "n_semantics": n_semantics,
        "Q": Q,
        "Q_source": q_key,
        "instance_path": instance_path,
        "instance_file": instance_name,
        "instance_source": instance_source,
        "claimed_cost": claimed_cost,
        "claimed_cost_source": cost_key,
        "claimed_instance_sha256": cert.get("instance_sha256"),
    }
    return normalized, errors, warnings


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def parse_int_row(line: str):
    toks = line.strip().split()
    if len(toks) != 7:
        return None
    try:
        return [int(t) for t in toks]
    except ValueError:
        return None


def load_instance(instance_path: str, n_customers: int):
    """
    Parse the source file and select explicit IDs 0..n_customers inclusive.
    Thus 150 customers => IDs 0..150 => 151 nodes.
    """
    rows_by_id: Dict[int, List[int]] = {}

    with open(instance_path, encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            row = parse_int_row(line)
            if row is None:
                continue
            node_id = row[0]
            if node_id in rows_by_id:
                raise ValueError(
                    f"duplicate node id {node_id} in {instance_path} near line {lineno}"
                )
            rows_by_id[node_id] = row

    expected_ids = list(range(n_customers + 1))
    missing = [i for i in expected_ids if i not in rows_by_id]
    if missing:
        shown = ", ".join(map(str, missing[:10]))
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(
            f"instance lacks required IDs 0..{n_customers}; missing {shown}{suffix}"
        )

    rows = [rows_by_id[i] for i in expected_ids]
    return {
        "coords": [(r[1], r[2]) for r in rows],
        "ready": [r[4] for r in rows],
        "due": [r[5] for r in rows],
        "input_demand": [r[3] for r in rows],
        "input_service": [r[6] for r in rows],
        "sha256": sha256_file(instance_path),
        "n_nodes": len(rows),
        "n_customers": n_customers,
    }


def dist(coords, i: int, j: int) -> float:
    return math.hypot(coords[i][0] - coords[j][0],
                      coords[i][1] - coords[j][1])


# ---------------------------------------------------------------------------
# Structural / feasibility verification
# ---------------------------------------------------------------------------

def validate_parent(parent, n_nodes):
    errors = []

    if len(parent) != n_nodes:
        return [f"parent length {len(parent)} != expected n_nodes {n_nodes}"]

    for i, p in enumerate(parent):
        if not is_int_not_bool(p):
            errors.append(f"parent[{i}] must be an integer, got {p!r}")
    if errors:
        return errors

    if parent[0] not in (-1, 0):
        errors.append(f"depot parent must be -1 or 0, got {parent[0]}")

    for v in range(1, n_nodes):
        p = parent[v]
        if p < 0 or p >= n_nodes:
            errors.append(
                f"invalid parent index parent[{v}]={p}; valid range 0..{n_nodes-1}"
            )
        elif p == v:
            errors.append(f"self-loop at node {v}")

    if errors:
        return errors

    for start in range(1, n_nodes):
        seen = set()
        cur = start
        while cur != 0:
            if cur in seen:
                errors.append(
                    f"cycle detected while tracing node {start}; revisited {cur}"
                )
                break
            seen.add(cur)
            cur = parent[cur]

    return errors


def verify_normalized(norm, instance, feas_tol=1e-7, cost_abs_tol=1e-6):
    errors = []
    parent = norm["parent"]
    coords = instance["coords"]
    ready = instance["ready"]
    due = instance["due"]
    Q = norm["Q"]
    n_nodes = norm["n_nodes"]

    errors.extend(validate_parent(parent, n_nodes))
    if errors:
        return False, {
            "n_nodes": n_nodes,
            "n_customers": n_nodes - 1,
            "instance_sha256": instance["sha256"],
        }, errors

    # Objective
    recomputed_cost = math.fsum(
        dist(coords, parent[v], v) for v in range(1, n_nodes)
    )

    # Unit-demand gate-subtree capacity
    loads = {}
    gate_of = [None] * n_nodes
    for v in range(1, n_nodes):
        cur = v
        while parent[cur] != 0:
            cur = parent[cur]
        gate_of[v] = cur
        loads[cur] = loads.get(cur, 0) + 1

    cap_violations = []
    for g, load in sorted(loads.items()):
        if load > Q:
            cap_violations.append((g, load))
            errors.append(
                f"capacity violation: gate {g} load={load} > Q={Q:g}"
            )

    # Earliest feasible service-start / arrival times
    arrival = [None] * n_nodes
    arrival[0] = 0.0

    def compute_arrival(v):
        chain = []
        cur = v
        while arrival[cur] is None:
            chain.append(cur)
            cur = parent[cur]
        for node in reversed(chain):
            p = parent[node]
            arrival[node] = max(
                float(ready[node]),
                float(arrival[p]) + dist(coords, p, node)
            )
        return float(arrival[v])

    tw_violations = []
    max_lateness = 0.0
    min_tw_slack = math.inf

    for v in range(1, n_nodes):
        a = compute_arrival(v)
        late = a - float(due[v])
        max_lateness = max(max_lateness, late)
        min_tw_slack = min(min_tw_slack, float(due[v]) - a)
        if late > feas_tol:
            tw_violations.append((v, a, due[v], late))

    for v, a, d, late in tw_violations[:20]:
        errors.append(
            f"time-window violation: node {v}, arrival={a:.12f}, "
            f"due={d}, late={late:.6e}"
        )
    if len(tw_violations) > 20:
        errors.append(f"... and {len(tw_violations)-20} more TW violations")

    # Claimed-cost comparison when a recognized exact/verified cost is present
    claimed_cost = norm["claimed_cost"]
    cost_match = None
    if claimed_cost is not None:
        diff = abs(recomputed_cost - claimed_cost)
        cost_match = diff <= cost_abs_tol
        if not cost_match:
            errors.append(
                f"cost mismatch: recomputed={recomputed_cost:.12f}, "
                f"{norm['claimed_cost_source']}={claimed_cost:.12f}, "
                f"diff={diff:.6e} > absolute tolerance {cost_abs_tol:.1e}"
            )

    # Optional instance hash
    hash_match = None
    claimed_hash = norm["claimed_instance_sha256"]
    if claimed_hash:
        hash_match = str(claimed_hash).lower() == instance["sha256"].lower()
        if not hash_match:
            errors.append(
                f"instance SHA-256 mismatch: recomputed={instance['sha256']}, "
                f"certificate={claimed_hash}"
            )

    rec = {
        "recomputed_cost": recomputed_cost,
        "claimed_cost": claimed_cost,
        "claimed_cost_source": norm["claimed_cost_source"],
        "cost_match": cost_match,
        "max_load": max(loads.values()) if loads else 0,
        "Q": Q,
        "capacity_ok": not cap_violations,
        "max_lateness": max_lateness,
        "min_time_window_slack": None if min_tw_slack == math.inf else min_tw_slack,
        "time_window_ok": not tw_violations,
        "n_nodes": n_nodes,
        "n_customers": n_nodes - 1,
        "n_main_subtrees": len(loads),
        "instance_sha256": instance["sha256"],
        "certificate_hash_match": hash_match,
    }
    return len(errors) == 0, rec, errors


# ---------------------------------------------------------------------------
# Instance and BKS lookup
# ---------------------------------------------------------------------------

def find_instance_file(instance_token: str, instances_dir: str, instance_index):
    if not isinstance(instance_token, str):
        return None

    token = instance_token.strip()
    if not token:
        return None

    # Exact / relative path
    exact = os.path.join(instances_dir, token)
    if os.path.isfile(exact):
        return exact

    # Stem lookup
    stem = os.path.splitext(os.path.basename(token))[0].lower()
    matches = instance_index.get(stem, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous instance stem {stem!r}: " + ", ".join(matches[:5])
        )

    # Common extension fallback
    for ext in ("", ".txt", ".TXT", ".dat"):
        c = os.path.join(instances_dir, token + ext)
        if os.path.isfile(c):
            return c
    return None


def q_variants(Q, n_customers):
    vals = []
    qf = float(Q)
    if qf.is_integer():
        vals.extend([str(int(qf)), f"{qf:g}"])
    else:
        vals.append(f"{qf:g}")

    # Q=1000 is the implementation representation of infinity in the paper.
    if qf >= 1000 and n_customers < qf:
        vals.extend(["inf", "INF", "infinity", "∞"])

    return list(dict.fromkeys(vals))


def extract_bks_value(v):
    if is_number(v):
        return float(v)
    if isinstance(v, str):
        try:
            x = float(v)
            if math.isfinite(x):
                return x
        except ValueError:
            return None
    if isinstance(v, dict):
        for k in ("bks", "value", "cost", "reference_bks", "best"):
            if k in v and is_number(v[k]):
                return float(v[k])
    return None


def lookup_bks(bks_map, instance_file, Q, n_customers):
    base = os.path.basename(instance_file)
    stem = os.path.splitext(base)[0]

    candidates = []
    for inst in (instance_file, base, stem):
        for q in q_variants(Q, n_customers):
            # Correct size token for manuscript/database convention:
            # number of CUSTOMERS, not number of parent entries.
            candidates.append(f"{inst}-{q}-{n_customers}")

    # Exact
    for key in candidates:
        if key in bks_map:
            val = extract_bks_value(bks_map[key])
            if val is not None:
                return key, val

    # Case-insensitive
    lower = {str(k).lower(): k for k in bks_map}
    for key in candidates:
        real = lower.get(key.lower())
        if real is not None:
            val = extract_bks_value(bks_map[real])
            if val is not None:
                return real, val

    return None, None


# ---------------------------------------------------------------------------
# Per-file verification
# ---------------------------------------------------------------------------

def verify_one(
    cert_path,
    instances_dir,
    instance_index,
    bks_map=None,
    feas_tol=1e-7,
    cost_tol=1e-6,
    numerical_bks_tol=1e-9,
    substantive_pct=0.01,
    require_substantive=False,
):
    try:
        with open(cert_path, encoding="utf-8") as f:
            cert = json.load(f)
    except Exception as e:
        return {
            "cert": os.path.basename(cert_path),
            "status": "ERROR",
            "errors": [f"cannot read certificate JSON: {e}"],
            "warnings": [],
            "normalized": {},
            "recomputed": {},
        }

    norm, errors, warnings = normalize_certificate(
        cert, cert_path, instances_dir, instance_index
    )
    if norm is None:
        return {
            "cert": os.path.basename(cert_path),
            "status": "ERROR",
            "errors": errors,
            "warnings": warnings,
            "normalized": {},
            "recomputed": {},
        }

    try:
        instance = load_instance(norm["instance_path"], norm["n_customers"])
    except Exception as e:
        return {
            "cert": os.path.basename(cert_path),
            "status": "ERROR",
            "errors": [f"instance parse/selection error: {e}"],
            "warnings": warnings,
            "normalized": norm,
            "recomputed": {},
        }

    ok, rec, verify_errors = verify_normalized(
        norm, instance, feas_tol=feas_tol, cost_abs_tol=cost_tol
    )
    errors.extend(verify_errors)

    # Reference BKS
    if bks_map is not None:
        key, bks = lookup_bks(
            bks_map, norm["instance_file"], norm["Q"], norm["n_customers"]
        )
        rec["reference_bks_key"] = key
        rec["reference_bks"] = bks

        if key is None:
            warnings.append(
                f"no BKS key found for instance={norm['instance_file']}, "
                f"Q={norm['Q']:g}, customers={norm['n_customers']}"
            )
            rec["strict_bks_improvement"] = None
            rec["improvement_abs"] = None
            rec["improvement_pct"] = None
            rec["substantive_bks_improvement"] = None
            if require_substantive:
                errors.append("cannot establish substantive BKS improvement: reference BKS missing")
                ok = False
        else:
            delta = bks - rec["recomputed_cost"]
            pct = 100.0 * delta / bks
            strict = delta > numerical_bks_tol
            substantive = strict and pct >= substantive_pct

            rec["improvement_abs"] = delta
            rec["improvement_pct"] = pct
            rec["strict_bks_improvement"] = strict
            rec["substantive_bks_improvement"] = substantive
            rec["substantive_threshold_pct"] = substantive_pct

            if require_substantive and not substantive:
                errors.append(
                    f"not a substantive BKS improvement: improvement={pct:.9f}% "
                    f"< required {substantive_pct:.6g}%"
                )
                ok = False

    status = "PASS" if ok and not errors else "FAIL"
    return {
        "cert": os.path.basename(cert_path),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "normalized": norm,
        "recomputed": rec,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="CMSTPTW certificate verifier compatible with legacy certificate schemas"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cert", help="single certificate JSON")
    g.add_argument("--cert-dir", help="directory containing certificate JSON files (searched recursively)")

    ap.add_argument("--instances", required=True, help="directory of original instance files")
    ap.add_argument("--bks", help="optional reference BKS JSON map")
    ap.add_argument("--summary", help="optional CSV summary output")
    ap.add_argument("--feas-tol", type=float, default=1e-7)
    ap.add_argument("--cost-tol", type=float, default=1e-6)
    ap.add_argument("--bks-numerical-tol", type=float, default=1e-9)
    ap.add_argument(
        "--substantive-pct",
        type=float,
        default=0.01,
        help="relative BKS-improvement threshold in PERCENT (paper default 0.01%%)",
    )
    ap.add_argument(
        "--require-substantive-improvement",
        action="store_true",
        help="FAIL unless recomputed solution improves the reference BKS by at least --substantive-pct",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.instances):
        print(f"ERROR: --instances directory not found: {args.instances}", file=sys.stderr)
        sys.exit(2)

    instance_index = build_instance_index(args.instances)

    bks_map = None
    if args.bks:
        try:
            with open(args.bks, encoding="utf-8") as f:
                bks_map = json.load(f)
            if not isinstance(bks_map, dict):
                raise ValueError("top-level BKS JSON must be an object/map")
        except Exception as e:
            print(f"ERROR: cannot read BKS JSON: {e}", file=sys.stderr)
            sys.exit(2)

    if args.require_substantive_improvement and bks_map is None:
        print("ERROR: --require-substantive-improvement requires --bks", file=sys.stderr)
        sys.exit(2)

    if args.cert:
        cert_files = [args.cert]
    else:
        cert_files = sorted(glob.glob(os.path.join(args.cert_dir, "**", "*.json"), recursive=True))

    if not cert_files:
        print("ERROR: no certificate JSON files found", file=sys.stderr)
        sys.exit(2)

    results = []
    for cp in cert_files:
        r = verify_one(
            cp,
            args.instances,
            instance_index,
            bks_map=bks_map,
            feas_tol=args.feas_tol,
            cost_tol=args.cost_tol,
            numerical_bks_tol=args.bks_numerical_tol,
            substantive_pct=args.substantive_pct,
            require_substantive=args.require_substantive_improvement,
        )
        results.append(r)

        norm = r["normalized"]
        rec = r["recomputed"]
        line = f"[{r['status']}] {r['cert']}"

        if norm:
            line += (
                f"  inst={norm['instance_file']}"
                f"  Q={norm['Q']:g}"
                f"  customers={norm['n_customers']}"
                f"  nodes={norm['n_nodes']}"
            )
        if rec.get("recomputed_cost") is not None:
            line += f"  cost={rec['recomputed_cost']:.9f}"
        if rec.get("improvement_pct") is not None:
            line += (
                f"  dBKS={rec['improvement_pct']:.6f}%"
                f"  substantive={rec['substantive_bks_improvement']}"
            )
        print(line)

        if norm and norm.get("raw_n") is not None:
            print(
                f"        n-schema: raw {norm['raw_n_source']}={norm['raw_n']}; "
                f"{norm['n_semantics']}"
            )
        if norm and norm.get("claimed_cost_source"):
            print(f"        cost field: {norm['claimed_cost_source']}")

        for w in r["warnings"]:
            print(f"        ? {w}")
        for e in r["errors"]:
            print(f"        ! {e}")

    n_pass = sum(r["status"] == "PASS" for r in results)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    n_err = sum(r["status"] == "ERROR" for r in results)

    # Unique (instance, Q, customers) counts and substantive BKS count
    unique_cases = set()
    substantive_cases = set()
    for r in results:
        norm, rec = r["normalized"], r["recomputed"]
        if norm:
            key = (
                os.path.splitext(norm["instance_file"])[0].lower(),
                float(norm["Q"]),
                int(norm["n_customers"]),
            )
            unique_cases.add(key)
            if rec.get("substantive_bks_improvement") is True:
                substantive_cases.add(key)

    print("\n" + "=" * 78)
    print(
        f"CERTIFICATES: {len(results)}  PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_err}"
    )
    print(f"UNIQUE INSTANCE-Q-SIZE CASES: {len(unique_cases)}")
    if bks_map is not None:
        print(
            f"SUBSTANTIVE BKS IMPROVEMENTS (>= {args.substantive_pct:g}%): "
            f"{len(substantive_cases)}"
        )
    print("=" * 78)

    if args.summary:
        with open(args.summary, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "certificate", "status",
                "instance", "instance_source",
                "Q", "Q_source",
                "raw_n", "raw_n_source", "n_semantics",
                "n_customers", "n_nodes",
                "parent_source",
                "claimed_cost", "claimed_cost_source",
                "recomputed_cost", "cost_match",
                "max_load", "capacity_ok",
                "max_lateness", "min_time_window_slack", "time_window_ok",
                "instance_sha256", "certificate_hash_match",
                "reference_bks_key", "reference_bks",
                "improvement_abs", "improvement_pct",
                "strict_bks_improvement", "substantive_bks_improvement",
                "warnings", "errors",
            ])
            for r in results:
                n = r["normalized"]
                x = r["recomputed"]
                w.writerow([
                    r["cert"], r["status"],
                    n.get("instance_file", ""), n.get("instance_source", ""),
                    n.get("Q", ""), n.get("Q_source", ""),
                    n.get("raw_n", ""), n.get("raw_n_source", ""), n.get("n_semantics", ""),
                    n.get("n_customers", ""), n.get("n_nodes", ""),
                    n.get("parent_source", ""),
                    x.get("claimed_cost", ""), x.get("claimed_cost_source", ""),
                    x.get("recomputed_cost", ""), x.get("cost_match", ""),
                    x.get("max_load", ""), x.get("capacity_ok", ""),
                    x.get("max_lateness", ""), x.get("min_time_window_slack", ""),
                    x.get("time_window_ok", ""),
                    x.get("instance_sha256", ""), x.get("certificate_hash_match", ""),
                    x.get("reference_bks_key", ""), x.get("reference_bks", ""),
                    x.get("improvement_abs", ""), x.get("improvement_pct", ""),
                    x.get("strict_bks_improvement", ""),
                    x.get("substantive_bks_improvement", ""),
                    " | ".join(r["warnings"]),
                    " | ".join(r["errors"]),
                ])
        print(f"Summary written to {args.summary}")

    # ERROR is a schema/file-resolution problem; FAIL is a verification failure.
    sys.exit(0 if n_fail == 0 and n_err == 0 else 1)


if __name__ == "__main__":
    main()
