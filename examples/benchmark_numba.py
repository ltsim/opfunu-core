#!/usr/bin/env python
"""Benchmark: original NumPy vs Numba-guvectorize across configs, dims and populations.

Usage:
    uv run python examples/benchmark_numba.py [options]

Options:
    --mode {compare,numba,numpy}   compare = numba matrix + pristine baseline (default)
    --dims 1,10,30,50,...          dimensions to sweep (default: 1..3000 set below)
    --pops 1,16,128,1024           batch populations (full list only at anchor dims)
    --anchors 30,100,500           dims where the full pop list runs (other dims: 256 only)
    --classes Ackley01,F12005      optional filter (default: built-in CEC roster)
    --budget 3.0                   seconds budget per timing cell
    --out FILE                     raw results JSON (appended incrementally, resumable)
    --resume                       skip (class, dim, config) cells already in --out
    --orig-src DIR                 pristine checkout for the numpy baseline

Only public API is used, plus feature-detected Numba hooks, so the same file
runs unmodified against the pristine tree for the baseline. Missing data
files for a (class, dim) cell are converted to skip-rows instead of killing
the process (upstream calls exit(1) there).
"""

import argparse
import inspect
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROSTER = [
    ("opfunu.name_based.a_func", "Ackley01"),
    ("opfunu.name_based.g_func", "Griewank"),
    ("opfunu.name_based.q_func", "Quartic"),
    ("opfunu.cec_based.cec2005", "F12005"),
    ("opfunu.cec_based.cec2005", "F152005"),
    ("opfunu.cec_based.cec2008", "F12008"),
    ("opfunu.cec_based.cec2008", "F72008"),
    ("opfunu.cec_based.cec2010", "F12010"),
    ("opfunu.cec_based.cec2010", "F102010"),
    ("opfunu.cec_based.cec2013", "F12013"),
    ("opfunu.cec_based.cec2013", "F172013"),
    ("opfunu.cec_based.cec2014", "F12014"),
    ("opfunu.cec_based.cec2014", "F172014"),
    ("opfunu.cec_based.cec2015", "F12015"),
    ("opfunu.cec_based.cec2015", "F142015"),
    ("opfunu.cec_based.cec2017", "F12017"),
    ("opfunu.cec_based.cec2017", "F222017"),
    ("opfunu.cec_based.cec2019", "F12019"),
    ("opfunu.cec_based.cec2019", "F32019"),
    ("opfunu.cec_based.cec2020", "F12020"),
    ("opfunu.cec_based.cec2020", "F102020"),
    ("opfunu.cec_based.cec2021", "F12021"),
    ("opfunu.cec_based.cec2021", "F102021"),
    ("opfunu.cec_based.cec2022", "F12022"),
    ("opfunu.cec_based.cec2022", "F32022"),
]

DEFAULT_DIMS = [1, 10, 30, 50, 100, 120, 300, 500, 1000, 3000]
DEFAULT_POPS = [1, 16, 128, 1024]
DEFAULT_ANCHORS = [30, 100, 500]
CONFIGS = [
    {"parallel": False, "fastmath": True},
    {"parallel": False, "fastmath": False},
    {"parallel": True, "fastmath": True},
    {"parallel": True, "fastmath": False},
]


def patch_exit():
    """Upstream load_matrix_data calls exit(1) on missing files; make it raisable."""
    try:
        from opfunu.benchmark.cec import CecBenchmark
    except ImportError:
        return
    orig = CecBenchmark.load_matrix_data

    def safe(self, filename):
        try:
            return orig(self, filename)
        except SystemExit as e:
            raise FileNotFoundError(filename) from e

    CecBenchmark.load_matrix_data = safe


def make(cls, ndim, parallel, fastmath):
    sig = inspect.signature(cls.__init__)
    kw = {}
    if "parallel" in sig.parameters:
        kw = {"parallel": parallel, "fastmath": fastmath}
    return cls(ndim=ndim, **kw)


def time_unit(fn, budget, min_reps=5):
    best = float("inf")
    reps, elapsed = min_reps, 0.0
    while True:
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        e = time.perf_counter() - t
        best = min(best, e / reps)
        elapsed += e
        if elapsed >= budget or reps >= 1_000_000:
            break
        reps *= 4
    return best


def bench_cell(mod_name, cls_name, dim, cfg, pops, budget, actual_seen):
    mod = __import__(mod_name, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    try:
        t0 = time.perf_counter()
        f = make(cls, dim, cfg["parallel"], cfg["fastmath"])
        t_inst = time.perf_counter() - t0
    except Exception as e:
        return [
            {
                "class": cls_name,
                "dim": dim,
                "config": cfg,
                "status": f"unsupported ({type(e).__name__})",
            }
        ]
    # Fixed-dimension problems ignore the requested ``dim`` and use their own
    # default; benchmark the dimension the instance actually reports, and only
    # once per (class, actual ndim, config) so a fixed problem is not re-timed
    # under every requested dim.
    actual = f.ndim
    akey = (cls_name, actual, json.dumps(cfg))
    if actual != dim:
        if akey in actual_seen:
            return [
                {
                    "class": cls_name,
                    "dim": dim,
                    "ndim": actual,
                    "config": cfg,
                    "status": f"dim-fixed (ndim={actual})",
                }
            ]
        actual_seen.add(akey)
    try:
        info = {
            "dim_default": f.dim_default,
            "dim_max": getattr(f, "dim_max", None),
            "dim_supported": getattr(f, "dim_supported", None),
            "f_bias": float(getattr(f, "f_bias", float("nan"))),
            "paras": sorted(getattr(f, "paras", {}).keys()),
        }
    except Exception:
        info = {}
    rows = []
    rng = np.random.default_rng(7)
    lb = np.asarray(f.lb, dtype=float)
    ub = np.asarray(f.ub, dtype=float)
    pts = lb + rng.uniform(0, 1, (8, actual)) * (ub - lb)
    for p in pts[:2]:
        f.evaluate(p)
    it = [0]

    def single():
        p = pts[it[0] % len(pts)]
        it[0] += 1
        f.evaluate(p)

    try:
        us = time_unit(single, budget) * 1e6
        status = "ok"
    except Exception as e:
        us, status = None, f"eval-failed ({type(e).__name__})"
    base = {
        "class": cls_name,
        "dim": dim,
        "ndim": actual,
        "config": cfg,
        "status": status,
        "compiled": bool(getattr(f, "numba_compiled", False)),
        "t_inst_s": t_inst,
        "single_us": us,
        "info": info,
    }
    rows.append(base)
    can_batch = bool(getattr(f, "numba_compiled", False))
    for pop in pops:
        X = lb + rng.uniform(0, 1, (pop, actual)) * (ub - lb)

        def loop():
            for row in X:
                f.evaluate(row)

        try:
            loop_ms = time_unit(loop, budget) * 1e3
        except Exception:
            loop_ms = None
        gu_ms = None
        if can_batch:
            try:
                f._evaluate_batch(np.ascontiguousarray(X[:1]))  # warmup
                gu_ms = time_unit(lambda: f._evaluate_batch(np.ascontiguousarray(X)), budget) * 1e3
            except Exception:
                gu_ms = None
        rows.append(
            {
                "class": cls_name,
                "dim": dim,
                "ndim": actual,
                "config": cfg,
                "pop": pop,
                "status": status,
                "compiled": base["compiled"],
                "loop_ms": loop_ms,
                "gufunc_ms": gu_ms,
            }
        )
    return rows


def run_matrix(dims, pops, anchors, classes, configs, budget, out, resume):
    import opfunu  # noqa: E402

    print(f"tree: {opfunu.__file__}", file=sys.stderr)
    done = set()
    rows: list = []
    if resume and out.exists():
        try:
            old = json.loads(out.read_text())
            rows = old if isinstance(old, list) else []
            for r in rows:
                if "pop" in r:
                    done.add((r["class"], r["dim"], json.dumps(r["config"]), r["pop"]))
                else:
                    done.add((r["class"], r["dim"], json.dumps(r["config"]), -1))
        except Exception:
            pass
    patch_exit()
    roster = [(m, c) for m, c in ROSTER if not classes or c in classes]
    actual_seen: set = set()
    for mod_name, cls_name in roster:
        for dim in dims:
            pops_here = pops if dim in anchors else [256]
            for cfg in configs:
                key = (cls_name, dim, json.dumps(cfg), -1)
                new_rows = []
                if key not in done:
                    new_rows = bench_cell(mod_name, cls_name, dim, cfg, pops_here, budget, actual_seen)
                    rows.extend(new_rows)
                    out.write_text(json.dumps(rows))
                print(
                    f"{cls_name} d={dim} p={cfg['parallel']} fm={cfg['fastmath']} -> "
                    f"{new_rows[0]['status'] if new_rows else 'resumed'}",
                    file=sys.stderr,
                )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["compare", "numba", "numpy"], default="compare")
    ap.add_argument("--dims", default=",".join(map(str, DEFAULT_DIMS)))
    ap.add_argument("--pops", default=",".join(map(str, DEFAULT_POPS)))
    ap.add_argument("--anchors", default=",".join(map(str, DEFAULT_ANCHORS)))
    ap.add_argument("--classes", default="")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--out", default="/tmp/opencode/bench_sweep.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--orig-src", default="/tmp/opencode/orig/src")
    args = ap.parse_args()
    dims = [int(d) for d in args.dims.split(",")]
    pops = [int(p) for p in args.pops.split(",")]
    anchors = [int(a) for a in args.anchors.split(",")]
    classes = [c for c in args.classes.split(",") if c]
    out = pathlib.Path(args.out)
    if args.mode in ("numba", "numpy"):
        cfgs = CONFIGS if args.mode == "numba" else [{"parallel": False, "fastmath": True}]
        rows = run_matrix(dims, pops, anchors, classes, cfgs, args.budget, out, args.resume)
        if args.mode == "numpy":
            print(json.dumps(rows))
        else:
            print(f"wrote {len(rows)} rows to {out}")
        return
    me = subprocess.run(
        [
            sys.executable,
            __file__,
            "--mode",
            "numba",
            "--dims",
            args.dims,
            "--pops",
            args.pops,
            "--anchors",
            args.anchors,
            "--classes",
            args.classes,
            "--budget",
            str(args.budget),
            "--out",
            str(out),
            *(["--resume"] if args.resume else []),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    print(me.stderr, file=sys.stderr)
    base = subprocess.run(
        [
            sys.executable,
            __file__,
            "--mode",
            "numpy",
            "--dims",
            args.dims,
            "--pops",
            args.pops,
            "--anchors",
            args.anchors,
            "--classes",
            args.classes,
            "--budget",
            str(args.budget),
            "--out",
            str(out.with_name(out.stem + ".numpy.json")),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": args.orig_src, "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    numba_rows = json.loads(out.read_text())
    numpy_rows = json.loads(base.stdout)
    nlookup: dict = {}
    for r in numpy_rows:
        if "pop" in r or r.get("single_us") is None:
            continue
        nlookup[(r["class"], r.get("ndim", r["dim"]))] = r
    print(
        f"\n{'problem':<10} {'dim':>5} "
        + "".join(f"{('P' if c['parallel'] else 's') + ('F' if c['fastmath'] else 'f'):>11}" for c in CONFIGS)
    )
    seen: list = []
    for r in numba_rows:
        if "pop" in r:
            continue
        key = (r["class"], r.get("ndim", r["dim"]))
        if key not in seen:
            seen.append(key)
    for cls_name, ndim in seen:
        if classes and cls_name not in classes:
            continue
        base_row = nlookup.get((cls_name, ndim))
        if base_row is None or base_row.get("single_us") is None:
            continue
        cells = []
        for cfg in CONFIGS:
            hit = [
                r
                for r in numba_rows
                if "pop" not in r
                and r["class"] == cls_name
                and r.get("ndim", r["dim"]) == ndim
                and r["config"] == cfg
                and r.get("single_us")
            ]
            if hit and base_row["single_us"]:
                cells.append(f"{base_row['single_us'] / hit[0]['single_us']:>10.1f}x")
            else:
                st = hit[0]["status"] if hit else "n/a"
                cells.append(f"{st[:10]:>10}")
        print(f"{cls_name:<10} {ndim:>5} " + "".join(cells))


if __name__ == "__main__":
    main()
