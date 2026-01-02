#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Krylov (partial) eigen-solver benchmark: CPU (SciPy) vs GPU (CuPyX) — per-dtype comparison

Key design:
  - Sparse only (CSR). No dense mode.
  - For each family and each sweep, plot CPU vs GPU for ONE dtype per figure.

Outputs (under results/krylov_cpu_gpu_<host>_<tag>/):
  - log file: environment + per-case results
  - CSV file: one row per benchmark point
  - PRB-style figures (PDF + PNG):
      For each (family, sweep): *_time.pdf/png and *_speedup.pdf/png
"""

import os
import sys
import time
import gc
import platform
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---- SciPy (CPU)
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh as sp_eigsh
from scipy.sparse.linalg import eigs as sp_eigs

try:
    import psutil
except ImportError:
    psutil = None

# ---- CuPy (GPU)
try:
    import cupy as cp
    import cupyx
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    from cupyx.scipy.sparse.linalg import eigsh as gpu_eigsh
    try:
        from cupyx.scipy.sparse.linalg import eigs as gpu_eigs
        HAS_GPU_EIGS = True
    except Exception:
        gpu_eigs = None
        HAS_GPU_EIGS = False
except ImportError:
    cp = None
    cupyx = None
    gpu_csr_matrix = None
    gpu_eigsh = None
    gpu_eigs = None
    HAS_GPU_EIGS = False


# =========================
# Utils: logging / system
# =========================
def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def open_log(log_path: Path):
    return open(log_path, "w", encoding="utf-8")


def log_print(f, s: str):
    print(s)
    f.write(s + "\n")
    f.flush()


def rss_bytes():
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss


def bytes_to_mib(x):
    return None if x is None else x / (1024**2)


def dtype_name(dtype) -> str:
    return np.dtype(dtype).name


def get_numpy_scipy_info() -> str:
    try:
        import numpy as _np
        from io import StringIO
        import contextlib
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            _np.show_config()
        np_cfg = buf.getvalue().strip()
        return f"NumPy {_np.__version__}\nSciPy {scipy.__version__}\n\n[NumPy config]\n{np_cfg}"
    except Exception as e:
        return f"(NumPy/SciPy info unavailable: {e})"


def get_gpu_info() -> str:
    if cp is None:
        return "CuPy not installed"
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], (bytes, bytearray)) else props["name"]
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        return (
            f"CuPy {cp.__version__}\n"
            f"GPU: {name}, device_id={dev.id}\n"
            f"VRAM: total={total_mem/(1024**3):.2f} GiB, free={free_mem/(1024**3):.2f} GiB\n"
            f"CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}\n"
            f"cupyx.scipy.sparse.linalg.eigs available: {HAS_GPU_EIGS}"
        )
    except Exception as e:
        return f"(GPU info unavailable: {e})"


# =========================
# Thread control (CPU)
# =========================
def set_cpu_threads(n_threads: Optional[int]):
    if n_threads is None:
        return
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n_threads)
    except Exception:
        pass


# =========================
# PRB-friendly plotting
# =========================
def prb_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    })


# =========================
# Sparse matrix generators
# =========================
def make_real_symmetric_sparse(N: int, density: float, dtype=np.float64, seed=0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    nnz = max(1, int(density * N * N))
    rows = rng.integers(0, N, size=nnz, endpoint=False)
    cols = rng.integers(0, N, size=nnz, endpoint=False)
    data = rng.standard_normal(nnz).astype(dtype, copy=False)
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    A = 0.5 * (A + A.T)
    return A.tocsr()


def make_complex_general_sparse(N: int, density: float, dtype=np.complex128, seed=0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    nnz = max(1, int(density * N * N))
    rows = rng.integers(0, N, size=nnz, endpoint=False)
    cols = rng.integers(0, N, size=nnz, endpoint=False)
    dr = rng.standard_normal(nnz)
    di = rng.standard_normal(nnz)
    data = (dr + 1j * di).astype(dtype, copy=False)
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    return A.tocsr()


# =========================
# Residual check
# =========================
def residual_norm_cpu(A_csr: csr_matrix, w: np.ndarray, V: np.ndarray) -> float:
    An = np.linalg.norm(A_csr.data) + 1e-30
    AV = A_csr @ V
    R = AV - V * w[np.newaxis, :]
    rn = np.linalg.norm(R, axis=0)
    vn = np.linalg.norm(V, axis=0) + 1e-30
    return float(np.max(rn / (An * vn + 1e-30)))


def residual_norm_gpu(A_gpu, w_gpu, V_gpu) -> float:
    if cp is None:
        return np.nan
    An = cp.linalg.norm(A_gpu.data) + 1e-30
    AV = A_gpu @ V_gpu
    R = AV - V_gpu * w_gpu[cp.newaxis, :]
    rn = cp.linalg.norm(R, axis=0)
    vn = cp.linalg.norm(V_gpu, axis=0) + 1e-30
    val = cp.max(rn / (An * vn + 1e-30))
    return float(val.get())


# =========================
# Timing helpers
# =========================
def time_cpu(callable_fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callable_fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        callable_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def time_gpu(callable_fn, warmup: int, repeats: int) -> float:
    if cp is None:
        return np.nan
    for _ in range(warmup):
        callable_fn()
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    times_ms = []
    for _ in range(repeats):
        start.record()
        callable_fn()
        end.record()
        end.synchronize()
        times_ms.append(cp.cuda.get_elapsed_time(start, end))
    return float(np.median(times_ms) / 1000.0)


# =========================
# Krylov solvers
# =========================
def cpu_eigsh(A_csr: csr_matrix, k: int, tol: float, which: str = "SA"):
    return sp_eigsh(A_csr, k=k, which=which, tol=tol)


def cpu_eigs(A_csr: csr_matrix, k: int, tol: float, which: str = "LM"):
    return sp_eigs(A_csr, k=k, which=which, tol=tol)


def gpu_eigsh_wrap(A_gpu, k: int, tol: float, which: str = "SA"):
    return gpu_eigsh(A_gpu, k=k, which=which, tol=tol)


def gpu_eigs_wrap(A_gpu, k: int, tol: float, which: str = "LM"):
    if not HAS_GPU_EIGS or (gpu_eigs is None):
        raise RuntimeError("cupyx.scipy.sparse.linalg.eigs is not available in this CuPy build.")
    return gpu_eigs(A_gpu, k=k, which=which, tol=tol)


# =========================
# One-case runner
# =========================
def run_one_case(
    family: str,
    A_cpu: csr_matrix,
    A_gpu,
    N: int,
    dtype,
    k: int,
    tol: float,
    use_eigs: bool,
    repeats_cpu: int,
    warmup_cpu: int,
    repeats_gpu: int,
    warmup_gpu: int,
    logf,
    check_residual: bool = True,
) -> Dict[str, Any]:
    gc.collect()
    rss0 = rss_bytes()

    # ---- CPU
    if not use_eigs:
        cpu_call = lambda: cpu_eigsh(A_cpu, k=k, tol=tol)
    else:
        cpu_call = lambda: cpu_eigs(A_cpu, k=k, tol=tol)

    w_cpu = None
    V_cpu = None

    def cpu_once_store():
        nonlocal w_cpu, V_cpu
        w_cpu, V_cpu = cpu_call()

    t_cpu = time_cpu(cpu_once_store, warmup=warmup_cpu, repeats=repeats_cpu)

    rss1 = rss_bytes()
    rss_delta = (rss1 - rss0) if (rss0 is not None and rss1 is not None) else None

    res_cpu = np.nan
    if check_residual:
        try:
            res_cpu = residual_norm_cpu(A_cpu, np.asarray(w_cpu), np.asarray(V_cpu))
        except Exception:
            res_cpu = np.nan

    log_print(logf, f"CPU: median={t_cpu:.6f} s, RSS_delta={bytes_to_mib(rss_delta)} MiB, residual={res_cpu:.3e}")

    # ---- GPU
    t_gpu = np.nan
    res_gpu = np.nan
    pool_used = np.nan
    pool_total = np.nan

    gpu_ok = (cp is not None) and (A_gpu is not None)
    if gpu_ok and use_eigs and (not HAS_GPU_EIGS):
        log_print(logf, "GPU: SKIPPED (cupyx.scipy.sparse.linalg.eigs not available)")
        gpu_ok = False

    if gpu_ok:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        pool = cp.get_default_memory_pool()
        used0 = pool.used_bytes()
        total0 = pool.total_bytes()

        if not use_eigs:
            gpu_call = lambda: gpu_eigsh_wrap(A_gpu, k=k, tol=tol)
        else:
            gpu_call = lambda: gpu_eigs_wrap(A_gpu, k=k, tol=tol)

        w_gpu = None
        V_gpu = None

        def gpu_once_store():
            nonlocal w_gpu, V_gpu
            w_gpu, V_gpu = gpu_call()

        try:
            t_gpu = time_gpu(gpu_once_store, warmup=warmup_gpu, repeats=repeats_gpu)
            used1 = pool.used_bytes()
            total1 = pool.total_bytes()
            pool_used = used1 - used0
            pool_total = total1 - total0

            if check_residual:
                try:
                    res_gpu = residual_norm_gpu(A_gpu, w_gpu, V_gpu)
                except Exception:
                    res_gpu = np.nan

            speedup = (t_cpu / t_gpu) if (np.isfinite(t_gpu) and t_gpu > 0) else np.nan
            log_print(
                logf,
                f"GPU: median={t_gpu:.6f} s, pool_used={bytes_to_mib(pool_used)} MiB, "
                f"pool_total={bytes_to_mib(pool_total)} MiB, residual={res_gpu:.3e}, speedup={speedup:.2f}x"
            )
        except Exception as e:
            log_print(logf, f"GPU: FAILED ({type(e).__name__}: {e})")
            t_gpu = np.nan
            res_gpu = np.nan
            pool_used = np.nan
            pool_total = np.nan

    row = dict(
        family=family,
        N=int(N),
        dtype=dtype_name(dtype),
        k=int(k),
        tol=float(tol),
        t_cpu_s=float(t_cpu),
        t_gpu_s=float(t_gpu),
        speedup=float((t_cpu / t_gpu) if (np.isfinite(t_gpu) and t_gpu > 0) else np.nan),
        residual_cpu=float(res_cpu),
        residual_gpu=float(res_gpu),
        rss_delta_bytes=float(rss_delta) if rss_delta is not None else np.nan,
        gpu_pool_used_bytes=float(pool_used),
        gpu_pool_total_bytes=float(pool_total),
    )
    return row


# =========================
# Sweeps (sparse only)
# =========================
def build_sparse_pair(base_name: str, N: int, density: float, dtype, seed: int):
    if "real_symmetric" in base_name:
        A_cpu = make_real_symmetric_sparse(N, density=density, dtype=dtype, seed=seed)
    else:
        A_cpu = make_complex_general_sparse(N, density=density, dtype=dtype, seed=seed)

    A_gpu = None
    if cp is not None:
        try:
            A_gpu = gpu_csr_matrix(A_cpu)
        except Exception:
            A_gpu = None
    return A_cpu, A_gpu


def run_scan_N(base_name: str, dtype, sizes: List[int], density: float, k: int, tol: float,
               repeats_cpu: int, warmup_cpu: int, repeats_gpu: int, warmup_gpu: int,
               seed: int, logf) -> List[Dict[str, Any]]:
    rows = []
    family = f"{base_name}_{dtype_name(dtype)}"
    use_eigs = base_name.endswith("eigs")
    for N in sizes:
        log_print(logf, f"\n-- scan N: N={N}, k={k}, tol={tol}, density={density}, dtype={dtype_name(dtype)} --")
        A_cpu, A_gpu = build_sparse_pair(base_name, N, density, dtype, seed)
        rows.append(run_one_case(family, A_cpu, A_gpu, N, dtype, k, tol,
                                 use_eigs, repeats_cpu, warmup_cpu, repeats_gpu, warmup_gpu, logf, True))
    return rows


def run_scan_k(base_name: str, dtype, N: int, density: float, ks: List[int], tol: float,
               repeats_cpu: int, warmup_cpu: int, repeats_gpu: int, warmup_gpu: int,
               seed: int, logf) -> List[Dict[str, Any]]:
    rows = []
    family = f"{base_name}_{dtype_name(dtype)}"
    use_eigs = base_name.endswith("eigs")
    log_print(logf, f"\n[build matrix once] N={N}, density={density}, dtype={dtype_name(dtype)}")
    A_cpu, A_gpu = build_sparse_pair(base_name, N, density, dtype, seed)

    for k in ks:
        log_print(logf, f"\n-- scan k: N={N}, k={k}, tol={tol}, dtype={dtype_name(dtype)} --")
        rows.append(run_one_case(family, A_cpu, A_gpu, N, dtype, k, tol,
                                 use_eigs, repeats_cpu, warmup_cpu, repeats_gpu, warmup_gpu, logf, True))
    return rows


def run_scan_tol(base_name: str, dtype, N: int, density: float, k: int, tols: List[float],
                 repeats_cpu: int, warmup_cpu: int, repeats_gpu: int, warmup_gpu: int,
                 seed: int, logf) -> List[Dict[str, Any]]:
    rows = []
    family = f"{base_name}_{dtype_name(dtype)}"
    use_eigs = base_name.endswith("eigs")
    log_print(logf, f"\n[build matrix once] N={N}, density={density}, dtype={dtype_name(dtype)}")
    A_cpu, A_gpu = build_sparse_pair(base_name, N, density, dtype, seed)

    for tol in tols:
        log_print(logf, f"\n-- scan tol: N={N}, k={k}, tol={tol}, dtype={dtype_name(dtype)} --")
        rows.append(run_one_case(family, A_cpu, A_gpu, N, dtype, k, tol,
                                 use_eigs, repeats_cpu, warmup_cpu, repeats_gpu, warmup_gpu, logf, True))
    return rows


# =========================
# CSV + plotting (multi-dtype)
# =========================
def save_rows_csv(rows: List[Dict[str, Any]], out_csv: Path):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_cpu_gpu(
    rows: List[Dict[str, Any]],
    xkey: str,
    outdir: Path,
    tag: str,
    title: str,
    xlog: bool = False,
    ylog: bool = True,
):
    """
    rows: one dtype for the same sweep
    Output:
      - one TIME plot with CPU/GPU curves
      - one SPEEDUP plot (only if GPU data exists)
    """
    prb_style()
    if not rows:
        return

    rr = sorted(rows, key=lambda r: float(r[xkey]))
    x = np.array([r[xkey] for r in rr], dtype=float)
    t_cpu = np.array([r["t_cpu_s"] for r in rr], dtype=float)
    t_gpu = np.array([r["t_gpu_s"] for r in rr], dtype=float)
    have_gpu = np.all(np.isfinite(t_gpu))
    dtname = rr[0]["dtype"]

    # ---- TIME
    fig = plt.figure(figsize=(3.45, 2.8))
    ax = fig.add_subplot(111)
    ax.plot(
        x, t_cpu,
        marker="o",
        linestyle="-",
        label=f"CPU ({dtname})"
    )
    if have_gpu:
        ax.plot(
            x, t_gpu,
            marker="s",
            linestyle="-",
            label=f"GPU ({dtname})"
        )

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xkey)
    ax.set_ylabel("Median time (s)")
    ax.set_title(f"{title} | time")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best", ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / f"{tag}_time.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{tag}_time.png", bbox_inches="tight")
    plt.close(fig)

    # ---- SPEEDUP (only plot if GPU data exists)
    fig = plt.figure(figsize=(3.45, 2.4))
    ax = fig.add_subplot(111)

    if have_gpu:
        spd = t_cpu / t_gpu
        ax.plot(
            x, spd,
            marker="o",
            linestyle="-",
            label=f"{dtname}"
        )
        if xlog:
            ax.set_xscale("log")
        ax.set_xlabel(xkey)
        ax.set_ylabel("Speedup (CPU/GPU)")
        ax.set_title(f"{title} | speedup")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.legend(loc="best", ncol=1)
        fig.tight_layout(pad=0.3)
        fig.savefig(outdir / f"{tag}_speedup.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"{tag}_speedup.png", bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    # -------------------------
    # User settings
    # -------------------------
    cpu_threads = 4
    set_cpu_threads(cpu_threads)

    seed = 0

    repeats_cpu = 5
    warmup_cpu = 1
    repeats_gpu = 8
    warmup_gpu = 2

    density = 0.01
    k_fixed = 16
    tol_fixed = 1e-6

    sizes_sym = [256, 384, 512, 768, 1024, 1536, 2048]
    sizes_gen = [128, 192, 256, 384, 512, 768, 1024]

    N_k_scan_sym = 1024
    ks_sym = [2, 4, 8, 16, 32, 48, 64]
    N_k_scan_gen = 512
    ks_gen = [2, 4, 8, 16, 24, 32]

    N_tol_scan_sym = 1024
    tols_sym = [1e-2, 1e-4, 1e-6, 1e-8]
    N_tol_scan_gen = 512
    tols_gen = [1e-2, 1e-4, 1e-6]

    # Dtypes
    sym_dtypes = [np.float32, np.float64]
    gen_dtypes = [np.complex64, np.complex128]

    # -------------------------
    # Output naming convention
    # -------------------------
    tag = now_tag()
    host = platform.node() or "host"
    outdir = ensure_dir(Path("results") / f"krylov_cpu_gpu_{host}_{tag}")
    log_path = outdir / f"krylov_cpu_gpu_{host}_{tag}.log"
    csv_path = outdir / f"krylov_cpu_gpu_{host}_{tag}.csv"

    logf = open_log(log_path)
    log_print(logf, f"[RUN] {tag}")
    log_print(logf, f"Output dir: {outdir.resolve()}")
    log_print(logf, f"Python: {sys.version.splitlines()[0]}")
    log_print(logf, f"Platform: {platform.platform()}")
    log_print(logf, f"CPU threads setting: {cpu_threads} (None = default/max)")
    log_print(logf, "\n[NumPy/SciPy]\n" + get_numpy_scipy_info())
    log_print(logf, "\n[GPU]\n" + get_gpu_info())

    rows_all: List[Dict[str, Any]] = []

    # =========================
    # Family 1: real symmetric (eigsh)
    # =========================
    base_sym = "real_symmetric_eigsh"
    log_print(logf, "\n========== FAMILY: real symmetric (eigsh) ==========")

    # ---- (A) scan N (per-dtype plots)
    log_print(logf, f"\n[A] scan N (k={k_fixed}, tol={tol_fixed}, density={density})  [per-dtype plots]")
    for dtp in sym_dtypes:
        rows = run_scan_N(base_sym, dtp, sizes_sym, density, k_fixed, tol_fixed,
                          repeats_cpu, warmup_cpu, repeats_gpu, warmup_gpu, seed, logf)
        rows_all += rows
        plot_cpu_gpu(
            rows, xkey="N", outdir=outdir, tag=f"{tag}_{base_sym}_scanN_{dtype_name(dtp)}",
            title=f"{base_sym} | dtype={dtype_name(dtp)} | CSR density={density} | k={k_fixed}, tol={tol_fixed}",
            xlog=True, ylog=True
        )

    # ---- (B) scan k
    log_print(logf, f"\n[B] scan k (N={N_k_scan_sym}, tol={tol_fixed}, density={density})  [per-dtype plots]")
    for dtp in sym_dtypes:
        rows = run_scan_k(base_sym, dtp, N_k_scan_sym, density, ks_sym, tol_fixed,
                          repeats_cpu, warmup_cpu, repeats_gpu, warmup_gpu, seed, logf)
        rows_all += rows
        plot_cpu_gpu(
            rows, xkey="k", outdir=outdir, tag=f"{tag}_{base_sym}_scank_{dtype_name(dtp)}",
            title=f"{base_sym} | dtype={dtype_name(dtp)} | CSR density={density} | N={N_k_scan_sym}, tol={tol_fixed}",
            xlog=False, ylog=True
        )

    # ---- (C) scan tol
    log_print(logf, f"\n[C] scan tol (N={N_tol_scan_sym}, k={k_fixed}, density={density})  [per-dtype plots]")
    for dtp in sym_dtypes:
        rows = run_scan_tol(base_sym, dtp, N_tol_scan_sym, density, k_fixed, tols_sym,
                            repeats_cpu, warmup_cpu, repeats_gpu, warmup_gpu, seed, logf)
        rows_all += rows
        plot_cpu_gpu(
            rows, xkey="tol", outdir=outdir, tag=f"{tag}_{base_sym}_scantol_{dtype_name(dtp)}",
            title=f"{base_sym} | dtype={dtype_name(dtp)} | CSR density={density} | N={N_tol_scan_sym}, k={k_fixed}",
            xlog=True, ylog=True
        )

    # =========================
    # Family 2: general complex (eigs)
    # =========================
    base_gen = "complex_general_eigs"
    log_print(logf, "\n========== FAMILY: general complex (eigs) ==========")
    log_print(logf, "NOTE: CuPyX eigs support may be unavailable depending on CuPy build/version.")

    # ---- (A) scan N
    log_print(logf, f"\n[A] scan N (k={k_fixed}, tol={tol_fixed}, density={density})  [per-dtype plots]")
    for dtp in gen_dtypes:
        rows = run_scan_N(base_gen, dtp, sizes_gen, density, k_fixed, tol_fixed,
                          repeats_cpu, warmup_cpu, repeats_gpu, warmup_gpu, seed, logf)
        rows_all += rows
        plot_cpu_gpu(
            rows, xkey="N", outdir=outdir, tag=f"{tag}_{base_gen}_scanN_{dtype_name(dtp)}",
            title=f"{base_gen} | dtype={dtype_name(dtp)} | CSR density={density} | k={k_fixed}, tol={tol_fixed}",
            xlog=True, ylog=True
        )

    # ---- (B) scan k
    log_print(logf, f"\n[B] scan k (N={N_k_scan_gen}, tol={tol_fixed}, density={density})  [per-dtype plots]")
    for dtp in gen_dtypes:
        rows = run_scan_k(base_gen, dtp, N_k_scan_gen, density, ks_gen, tol_fixed,
                          repeats_cpu, warmup_cpu, repeats_gpu, warmup_gpu, seed, logf)
        rows_all += rows
        plot_cpu_gpu(
            rows, xkey="k", outdir=outdir, tag=f"{tag}_{base_gen}_scank_{dtype_name(dtp)}",
            title=f"{base_gen} | dtype={dtype_name(dtp)} | CSR density={density} | N={N_k_scan_gen}, tol={tol_fixed}",
            xlog=False, ylog=True
        )

    # ---- (C) scan tol
    log_print(logf, f"\n[C] scan tol (N={N_tol_scan_gen}, k={k_fixed}, density={density})  [per-dtype plots]")
    for dtp in gen_dtypes:
        rows = run_scan_tol(base_gen, dtp, N_tol_scan_gen, density, k_fixed, tols_gen,
                            repeats_cpu, warmup_cpu, repeats_gpu, warmup_gpu, seed, logf)
        rows_all += rows
        plot_cpu_gpu(
            rows, xkey="tol", outdir=outdir, tag=f"{tag}_{base_gen}_scantol_{dtype_name(dtp)}",
            title=f"{base_gen} | dtype={dtype_name(dtp)} | CSR density={density} | N={N_tol_scan_gen}, k={k_fixed}",
            xlog=True, ylog=True
        )

    # =========================
    # Save CSV & close
    # =========================
    save_rows_csv(rows_all, csv_path)
    log_print(logf, f"\nSaved CSV: {csv_path.name}")
    log_print(logf, "Saved figures: *_time.pdf/png and *_speedup.pdf/png under output directory.")
    logf.close()

    print(f"\nDONE.\nLog: {log_path}\nCSV: {csv_path}\nFigures in: {outdir}")


if __name__ == "__main__":
    main()
