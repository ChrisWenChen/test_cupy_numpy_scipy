#!/usr/bin/env python3
"""
Full diagonalization benchmarks: CPU (SciPy) vs GPU (CuPy/cuSOLVER via CuPy/cupyx).

Covers:
  1) Real symmetric: eigh
  2) Complex Hermitian: eigh
  3) Complex general (non-Hermitian): eig  (GPU uses cupyx.scipy.linalg.eig)

Outputs:
  - .log and .csv with system + benchmark records
  - PRB-friendly figures in PDF/PNG (time, throughput, memory, speedup)

Notes:
  - For RTX 4060, FP64 performance is much weaker than FP32 -> smaller speedups are expected.
  - General eig on GPU is routed through cupyx.scipy.linalg (not cp.linalg).
"""

import os
import sys
import time
import gc
import platform
import datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt

# --- CPU linear algebra
try:
    import scipy
    import scipy.linalg as spla
except Exception as e:
    raise RuntimeError("SciPy is required for CPU benchmarks (scipy.linalg).") from e

# --- GPU linear algebra
try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.linalg as cxspla
except Exception:
    cp = None
    cxspla = None

# --- Optional memory
try:
    import psutil
except ImportError:
    psutil = None


# ============================================================
# Utilities: paths / logging / system info
# ============================================================
def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def open_log(path: Path):
    return open(path, "w", encoding="utf-8")


def log_print(f, s: str):
    print(s)
    f.write(s + "\n")
    f.flush()


def rss_bytes() -> Optional[int]:
    """Process resident set size (RSS) in bytes, CPU RAM."""
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss


def bytes_to_mib(x: Optional[float]) -> Optional[float]:
    return None if x is None else x / (1024**2)


def dtype_name(dtype) -> str:
    return np.dtype(dtype).name


def get_numpy_scipy_info() -> str:
    out = []
    out.append(f"NumPy version: {np.__version__}")
    out.append(f"SciPy version: {scipy.__version__}")
    try:
        from io import StringIO
        import contextlib
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        out.append("\n[NumPy show_config]\n" + buf.getvalue().strip())
    except Exception as e:
        out.append(f"(np.show_config unavailable: {e})")
    return "\n".join(out)


def get_gpu_info() -> str:
    if cp is None:
        return "CuPy not installed"
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], (bytes, bytearray)) else props["name"]
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        return (f"CuPy version: {cp.__version__}\n"
                f"GPU: {name}, device_id={dev.id}\n"
                f"VRAM total: {total_mem/(1024**3):.2f} GiB, free: {free_mem/(1024**3):.2f} GiB\n"
                f"CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
    except Exception as e:
        return f"(GPU info unavailable: {e})"


# ============================================================
# Thread control (CPU BLAS)
# ============================================================
def set_cpu_threads(n_threads: Optional[int]):
    """
    Best-effort thread control for BLAS backends.
    Set before heavy work for best effect.
    """
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


# ============================================================
# Matrix generators
# ============================================================
def make_real_symmetric(N: int, dtype=np.float32, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, N), dtype=np.float64).astype(dtype, copy=False)
    A = (X + X.T) * 0.5
    return A


def make_complex_hermitian(N: int, dtype=np.complex64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xr = rng.standard_normal((N, N), dtype=np.float64)
    Xi = rng.standard_normal((N, N), dtype=np.float64)
    X = (Xr + 1j * Xi).astype(dtype, copy=False)
    A = (X + X.conj().T) * 0.5
    return A


def make_complex_general(N: int, dtype=np.complex64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xr = rng.standard_normal((N, N), dtype=np.float64)
    Xi = rng.standard_normal((N, N), dtype=np.float64)
    A = (Xr + 1j * Xi).astype(dtype, copy=False)
    return A


# ============================================================
# Residual checks
# ============================================================
def residual_eigh(A, w, V) -> float:
    """Relative Frobenius residual: ||A V - V diag(w)||_F / ||A||_F."""
    AV = A @ V
    VD = V * w  # broadcast columns
    num = np.linalg.norm(AV - VD, ord="fro")
    den = np.linalg.norm(A, ord="fro")
    return float(num / den)


def residual_eig(A, w, V) -> float:
    """Relative Frobenius residual: ||A V - V diag(w)||_F / ||A||_F."""
    AV = A @ V
    VD = V * w
    num = np.linalg.norm(AV - VD, ord="fro")
    den = np.linalg.norm(A, ord="fro")
    return float(num / den)


# ============================================================
# Throughput model (rough)
# ============================================================
def gflops_eigh(N: int, t_sec: float) -> float:
    """
    Very rough flop model for dense symmetric/Hermitian eigen-decomposition.
    Common rule-of-thumb ~ O( (4/3) N^3 ) to (10/3) N^3 depending on algorithm.
    We'll report a conservative proxy: 4/3 N^3.
    """
    flops = (4.0 / 3.0) * (N**3)
    return flops / t_sec / 1e9


def gflops_eig(N: int, t_sec: float) -> float:
    """
    Rough flop proxy for dense general eigen-decomposition ~ O( (20/3) N^3 ) typical.
    We'll use 20/3 N^3 as a proxy.
    """
    flops = (20.0 / 3.0) * (N**3)
    return flops / t_sec / 1e9


# ============================================================
# CPU / GPU benchmark kernels
# ============================================================
def cpu_eigh_bench(A: np.ndarray, repeats=10, warmup=2, check_residual=True) -> Tuple[float, float]:
    """
    CPU symmetric/Hermitian eigen: SciPy eigh.
    Returns: median_time_sec, residual
    """
    # warmup
    for _ in range(warmup):
        w, V = spla.eigh(A, overwrite_a=False, check_finite=False)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        w, V = spla.eigh(A, overwrite_a=False, check_finite=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    med = float(np.median(times))

    if check_residual:
        res = residual_eigh(A, w, V)
    else:
        res = np.nan
    return med, float(res)


def cpu_eig_bench(A: np.ndarray, repeats=10, warmup=2, check_residual=True) -> Tuple[float, float]:
    """
    CPU general eigen: SciPy eig.
    Returns: median_time_sec, residual
    """
    for _ in range(warmup):
        w, V = spla.eig(A, overwrite_a=False, check_finite=False)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        w, V = spla.eig(A, overwrite_a=False, check_finite=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    med = float(np.median(times))

    if check_residual:
        res = residual_eig(A, w, V)
    else:
        res = np.nan
    return med, float(res)


def gpu_eigh_bench(A_gpu, repeats=30, warmup=5, check_residual=True) -> Tuple[float, int, int, float]:
    """
    GPU symmetric/Hermitian eigen: CuPy eigh (cuSOLVER).
    Returns: median_time_sec, pool_used_delta_bytes, pool_total_delta_bytes, residual
    """
    if cp is None:
        raise RuntimeError("CuPy not available.")

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    pool = cp.get_default_memory_pool()
    used0 = pool.used_bytes()
    total0 = pool.total_bytes()

    # warmup
    for _ in range(warmup):
        w, V = cp.linalg.eigh(A_gpu)
    cp.cuda.Stream.null.synchronize()

    # timing with CUDA events
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    times_ms = []
    for _ in range(repeats):
        start.record()
        w, V = cp.linalg.eigh(A_gpu)
        end.record()
        end.synchronize()
        times_ms.append(cp.cuda.get_elapsed_time(start, end))

    med = float(np.median(times_ms) / 1000.0)

    used1 = pool.used_bytes()
    total1 = pool.total_bytes()

    if check_residual:
        # compute residual on GPU then bring scalar back
        AV = A_gpu @ V
        VD = V * w
        num = cp.linalg.norm(AV - VD)
        den = cp.linalg.norm(A_gpu)
        res = float((num / den).get())
    else:
        res = np.nan

    return med, int(used1 - used0), int(total1 - total0), float(res)


def gpu_eig_bench(A_gpu, repeats=30, warmup=5, check_residual=True) -> Tuple[float, int, int, float]:
    """
    GPU general eigen: cupyx.scipy.linalg.eig (NOT cp.linalg.eig).
    Returns: median_time_sec, pool_used_delta_bytes, pool_total_delta_bytes, residual
    """
    if cp is None or cxspla is None:
        raise RuntimeError("CuPy/cupyx not available for GPU general eig.")

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    pool = cp.get_default_memory_pool()
    used0 = pool.used_bytes()
    total0 = pool.total_bytes()

    for _ in range(warmup):
        w, V = cxspla.eig(A_gpu)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    times_ms = []
    for _ in range(repeats):
        start.record()
        w, V = cxspla.eig(A_gpu)
        end.record()
        end.synchronize()
        times_ms.append(cp.cuda.get_elapsed_time(start, end))

    med = float(np.median(times_ms) / 1000.0)

    used1 = pool.used_bytes()
    total1 = pool.total_bytes()

    if check_residual:
        AV = A_gpu @ V
        VD = V * w
        num = cp.linalg.norm(AV - VD)
        den = cp.linalg.norm(A_gpu)
        res = float((num / den).get())
    else:
        res = np.nan

    return med, int(used1 - used0), int(total1 - total0), float(res)


# ============================================================
# Plotting (PRB-friendly)
# ============================================================
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


def plot_family(rows: List[Dict], outdir: Path, tag: str, title: str):
    prb_style()

    Ns = np.array([r["N"] for r in rows], dtype=int)
    t_cpu = np.array([r["t_cpu_s"] for r in rows], dtype=float)
    g_cpu = np.array([r["gflops_cpu"] for r in rows], dtype=float)

    have_gpu = all(r["t_gpu_s"] is not None for r in rows)
    if have_gpu:
        t_gpu = np.array([r["t_gpu_s"] for r in rows], dtype=float)
        g_gpu = np.array([r["gflops_gpu"] for r in rows], dtype=float)
        spd = np.array([r["speedup"] for r in rows], dtype=float)

    # --- time
    fig = plt.figure(figsize=(3.45, 2.6))
    ax = fig.add_subplot(111)
    ax.plot(Ns, t_cpu, marker="o", label="SciPy (CPU)")
    if have_gpu:
        ax.plot(Ns, t_gpu, marker="s", label="CuPy (GPU)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"Matrix size $N$")
    ax.set_ylabel(r"Median time (s)")
    ax.set_title(title + " | Time")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / f"{tag}_time.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{tag}_time.png", bbox_inches="tight")
    plt.close(fig)

    # --- throughput
    fig = plt.figure(figsize=(3.45, 2.6))
    ax = fig.add_subplot(111)
    ax.plot(Ns, g_cpu, marker="o", label="SciPy (CPU)")
    if have_gpu:
        ax.plot(Ns, g_gpu, marker="s", label="CuPy (GPU)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Matrix size $N$")
    ax.set_ylabel(r"Proxy throughput (GFLOP/s)")
    ax.set_title(title + " | Throughput (proxy)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / f"{tag}_gflops.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{tag}_gflops.png", bbox_inches="tight")
    plt.close(fig)

    # --- memory
    fig = plt.figure(figsize=(3.45, 2.6))
    ax = fig.add_subplot(111)
    rss = np.array([np.nan if r["rss_delta_bytes"] is None else r["rss_delta_bytes"] for r in rows], dtype=float)
    ax.plot(Ns, rss/(1024**2), marker="o", label="CPU RSS delta (MiB)")

    theo = np.array([r["theo_bytes"] for r in rows], dtype=float)
    ax.plot(Ns, theo/(1024**2), marker="^", label="Theoretical A + outputs (MiB)")

    if have_gpu:
        pool_total = np.array([r["gpu_pool_total_delta_bytes"] for r in rows], dtype=float)
        ax.plot(Ns, pool_total/(1024**2), marker="s", label="GPU pool total delta (MiB)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"Matrix size $N$")
    ax.set_ylabel(r"Memory (MiB)")
    ax.set_title(title + " | Memory")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / f"{tag}_memory.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{tag}_memory.png", bbox_inches="tight")
    plt.close(fig)

    # --- speedup
    if have_gpu:
        fig = plt.figure(figsize=(3.45, 2.3))
        ax = fig.add_subplot(111)
        ax.plot(Ns, spd, marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"Matrix size $N$")
        ax.set_ylabel(r"Speedup (CPU / GPU)")
        ax.set_title(title + " | Speedup")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        fig.tight_layout(pad=0.3)
        fig.savefig(outdir / f"{tag}_speedup.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"{tag}_speedup.png", bbox_inches="tight")
        plt.close(fig)


# ============================================================
# Benchmark runner
# ============================================================
def save_rows_csv(rows: List[Dict], path: Path):
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_family(
    family_name: str,
    sizes: List[int],
    dtype,
    generator_fn,
    cpu_solver: str,         # "eigh" or "eig"
    gpu_solver: Optional[str], # "eigh" or "eig"
    repeats_cpu: int,
    repeats_gpu: int,
    warmup_cpu: int,
    warmup_gpu: int,
    seed: int,
    check_residual: bool,
    logf
) -> List[Dict]:

    rows = []
    log_print(logf, f"\n========== {family_name} ==========")
    log_print(logf, f"dtype={dtype}, sizes={sizes}")
    for N in sizes:
        log_print(logf, f"\n-- N={N} --")

        gc.collect()
        rss0 = rss_bytes()

        # generate matrix on CPU
        A = generator_fn(N, dtype=dtype, seed=seed)

        # CPU solve
        if cpu_solver == "eigh":
            t_cpu, res_cpu = cpu_eigh_bench(A, repeats=repeats_cpu, warmup=warmup_cpu, check_residual=check_residual)
            g_cpu = gflops_eigh(N, t_cpu)
        else:
            t_cpu, res_cpu = cpu_eig_bench(A, repeats=repeats_cpu, warmup=warmup_cpu, check_residual=check_residual)
            g_cpu = gflops_eig(N, t_cpu)

        rss1 = rss_bytes()
        rss_delta = (rss1 - rss0) if (rss0 is not None and rss1 is not None) else None

        log_print(logf, f"CPU: median={t_cpu:.6f} s, RSS_delta={bytes_to_mib(rss_delta)} MiB, residual={res_cpu:.3e}")

        # theoretical storage proxy: input A + outputs (w,V)
        # - eigh/eig returns eigenvectors (N^2) + eigenvalues (N)
        item = np.dtype(dtype).itemsize
        theo = (N*N)*item + (N*N)*item + (N)*item  # A + V + w (very rough)

        # GPU solve
        t_gpu = used_d = total_d = res_gpu = None
        g_gpu = speedup = None
        if (cp is not None) and (gpu_solver is not None):
            try:
                # move A to GPU
                A_gpu = cp.asarray(A)

                if gpu_solver == "eigh":
                    t_gpu, used_d, total_d, res_gpu = gpu_eigh_bench(
                        A_gpu, repeats=repeats_gpu, warmup=warmup_gpu, check_residual=check_residual
                    )
                    g_gpu = gflops_eigh(N, t_gpu)
                else:
                    t_gpu, used_d, total_d, res_gpu = gpu_eig_bench(
                        A_gpu, repeats=repeats_gpu, warmup=warmup_gpu, check_residual=check_residual
                    )
                    g_gpu = gflops_eig(N, t_gpu)

                speedup = t_cpu / t_gpu
                log_print(
                    logf,
                    f"GPU: median={t_gpu:.6f} s, pool_used={used_d/(1024**2):.2f} MiB, "
                    f"pool_total={total_d/(1024**2):.2f} MiB, residual={res_gpu:.3e}, speedup={speedup:.2f}x"
                )
            except Exception as e:
                log_print(logf, f"[GPU ERROR] {e}")
                t_gpu = used_d = total_d = res_gpu = None

        rows.append(dict(
            family=family_name,
            N=N,
            dtype=str(dtype),
            cpu_solver=cpu_solver,
            gpu_solver=gpu_solver if gpu_solver is not None else "",
            t_cpu_s=t_cpu,
            t_gpu_s=t_gpu,
            gflops_cpu=g_cpu,
            gflops_gpu=g_gpu,
            speedup=speedup,
            residual_cpu=res_cpu,
            residual_gpu=res_gpu,
            rss_delta_bytes=rss_delta,
            gpu_pool_used_delta_bytes=used_d,
            gpu_pool_total_delta_bytes=total_d,
            theo_bytes=theo,
        ))

    return rows


def main():
    # -------------------------
    # User settings
    # -------------------------
    cpu_threads = 4          # set None for "max/default"
    seed = 0
    check_residual = True

    # repeats/warmup
    repeats_cpu = 8
    warmup_cpu = 2
    repeats_gpu = 20
    warmup_gpu = 5

    # sizes (adjust for your 8GB VRAM)
    sizes_eigh = [256, 384, 512, 768, 1024, 1536, 2048]
    sizes_eig  = [128, 192, 256, 384, 512, 768, 1024]  # general eig is heavier

    # -------------------------
    # Output naming
    # -------------------------
    tag = now_tag()
    host = platform.node() or "host"
    outdir = ensure_dir(Path("results") / f"eigensolve_{host}_{tag}")
    log_path = outdir / f"eigensolve_{host}_{tag}.log"
    csv_path = outdir / f"eigensolve_{host}_{tag}.csv"

    # threads
    set_cpu_threads(cpu_threads)

    # -------------------------
    # Logging header
    # -------------------------
    logf = open_log(log_path)
    log_print(logf, f"[RUN] {tag}")
    log_print(logf, f"Output dir: {outdir.resolve()}")
    log_print(logf, f"Python: {sys.version.splitlines()[0]}")
    log_print(logf, f"Platform: {platform.platform()}")
    log_print(logf, f"CPU threads setting: {cpu_threads} (None = default/max)")
    log_print(logf, "\n" + get_numpy_scipy_info())
    log_print(logf, "\n[GPU]\n" + get_gpu_info())

    all_rows: List[Dict] = []

    # -------------------------
    # 1) Real symmetric (float32/float64) : eigh
    # -------------------------
    all_rows += run_family(
        family_name="real_symmetric_eigh_float32",
        sizes=sizes_eigh,
        dtype=np.float32,
        generator_fn=make_real_symmetric,
        cpu_solver="eigh",
        gpu_solver="eigh" if cp is not None else None,
        repeats_cpu=repeats_cpu, repeats_gpu=repeats_gpu,
        warmup_cpu=warmup_cpu, warmup_gpu=warmup_gpu,
        seed=seed, check_residual=check_residual, logf=logf
    )

    all_rows += run_family(
        family_name="real_symmetric_eigh_float64",
        sizes=sizes_eigh[:-1],  # a bit smaller for FP64
        dtype=np.float64,
        generator_fn=make_real_symmetric,
        cpu_solver="eigh",
        gpu_solver="eigh" if cp is not None else None,
        repeats_cpu=repeats_cpu, repeats_gpu=repeats_gpu,
        warmup_cpu=warmup_cpu, warmup_gpu=warmup_gpu,
        seed=seed, check_residual=check_residual, logf=logf
    )

    # -------------------------
    # 2) Complex Hermitian (complex64/complex128) : eigh
    # -------------------------
    all_rows += run_family(
        family_name="complex_hermitian_eigh_complex64",
        sizes=sizes_eigh,
        dtype=np.complex64,
        generator_fn=make_complex_hermitian,
        cpu_solver="eigh",
        gpu_solver="eigh" if cp is not None else None,
        repeats_cpu=repeats_cpu, repeats_gpu=repeats_gpu,
        warmup_cpu=warmup_cpu, warmup_gpu=warmup_gpu,
        seed=seed, check_residual=check_residual, logf=logf
    )

    all_rows += run_family(
        family_name="complex_hermitian_eigh_complex128",
        sizes=sizes_eigh[:-1],
        dtype=np.complex128,
        generator_fn=make_complex_hermitian,
        cpu_solver="eigh",
        gpu_solver="eigh" if cp is not None else None,
        repeats_cpu=repeats_cpu, repeats_gpu=repeats_gpu,
        warmup_cpu=warmup_cpu, warmup_gpu=warmup_gpu,
        seed=seed, check_residual=check_residual, logf=logf
    )

    # -------------------------
    # 3) Complex general (non-Hermitian): eig
    # GPU uses cupyx.scipy.linalg.eig
    # -------------------------
    gpu_has_general_eig = (cp is not None) and (cxspla is not None) and hasattr(cxspla, "eig")

    all_rows += run_family(
        family_name="complex_general_eig_complex64",
        sizes=sizes_eig,
        dtype=np.complex64,
        generator_fn=make_complex_general,
        cpu_solver="eig",
        gpu_solver="eig" if gpu_has_general_eig else None,
        repeats_cpu=repeats_cpu, repeats_gpu=repeats_gpu,
        warmup_cpu=warmup_cpu, warmup_gpu=warmup_gpu,
        seed=seed, check_residual=check_residual, logf=logf
    )

    all_rows += run_family(
        family_name="complex_general_eig_complex128",
        sizes=sizes_eig[:-2],
        dtype=np.complex128,
        generator_fn=make_complex_general,
        cpu_solver="eig",
        gpu_solver="eig" if gpu_has_general_eig else None,
        repeats_cpu=repeats_cpu, repeats_gpu=repeats_gpu,
        warmup_cpu=warmup_cpu, warmup_gpu=warmup_gpu,
        seed=seed, check_residual=check_residual, logf=logf
    )

    # -------------------------
    # Save CSV
    # -------------------------
    save_rows_csv(all_rows, csv_path)
    log_print(logf, f"\nSaved CSV: {csv_path.name}")

    # -------------------------
    # Plot each family separately
    # -------------------------
    # group by family
    families = {}
    for r in all_rows:
        families.setdefault(r["family"], []).append(r)

    for fam, rows in families.items():
        # stable ordering
        rows_sorted = sorted(rows, key=lambda x: x["N"])
        # short tag
        fig_tag = f"{tag}_{fam}"
        plot_family(
            rows_sorted, outdir=outdir,
            tag=fig_tag,
            title=f"{fam} (CPU threads={cpu_threads if cpu_threads is not None else 'default'})"
        )

    log_print(logf, "Saved figures: *_time.pdf/png, *_gflops.pdf/png, *_memory.pdf/png, *_speedup.pdf/png (if GPU).")
    logf.close()

    print(f"\nDONE.\nLog: {log_path}\nCSV: {csv_path}\nFigures in: {outdir}")


if __name__ == "__main__":
    main()
