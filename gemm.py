import os
import sys
import time
import gc
import platform
import datetime as dt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import psutil
except ImportError:
    psutil = None


# -------------------------
# Utils: logging / system
# -------------------------
def now_tag():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def rss_bytes():
    """Process resident memory (CPU RAM)."""
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss


def bytes_to_mib(x):
    return None if x is None else x / (1024**2)


def dtype_name(dtype):
    return np.dtype(dtype).name


def gflops_gemm(N, t_sec):
    # GEMM FLOP count ~ 2*N^3
    return (2.0 * (N**3)) / t_sec / 1e9


def get_numpy_blas_info():
    try:
        import numpy as _np
        # numpy.show_config() prints to stdout; capture and return as string
        from io import StringIO
        import contextlib
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            _np.show_config()
        return buf.getvalue().strip()
    except Exception as e:
        return f"(numpy.show_config() unavailable: {e})"


def get_gpu_info():
    if cp is None:
        return "CuPy not installed"
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], (bytes, bytearray)) else props["name"]
        total_mem = cp.cuda.runtime.memGetInfo()[1]
        return f"GPU: {name}, device_id={dev.id}, total_mem={total_mem/(1024**3):.2f} GiB"
    except Exception as e:
        return f"(GPU info unavailable: {e})"


def open_log(log_path: Path):
    f = open(log_path, "w", encoding="utf-8")
    return f


def log_print(f, s):
    print(s)
    f.write(s + "\n")
    f.flush()


# -------------------------
# Thread control (CPU)
# -------------------------
def set_cpu_threads(n_threads: int | None):
    """
    Best-effort thread control for BLAS backends.
    If n_threads is None: do nothing (use system default).
    """
    if n_threads is None:
        return

    # Ideally set before NumPy imports/initializes BLAS; still best-effort here.
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

    # If threadpoolctl exists, also enforce at runtime.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n_threads)
    except Exception:
        pass


# -------------------------
# Bench kernels
# -------------------------
def numpy_gemm_bench(N, dtype=np.float32, repeats=10, warmup=3, seed=0):
    gc.collect()
    rss0 = rss_bytes()

    rng = np.random.default_rng(seed)
    # Generate in float64 then cast to requested dtype to keep distribution consistent
    A = rng.standard_normal((N, N), dtype=np.float64).astype(dtype, copy=False)
    B = rng.standard_normal((N, N), dtype=np.float64).astype(dtype, copy=False)

    # warm-up
    for _ in range(warmup):
        _ = A @ B

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = A @ B
        t1 = time.perf_counter()
        times.append(t1 - t0)

    rss1 = rss_bytes()
    mem_rss = (rss1 - rss0) if (rss0 is not None and rss1 is not None) else None

    theo = 3 * (N * N * np.dtype(dtype).itemsize)  # A,B,C
    return float(np.median(times)), mem_rss, theo


def cupy_gemm_bench(N, dtype=np.float32, repeats=30, warmup=10, seed=0):
    if cp is None:
        raise RuntimeError("CuPy not installed")

    # Reset pools for cleaner VRAM accounting
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    pool = cp.get_default_memory_pool()
    used0 = pool.used_bytes()
    total0 = pool.total_bytes()

    cp.random.seed(seed)

    A = cp.random.standard_normal((N, N), dtype=dtype)
    B = cp.random.standard_normal((N, N), dtype=dtype)

    # warm-up
    for _ in range(warmup):
        _ = A @ B
    cp.cuda.Stream.null.synchronize()

    # Timing using CUDA events
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    times_ms = []
    for _ in range(repeats):
        start.record()
        _ = A @ B
        end.record()
        end.synchronize()
        times_ms.append(cp.cuda.get_elapsed_time(start, end))

    used1 = pool.used_bytes()
    total1 = pool.total_bytes()

    theo = 3 * (N * N * np.dtype(dtype).itemsize)

    mem_used = used1 - used0
    mem_total = total1 - total0
    return float(np.median(times_ms) / 1000.0), mem_used, mem_total, theo


# -------------------------
# Running & plotting
# -------------------------
def run_suite(sizes, dtype=np.float32, repeats_np=10, repeats_cp=30,
              warmup_np=3, warmup_cp=10, seed=0, logf=None):
    rows = []
    for N in sizes:
        log_print(logf, f"\n== GEMM N={N}, dtype={dtype_name(dtype)} ==")

        t_np, mem_np, theo = numpy_gemm_bench(
            N, dtype=dtype, repeats=repeats_np, warmup=warmup_np, seed=seed
        )
        gf_np = gflops_gemm(N, t_np)
        log_print(
            logf,
            f"NumPy:  median={t_np:.6f} s,  GFLOP/s={gf_np:.2f},  "
            f"RSS_delta={bytes_to_mib(mem_np)} MiB,  theo={theo/(1024**2):.2f} MiB"
        )

        if cp is not None:
            t_cp, mem_used, mem_total, theo_cp = cupy_gemm_bench(
                N, dtype=dtype, repeats=repeats_cp, warmup=warmup_cp, seed=seed
            )
            gf_cp = gflops_gemm(N, t_cp)
            speedup = t_np / t_cp
            log_print(
                logf,
                f"CuPy:   median={t_cp:.6f} s,  GFLOP/s={gf_cp:.2f},  speedup={speedup:.2f}x,  "
                f"pool_used={mem_used/(1024**2):.2f} MiB,  pool_total={mem_total/(1024**2):.2f} MiB,  "
                f"theo={theo_cp/(1024**2):.2f} MiB"
            )
        else:
            t_cp = mem_used = mem_total = None
            gf_cp = speedup = None

        rows.append(
            dict(
                N=N,
                dtype=dtype_name(dtype),
                t_numpy_s=t_np,
                t_cupy_s=t_cp,
                gflops_numpy=gf_np,
                gflops_cupy=gf_cp,
                speedup=speedup if cp is not None else None,
                rss_delta_bytes=mem_np,
                cupy_pool_used_delta_bytes=mem_used,
                cupy_pool_total_delta_bytes=mem_total,
                theo_bytes=theo,
            )
        )
    return rows


def prb_style():
    """
    A conservative, PRB-friendly matplotlib style:
    - vector PDF export
    - embedded TrueType fonts
    - clean ticks/labels
    """
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
        "pdf.fonttype": 42,   # embed TrueType (good for journals)
        "ps.fonttype": 42,
        "text.usetex": False, # set True if you have LaTeX; keep False for portability
    })


def plot_and_save(rows, outdir: Path, tag: str, title_prefix: str):
    prb_style()

    Ns = np.array([r["N"] for r in rows], dtype=int)
    tnp = np.array([r["t_numpy_s"] for r in rows], dtype=float)
    gfnp = np.array([r["gflops_numpy"] for r in rows], dtype=float)

    have_gpu = all(r["t_cupy_s"] is not None for r in rows)
    if have_gpu:
        tcp = np.array([r["t_cupy_s"] for r in rows], dtype=float)
        gfcp = np.array([r["gflops_cupy"] for r in rows], dtype=float)
        spd = np.array([r["speedup"] for r in rows], dtype=float)

    # --- Figure 1: time (log-log)
    fig1 = plt.figure(figsize=(3.45, 2.6))  # ~ single-column width
    ax1 = fig1.add_subplot(111)
    ax1.plot(Ns, tnp, marker="o", label="NumPy (CPU)")
    if have_gpu:
        ax1.plot(Ns, tcp, marker="s", label="CuPy (GPU)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Matrix size $N$ (for $N\times N$)")
    ax1.set_ylabel(r"Median time (s)")
    ax1.set_title(f"{title_prefix}  |  Time")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax1.legend(loc="best")
    fig1.tight_layout(pad=0.3)

    fig1_pdf = outdir / f"{tag}_time.pdf"
    fig1_png = outdir / f"{tag}_time.png"
    fig1.savefig(fig1_pdf, bbox_inches="tight")
    fig1.savefig(fig1_png, bbox_inches="tight")
    plt.close(fig1)

    # --- Figure 2: GFLOP/s (semi-log x)
    fig2 = plt.figure(figsize=(3.45, 2.6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(Ns, gfnp, marker="o", label="NumPy (CPU)")
    if have_gpu:
        ax2.plot(Ns, gfcp, marker="s", label="CuPy (GPU)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel(r"Matrix size $N$")
    ax2.set_ylabel(r"Throughput (GFLOP/s)")
    ax2.set_title(f"{title_prefix}  |  Throughput")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax2.legend(loc="best")
    fig2.tight_layout(pad=0.3)

    fig2_pdf = outdir / f"{tag}_gflops.pdf"
    fig2_png = outdir / f"{tag}_gflops.png"
    fig2.savefig(fig2_pdf, bbox_inches="tight")
    fig2.savefig(fig2_png, bbox_inches="tight")
    plt.close(fig2)

    # --- Figure 3: memory (log-log, MiB)
    fig3 = plt.figure(figsize=(3.45, 2.6))
    ax3 = fig3.add_subplot(111)
    rss = np.array(
        [r["rss_delta_bytes"] if r["rss_delta_bytes"] is not None else np.nan for r in rows],
        dtype=float
    )
    ax3.plot(Ns, rss / (1024**2), marker="o", label="NumPy RSS delta (MiB)")

    theo = np.array([r["theo_bytes"] for r in rows], dtype=float)
    ax3.plot(Ns, theo / (1024**2), marker="^", label="Theoretical A+B+C (MiB)")

    if have_gpu:
        pool_total = np.array([r["cupy_pool_total_delta_bytes"] for r in rows], dtype=float)
        ax3.plot(Ns, pool_total / (1024**2), marker="s", label="CuPy pool total delta (MiB)")

    ax3.set_xscale("log", base=2)
    ax3.set_yscale("log")
    ax3.set_xlabel(r"Matrix size $N$")
    ax3.set_ylabel(r"Memory (MiB)")
    ax3.set_title(f"{title_prefix}  |  Memory")
    ax3.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax3.legend(loc="best")
    fig3.tight_layout(pad=0.3)

    fig3_pdf = outdir / f"{tag}_memory.pdf"
    fig3_png = outdir / f"{tag}_memory.png"
    fig3.savefig(fig3_pdf, bbox_inches="tight")
    fig3.savefig(fig3_png, bbox_inches="tight")
    plt.close(fig3)

    # --- Figure 4: speedup (optional)
    if have_gpu:
        fig4 = plt.figure(figsize=(3.45, 2.3))
        ax4 = fig4.add_subplot(111)
        ax4.plot(Ns, spd, marker="o")
        ax4.set_xscale("log", base=2)
        ax4.set_xlabel(r"Matrix size $N$")
        ax4.set_ylabel(r"Speedup (CPU time / GPU time)")
        ax4.set_title(f"{title_prefix}  |  Speedup")
        ax4.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        fig4.tight_layout(pad=0.3)

        fig4_pdf = outdir / f"{tag}_speedup.pdf"
        fig4_png = outdir / f"{tag}_speedup.png"
        fig4.savefig(fig4_pdf, bbox_inches="tight")
        fig4.savefig(fig4_png, bbox_inches="tight")
        plt.close(fig4)

    return True


def save_rows_csv(rows, out_csv: Path):
    import csv
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    # -------------------------
    # USER SETTINGS
    # -------------------------
    # With ~8GB VRAM, float32 at N=8192 is tight. A+B+C is ~768 MiB,
    # but GEMM workspace + allocator pool can increase the peak.
    # If you hit OOM, drop 8192 or stop at 6144.
    sizes_f32 = [256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    sizes_f64 = [256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]

    repeats_np = 10
    repeats_cp = 30
    warmup_np = 3
    warmup_cp = 10
    seed = 0

    # CPU thread setting: None means "use default/max threads".
    # You may set to 1 / 8 / 16 for comparisons.
    # cpu_threads = None
    cpu_threads = 4

    # -------------------------
    # Output naming convention
    # -------------------------
    tag = now_tag()
    host = platform.node() or "host"
    outdir = ensure_dir(Path("results") / f"gemm_{host}_{tag}")
    log_path = outdir / f"gemm_{host}_{tag}.log"
    csv_path = outdir / f"gemm_{host}_{tag}.csv"

    # -------------------------
    # Apply thread control
    # -------------------------
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
    log_print(logf, "\n[NumPy config]\n" + get_numpy_blas_info())

    if cp is not None:
        log_print(logf, f"\n[CuPy]\nCuPy version: {cp.__version__}")
        log_print(logf, get_gpu_info())
        try:
            log_print(logf, f"CUDA runtime: {cp.cuda.runtime.runtimeGetVersion()}")
            log_print(logf, f"cuBLAS: {cp.cuda.runtime.getVersion() if hasattr(cp.cuda.runtime, 'getVersion') else '(n/a)'}")
        except Exception as e:
            log_print(logf, f"(CUDA version info unavailable: {e})")
    else:
        log_print(logf, "\n[CuPy] not installed")

    # -------------------------
    # Run float32
    # -------------------------
    title_prefix = f"GEMM NumPy vs CuPy ({'max threads' if cpu_threads is None else f'{cpu_threads} threads'})"
    log_print(logf, "\n========== FLOAT32 ==========")
    rows32 = run_suite(
        sizes_f32, dtype=np.float32,
        repeats_np=repeats_np, repeats_cp=repeats_cp,
        warmup_np=warmup_np, warmup_cp=warmup_cp,
        seed=seed, logf=logf
    )

    # -------------------------
    # Run float64
    # -------------------------
    log_print(logf, "\n========== FLOAT64 ==========")
    rows64 = run_suite(
        sizes_f64, dtype=np.float64,
        repeats_np=repeats_np, repeats_cp=repeats_cp,
        warmup_np=warmup_np, warmup_cp=warmup_cp,
        seed=seed, logf=logf
    )

    # -------------------------
    # Save CSV + plots
    # -------------------------
    rows_all = rows32 + rows64
    save_rows_csv(rows_all, csv_path)
    log_print(logf, f"\nSaved CSV: {csv_path.name}")

    plot_and_save(rows32, outdir, tag=f"{tag}_f32", title_prefix=title_prefix + " | float32")
    plot_and_save(rows64, outdir, tag=f"{tag}_f64", title_prefix=title_prefix + " | float64")
    log_print(logf, "Saved figures: *_time.pdf/png, *_gflops.pdf/png, *_memory.pdf/png, (and *_speedup.* if GPU).")

    logf.close()
    print(f"\nDONE. Log: {log_path}")
    print(f"Figures/CSV in: {outdir}")


if __name__ == "__main__":
    main()
