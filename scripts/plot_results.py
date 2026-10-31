"""Plot GFLOPS for every successful configuration against the cuBLAS baseline."""
import matplotlib.pyplot as plt

from agent import store
from bench.run_benchmark import KERNEL_NAMES

rows = [r for r in store.all_results() if r["status"] == "ok"]
rows.sort(key=lambda r: r["gflops"])
labels = [f"{KERNEL_NAMES[r['kernel']]}\nT{r['tile']} C{r['coarsen']}" for r in rows]
gflops = [r["gflops"] for r in rows]
cublas = max((r["gflops"] / (r["pct_of_cublas"] / 100) for r in rows if r["pct_of_cublas"]), default=None)

plt.figure(figsize=(max(8, len(rows) * 0.8), 5))
plt.bar(labels, gflops)
if cublas:
    plt.axhline(cublas, linestyle="--", label=f"cuBLAS ({cublas:.0f} GFLOPS)")
    plt.legend()
plt.ylabel("GFLOPS")
plt.title("SGEMM performance by configuration (2048 x 2048)")
plt.tight_layout()
plt.savefig("results/gflops.png", dpi=150)
print("Saved results/gflops.png")
