"""Collect Nsight Compute metrics for one configuration (run after it has been built)."""
import csv
import io
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "memory_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
}


def profile(kernel: int, tile: int, coarsen: int, size: int = 2048) -> dict:
    binary = ROOT / "build" / f"gemm_k{kernel}_t{tile}_c{coarsen}"
    cmd = ["ncu", "--csv", "--kernel-name", "regex:gemm_", "--launch-count", "1",
           "--metrics", ",".join(METRICS), str(binary), str(size), str(size), str(size), "1"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return {"error": out.stderr[-1000:]}
    rows = [line for line in out.stdout.splitlines() if line.startswith('"')]
    metrics = {}
    for row in csv.DictReader(io.StringIO("\n".join(rows))):
        name = row.get("Metric Name")
        if name in METRICS:
            metrics[METRICS[name]] = float(row["Metric Value"].replace(",", ""))
    return metrics


if __name__ == "__main__":
    print(profile(2, 64, 8))
