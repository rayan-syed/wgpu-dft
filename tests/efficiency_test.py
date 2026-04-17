from pathlib import Path
import csv
import subprocess
import os

import numpy as np
import matplotlib.pyplot as plt

# benchmarking params
WARMUP_DIMENSIONS = [128]
DIMENSIONS = [256, 512, 1024, 2048, 4096]
REPEATS = 5

# output artifacts
OUTPUT_DIR = "tests/artifacts"
CSV_FILE = f"{OUTPUT_DIR}/efficiency_results.csv"
RAW_CSV_FILE = f"{OUTPUT_DIR}/efficiency_raw_results.csv"
FORWARD_PLOT = f"{OUTPUT_DIR}/efficiency_forward.png"
BACKWARD_PLOT = f"{OUTPUT_DIR}/efficiency_backward.png"


def generate_input_file(path, dimension):
    real = np.random.rand(dimension, dimension).astype(np.float32)
    imag = np.random.rand(dimension, dimension).astype(np.float32)
    values = real + 1j * imag

    with open(path, "w") as handle:
        handle.write(f"{dimension} {dimension}\n")
        for row in values:
            entries = [f"{value.real} {value.imag}" for value in row]
            handle.write(" ".join(entries) + "\n")


def build_wgpu():
    subprocess.run(
        ["cmake", "-B", "build", "-S", "."],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["cmake", "--build", "build"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def run_transform(force_dft, mode):
    command = ["build/wgpu_dft", f"--mode={mode}"]
    if force_dft:
        command.append("--force-dft")

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_benchmark_output(output):
    stats = {}
    for line in output.strip().splitlines():
        parts = line.split()
        if not parts or parts[0] == "benchmark":
            continue
        if parts[0] == "raw_ms":
            stats["raw_ms"] = [float(value) for value in parts[1:]]
        else:
            stats[parts[0]] = float(parts[1])

    return {
        "mean_ms": stats["mean_ms"],
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
        "raw_ms": stats["raw_ms"],
    }


def benchmark_pass(force_dft, mode, repeats):
    command = ["build/wgpu_dft", f"--mode={mode}", f"--benchmark={repeats}"]
    if force_dft:
        command.append("--force-dft")

    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
    )
    return parse_benchmark_output(result.stdout)


def benchmark_dimension(dimension, repeats):
    generate_input_file("tests/artifacts/input.txt", dimension)
    results = {"DFT": {}, "FFT": {}}
    for algorithm, force_dft in [("DFT", True), ("FFT", False)]:
        for direction in ["forward", "backward"]:
            print(f"Running benchmarks: {dimension}x{dimension} | {algorithm} | {direction}")
            results[algorithm][direction] = benchmark_pass(
                force_dft=force_dft,
                mode=direction,
                repeats=repeats,
            )
    return results


def run_warmup():
    for dimension in WARMUP_DIMENSIONS:
        generate_input_file("tests/artifacts/input.txt", dimension)
        for algorithm, force_dft in [("DFT", True), ("FFT", False)]:
            for direction in ["forward", "backward"]:
                print(f"Warmup: {dimension}x{dimension} | {algorithm} | {direction}")
                run_transform(force_dft=force_dft, mode=direction)


def write_csv(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CSV_FILE, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "algorithm", "direction", "mean_ms", "min_ms", "max_ms"])
        for dimension, algorithms in results.items():
            for algorithm, directions in algorithms.items():
                for direction, stats in directions.items():
                    writer.writerow([
                        dimension,
                        algorithm,
                        direction,
                        f"{stats['mean_ms']:.6f}",
                        f"{stats['min_ms']:.6f}",
                        f"{stats['max_ms']:.6f}",
                    ])


def write_raw_csv(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RAW_CSV_FILE, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "algorithm", "direction", "run_index", "runtime_ms"])
        for dimension, algorithms in results.items():
            for algorithm, directions in algorithms.items():
                for direction, stats in directions.items():
                    for run_index, runtime_ms in enumerate(stats["raw_ms"], start=1):
                        writer.writerow([dimension, algorithm, direction, run_index, f"{runtime_ms:.6f}"])


def plot_direction(results, direction, destination):
    dimensions = sorted(results.keys())
    dft_values = [results[dimension]["DFT"][direction]["mean_ms"] for dimension in dimensions]
    fft_values = [results[dimension]["FFT"][direction]["mean_ms"] for dimension in dimensions]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dimensions, dft_values, marker="o", linewidth=2, label="DFT")
    ax.plot(dimensions, fft_values, marker="s", linewidth=2, label="FFT")
    ax.set_title(f"{direction.capitalize()} Transform Runtime")
    ax.set_xlabel("Matrix dimension (N for NxN)")
    ax.set_ylabel("Mean runtime (ms)")
    ax.set_xticks(dimensions)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return True


def main():
    print("Building WGPU project...")
    build_wgpu()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting warmup runs...")
    run_warmup()
    print("Warmup complete.")

    print(f"Benchmark dimensions: {DIMENSIONS}")
    print(f"Repeats per measurement: {REPEATS}")

    results = {}
    for dimension in DIMENSIONS:
        results[dimension] = benchmark_dimension(dimension, REPEATS)

    write_csv(results)
    write_raw_csv(results)

    forward_created = plot_direction(results, "forward", FORWARD_PLOT)
    backward_created = plot_direction(results, "backward", BACKWARD_PLOT)

    print("Benchmark artifacts saved to tests/artifacts/")
    if not (forward_created and backward_created):
        print("Plot generation skipped because matplotlib is not installed.")


if __name__ == "__main__":
    main()
