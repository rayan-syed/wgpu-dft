import numpy as np
import subprocess

ROWS, COLS = 512, 512
tolerance = 1e-4

def generate_input_file(filename, rows=512, cols=512):
    real = np.random.rand(rows, cols).astype(np.float32)
    imag = np.random.rand(rows, cols).astype(np.float32)
    input_complex = real + 1j * imag

    # write matrix to txt file for wgpu to read in
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")
        for i in range(rows):
            row_data = []
            for j in range(cols):
                row_data.append(f"{input_complex[i, j].real} {input_complex[i, j].imag}")
            f.write(" ".join(row_data) + "\n")
    return input_complex

def build_and_run_wgpu():
    subprocess.run(["cmake", "-B", "build", "-S", "."], check=True)
    subprocess.run(["cmake", "--build", "build"], check=True)
    # capture stdout of wgpu for comparison
    result = subprocess.run(["./build/wgpu_dft"], check=True, stdout=subprocess.PIPE, universal_newlines=True)
    return result.stdout

def parse_wgpu_output(output):
    lines = output.strip().splitlines()
    dims = lines[0].split()
    rows, cols = int(dims[0]), int(dims[1])
    result = np.zeros((rows, cols), dtype=np.complex64)
    for i in range(rows):
        parts = lines[i+1].strip().split()
        for j in range(cols):
            re = float(parts[j * 2])
            im = float(parts[j * 2 + 1])
            result[i, j] = re + 1j * im
    return result

def main():
    input_filename = "input.txt"
    rows, cols = ROWS, COLS
    rel_tol = tolerance # relative tolerance

    np_input = generate_input_file(input_filename, rows, cols)
    np_dft = np.fft.fft2(np_input)
    output = build_and_run_wgpu()
    wgpu_dft = parse_wgpu_output(output)

    # get relative difference
    is_close = np.isclose(wgpu_dft, np_dft, rtol=rel_tol, atol=0) 
    mismatches = np.sum(~is_close)

    print(f"Number of elements with relative difference > {rel_tol}: {mismatches}")

if __name__ == "__main__":
    main()
