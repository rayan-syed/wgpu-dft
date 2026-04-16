import numpy as np
import subprocess

ROWS, COLS = 512, 512
TOLERANCE = 1e-4

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

def build_wgpu():
    subprocess.run(["cmake", "-B", "build", "-S", "."], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", "build"], check=True, stdout=subprocess.DEVNULL)

def run_wgpu(force_dft=False):
    command = ["./build/wgpu_dft"]
    if force_dft:
        command.append("--force-dft")

    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    return result.stdout

def parse_wgpu_output(output):
    lines = output.strip().splitlines()
    dims = lines[0].split()
    rows, cols = int(dims[0]), int(dims[1])
    result = np.zeros((rows, cols), dtype=np.complex64)
    iresult = np.zeros((rows, cols), dtype=np.complex64)
    for i in range(rows):
        parts = lines[i+1].strip().split()
        for j in range(cols):
            re = float(parts[j * 2])
            im = float(parts[j * 2 + 1])
            result[i, j] = re + 1j * im
    for i in range(rows):
        parts = lines[rows+i+1].strip().split()
        for j in range(cols):
            re = float(parts[j * 2])
            im = float(parts[j * 2 + 1])
            iresult[i, j] = re + 1j * im
    return result, iresult

def compare_results(wgpu_result, numpy_result, rel_tol=TOLERANCE):
    is_close = np.isclose(wgpu_result, numpy_result, rtol=rel_tol, atol=1e-4)
    mismatches = np.sum(~is_close)
    offender = None
    for row, col in np.argwhere(~is_close):
        actual = wgpu_result[row, col]
        expected = numpy_result[row, col]
        abs_err = abs(actual - expected)
        rel_err = abs_err / abs(expected) if abs(expected) > 0 else abs_err
        if offender is None or rel_err > offender["rel_err"]:
            offender = {
                "index": (int(row), int(col)),
                "actual": actual,
                "expected": expected,
                "abs_err": float(abs_err),
                "rel_err": float(rel_err),
            }

    return mismatches, offender

def print_divider(char="-", section_width=30):
    print(char * section_width)

def print_section(title):
    print()
    print_divider("=")
    print(title)
    print_divider("=")

def print_subsection(title):
    print()
    print(title)
    print_divider("-",10)

def print_summary(mismatches, offender):
    print(f"mismatches: {mismatches}")
    if offender is None:
        print("top offender: none")
        return

    row, col = offender["index"]
    print(
        f"top offender: ({row}, {col}) "
        f"| rel_err={offender['rel_err']:.6e} "
        f"| abs_err={offender['abs_err']:.6e}"
    )
    print(f"wgpu : {offender['actual']}")
    print(f"numpy: {offender['expected']}")

def test_mode(title, force_dft, np_input):
    np_forward = np.fft.fft2(np_input).astype(np.complex64)
    np_inverse = np.fft.ifft2(np_input).astype(np.complex64)

    output = run_wgpu(force_dft=force_dft)
    wgpu_forward, wgpu_inverse = parse_wgpu_output(output)

    forward_mismatches, forward_offender = compare_results(wgpu_forward, np_forward)
    inverse_mismatches, inverse_offender = compare_results(wgpu_inverse, np_inverse)

    print_section(title)
    print_subsection("Forward")
    print_summary(forward_mismatches, forward_offender)
    print_subsection("Backward")
    print_summary(inverse_mismatches, inverse_offender)

def main():
    print("Building WGPU DFT project...")
    build_wgpu()

    np_input = generate_input_file("tests/artifacts/input.txt", ROWS, COLS)
    print(f"Input shape: {ROWS}x{COLS}")

    test_mode("DFT", force_dft=True, np_input=np_input)
    test_mode("FFT", force_dft=False, np_input=np_input)

if __name__ == "__main__":
    main()
