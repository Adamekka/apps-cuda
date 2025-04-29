import os
import subprocess

cuda_path = os.getenv("CUDA_PATH")

if not cuda_path:
    raise EnvironmentError("CUDA_PATH environment variable is not set.")

opencv_includes = subprocess.check_output(
    ["pkg-config", "--cflags", "opencv4"], text=True).strip()

opencv_libs = subprocess.check_output(
    ["pkg-config", "--libs-only-L", "opencv4"], text=True).strip()

if not opencv_includes or not opencv_libs:
    raise EnvironmentError(
        "OpenCV is not installed or pkg-config is not configured correctly.")

clangd_content = f"""CompileFlags:
  Add:
    - -std=c++17
    - -Dnoinline=1
    - --cuda-path={cuda_path}
    - -I{cuda_path}/include
    - -L{cuda_path}/lib
    - {opencv_includes}
    - {opencv_libs}
"""

with open(".clangd", "w") as f:
    f.write(clangd_content)

print("CONFIGURE: DONE")
