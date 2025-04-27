import os

cuda_path = os.getenv("CUDA_PATH")

if not cuda_path:
    raise EnvironmentError("CUDA_PATH environment variable is not set.")

clangd_content = f"""CompileFlags:
  Add:
    - -std=c++17
    - -Dnoinline=1
    - --cuda-path={cuda_path}
    - -L{cuda_path}/lib
    - -I{cuda_path}/include
"""

with open(".clangd", "w") as f:
    f.write(clangd_content)

print("CONFIGURE: DONE")
