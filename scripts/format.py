from subprocess import run
import sys
from pathlib import Path

c_style_file = [".h", ".hh", ".hpp", ".c", ".cc", ".cpp", ".cxx", ".cu", ".mlu"]
py_file = ".py"
proj_path = Path(sys.path[0]).parent

for line in (
    run("git status", cwd=proj_path, capture_output=True, shell=True)
    .stdout.decode()
    .splitlines()
):
    line = line.strip()
    # Only formats git added files.
    for pre in ["new file:", "modified:"]:
        if line.startswith(pre):
            file = Path(proj_path.joinpath(line[len(pre) :].strip()))
            if file.suffix in c_style_file:
                run(f"clang-format-14 -i {file}", cwd=proj_path, shell=True)
                run(f"git add {file}", cwd=proj_path, shell=True)
            elif file.suffix == py_file:
                run(f"black {file}", cwd=proj_path, shell=True)
                run(f"git add {file}", cwd=proj_path, shell=True)
        break
