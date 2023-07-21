from subprocess import run
import sys
from pathlib import Path

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
            if file.suffix in [".h", ".hh", ".hpp", ".c", ".cc", ".cpp", ".cxx"]:
                run(f"clang-format-14 -i {file}", cwd=proj_path, shell=True)
                run(f"git add {file}", cwd=proj_path, shell=True)
        break
