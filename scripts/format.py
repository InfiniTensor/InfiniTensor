import sys
from pathlib import Path
from subprocess import run

c_style_file = [".h", ".hh", ".hpp", ".c", ".cc", ".cpp", ".cxx", ".cu", ".mlu"]
py_file = ".py"
proj_path = Path(sys.path[0]).parent


# Formats one file under project path.
def format_file(file):
    file = Path(proj_path.joinpath(file))
    if file.suffix in c_style_file:
        run(f"clang-format-14 -i {file}", cwd=proj_path, shell=True)
        run(f"git add {file}", cwd=proj_path, shell=True)
    elif file.suffix == py_file:
        run(f"black {file}", cwd=proj_path, shell=True)
        run(f"git add {file}", cwd=proj_path, shell=True)


if len(sys.argv) == 1:
    # Last commit.
    print("Formats git added files.")
    for line in (
        run("git status", cwd=proj_path, capture_output=True, shell=True)
        .stdout.decode()
        .splitlines()
    ):
        line = line.strip()
        # Only formats git added files.
        for pre in ["new file:", "modified:"]:
            if line.startswith(pre):
                format_file(line[len(pre) :].strip())
            break
else:
    # Origin commit.
    origin = sys.argv[1]
    print(f'Formats changed files from "{origin}".')
    for line in (
        run(f"git diff {origin}", cwd=proj_path, capture_output=True, shell=True)
        .stdout.decode()
        .splitlines()
    ):
        diff = "diff --git "
        if line.startswith(diff):
            files = line[len(diff) :].split(" ")
            assert len(files) == 2
            assert files[0][:2] == "a/"
            assert files[1][:2] == "b/"
            format_file(files[1][2:])
