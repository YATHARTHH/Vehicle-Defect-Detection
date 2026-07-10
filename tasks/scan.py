import os
import shutil
import subprocess
import sys


def git_bash_from_exec_path():
    try:
        git_exec = subprocess.check_output(["git", "--exec-path"], text=True).strip()
    except Exception:
        return None

    git_root = git_exec
    for _ in range(3):
        git_root = os.path.dirname(git_root)

    candidate = os.path.join(git_root, "bin", "bash.exe")
    if os.path.exists(candidate):
        return candidate

    return None


def is_wsl_bash(path):
    if not path:
        return False
    normalized = os.path.normcase(os.path.normpath(path))
    return normalized.endswith(os.path.normcase(r"\Windows\System32\bash.exe"))


def find_bash():
    if os.name == "nt":
        git_bash = git_bash_from_exec_path()
        if git_bash:
            return git_bash

    bash = shutil.which("bash")
    if bash and not is_wsl_bash(bash):
        return bash

    git_bash = git_bash_from_exec_path()
    if git_bash:
        return git_bash

    return None


bash = find_bash()
if not bash:
    print("ERROR: bash not found. Run from Git Bash or add Git/bin to PATH.")
    sys.exit(1)

result = subprocess.run([bash, ".githooks/pre-commit"])
sys.exit(result.returncode)
