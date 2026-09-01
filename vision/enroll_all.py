from __future__ import annotations

import subprocess
import sys


def run_step(args):
    completed = subprocess.run([sys.executable, "-m"] + args, check=False)
    if completed.returncode != 0:
        sys.exit(completed.returncode)


def main():
    run_step(["vision.extract_faces"])
    run_step(["vision.train"])
    run_step(["vision.build_embeddings"])


if __name__ == "__main__":
    main()
