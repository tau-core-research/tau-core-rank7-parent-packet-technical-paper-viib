#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paperVIIB_submission_source"


def run(*args):
    print("$", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def build_archive():
    archive = ROOT / "arxiv_submission_source.zip"
    members = [
        SOURCE / "main.tex",
        SOURCE / "p3_main.tex",
        SOURCE / "public_audit.tex",
        SOURCE / "refs.bib",
        SOURCE / "appendices" / "p3_technical_derivations.tex",
    ]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in members:
            zf.write(path, path.relative_to(SOURCE))


def main():
    py = sys.executable
    for script in [
        "audit_public_source_eligibility.py",
        "audit_minimal_pair_nature_test.py",
        "audit_federated_gram_completion.py",
        "audit_rank7_parent_law.py",
        "audit_tau_local_standard_composition.py",
        "audit_component_local_p3_joint_occupation.py",
        "audit_pair_top_exterior_closure.py",
    ]:
        run(py, f"scripts/{script}")
    run("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
        "-cd", "paperVIIB_submission_source/main.tex")
    shutil.copy2(SOURCE / "main.pdf", ROOT / "paperVIIB.pdf")
    build_archive()
    run(py, "-m", "pytest", "-q")
    print("TECHNICAL_PAPER_VIIB_REPRODUCTION_COMPLETE")


if __name__ == "__main__":
    main()
