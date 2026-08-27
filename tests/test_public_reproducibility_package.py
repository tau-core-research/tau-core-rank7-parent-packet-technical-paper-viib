import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_publication_files_exist():
    for name in [
        "README.md",
        "CITATION.cff",
        "paperVIIB.pdf",
        "arxiv_submission_source.zip",
        "paperVIIB_submission_source/main.tex",
        "paperVIIB_submission_source/p3_main.tex",
        "paperVIIB_submission_source/public_audit.tex",
        "paperVIIB_submission_source/appendices/p3_technical_derivations.tex",
    ]:
        assert (ROOT / name).exists(), name


def test_claim_boundary_and_owned_results():
    main = (ROOT / "paperVIIB_submission_source/main.tex").read_text()
    p3 = (ROOT / "paperVIIB_submission_source/p3_main.tex").read_text()
    appendix = (
        ROOT / "paperVIIB_submission_source/appendices/p3_technical_derivations.tex"
    ).read_text()
    assert "Class-minimal" in main
    assert "does not mean" in main
    assert "Support--port reconstruction and class-minimal" in p3
    assert "Tau-local standard subsystem composition" in p3
    assert "Operator-square source selector" in p3
    assert "Pointed exterior-seed closure" in p3
    assert "Enriched-parent variational ownership" in appendix
    assert "unrestricted physical base--seed parent" in main
    assert "Handoff to the joint-terminal assembly" in p3
    assert "Conditional three-level enriched-MVP closure" not in p3


def test_generated_audits():
    checks = {
        "rank7_parent_law_audit.json": 8,
        "tau_local_standard_composition_audit.json": 8,
        "component_local_p3_joint_occupation_audit.json": 8,
        "pair_top_exterior_closure_audit.json": 7,
    }
    for name, minimum in checks.items():
        data = json.loads((ROOT / "data/derived" / name).read_text())
        assert data["status"] == "PASS"
        assert data["checks_passed"] >= minimum
    public = json.loads(
        (ROOT / "data/derived/public_source_eligibility_audit.json").read_text()
    )
    assert public["eligible_count"] == 0
    assert public["candidate_count"] >= 35
    pair = json.loads(
        (ROOT / "data/derived/minimal_pair_nature_test.json").read_text()
    )
    assert pair["candidate_count"] == 31
    assert pair["eligible_count"] == 0
    assert pair["checks_passed"] == pair["checks_total"]
    assert pair["product_free_marginal_disk_falsifier"]["law"] == (
        "<C_a>^2+<C_b>^2 <= 1"
    )


def test_arxiv_archive_is_self_contained():
    with zipfile.ZipFile(ROOT / "arxiv_submission_source.zip") as zf:
        names = set(zf.namelist())
    assert "main.tex" in names
    assert "p3_main.tex" in names
    assert "public_audit.tex" in names
    assert "appendices/p3_technical_derivations.tex" in names
    assert "refs.bib" in names
