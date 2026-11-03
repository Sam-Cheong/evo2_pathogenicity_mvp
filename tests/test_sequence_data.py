import csv
import importlib.util
from pathlib import Path

import pytest

from evo2_mvp.data import load_variants, make_splits, verify_splits
from evo2_mvp.sequence import ReferenceGenome, Variant


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / "ref.fa"
    path.write_text(">chr1\n" + "ACGT" * 300 + "\n", encoding="ascii")
    return path


def test_context_coordinates_and_replacement(fasta):
    variant = Variant("chr1", 401, "A", "G", "GENE", 1)
    with ReferenceGenome(fasta) as genome:
        result = genome.context(variant, 256)
    assert variant.id == "1-401-A-G"
    assert len(result.ref_sequence) == len(result.alt_sequence) == 256
    assert result.variant_index == 128
    assert result.ref_sequence[128] == "A"
    assert result.alt_sequence[128] == "G"
    assert [i for i, (r, a) in enumerate(zip(result.ref_sequence, result.alt_sequence)) if r != a] == [128]


def test_reject_mismatch_bounds_and_ambiguous(fasta, tmp_path):
    with ReferenceGenome(fasta) as genome:
        with pytest.raises(ValueError, match="REF mismatch"):
            genome.context(Variant("1", 401, "C", "G"))
        with pytest.raises(ValueError, match="full"):
            genome.context(Variant("1", 1, "A", "G"))
        with pytest.raises(ValueError, match="full"):
            genome.context(Variant("1", 1200, "T", "G"))
    n_path = tmp_path / "n.fa"
    n_path.write_text(">1\n" + "A"*200 + "N" + "A"*399 + "\n")
    with ReferenceGenome(n_path) as genome:
        with pytest.raises(ValueError, match="ambiguous"):
            genome.context(Variant("1", 300, "A", "C"))


@pytest.mark.parametrize("args", [("X", 1, "A", "G"), ("1", 0, "A", "G"),
                                      ("1", 1, "AA", "G"), ("1", 1, "A", "A"),
                                      ("1", 1, "N", "A"), ("1", 1.5, "A", "G")])
def test_invalid_variant(args):
    with pytest.raises(ValueError):
        Variant(*args)


def variants_for_split():
    return [Variant("1", 400+i, "A", "G", f"G{i//2}", i % 2) for i in range(24)]


def test_group_split_reproducible_no_leakage():
    variants = variants_for_split()
    splits = make_splits(variants)
    assert splits == make_splits(variants)
    verify_splits(variants, splits)
    lookup = {v.id: v for v in variants}
    genes = [{lookup[x].gene for x in ids} for ids in splits.values()]
    assert not (genes[0] & genes[1] or genes[0] & genes[2] or genes[1] & genes[2])
    assert all({lookup[x].label for x in ids} == {0, 1} for ids in splits.values())


def test_overlapping_genes_join_components():
    variants = variants_for_split()
    variants[0] = Variant("1", 400, "A", "G", "G0;G1", 0)
    splits = make_splits(variants)
    assigned = {variant_id: name for name, ids in splits.items() for variant_id in ids}
    assert assigned[variants[0].id] == assigned[variants[3].id]


def test_insufficient_or_missing_genes():
    with pytest.raises(ValueError, match="independent gene"):
        make_splits([Variant("1", 400+i, "A", "G", "G1", i % 2) for i in range(12)])
    with pytest.raises(ValueError, match="known gene"):
        make_splits([Variant("1", 400+i, "A", "G", "", i % 2) for i in range(12)])


def test_split_tampering_detected():
    variants = variants_for_split()
    splits = make_splits(variants)
    splits["test"].append(splits["train"][0])
    with pytest.raises(ValueError, match="multiple partitions"):
        verify_splits(variants, splits)


def test_csv_duplicate_rejected(tmp_path):
    path = tmp_path / "variants.csv"
    path.write_text("chrom,pos,ref,alt,gene,label\n1,400,A,G,G1,1\n1,400,A,G,G1,1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_variants(path)


def load_prepare():
    path = Path(__file__).parents[1] / "scripts" / "prepare_clinvar.py"
    spec = importlib.util.spec_from_file_location("prepare_clinvar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clinvar_strict_filter_and_late_duplicate(tmp_path, fasta):
    module = load_prepare()
    assert module.clinical_label("Pathogenic/Likely_pathogenic") == 1
    assert module.clinical_label("Uncertain_significance") is None
    assert module.clinical_label("Pathogenic|risk_factor") is None
    assert module.review_stars("criteria_provided,_multiple_submitters,_no_conflicts") == 2
    assert module.review_stars("criteria_provided,_single_submitter") == 0
    vcf = tmp_path / "clinvar.vcf"
    lines = ["##fileformat=VCFv4.1", "##reference=GRCh38", "##fileDate=2026-11-01",
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
    def record(pos, classification, review="reviewed_by_expert_panel", ref="A", extra=""):
        return f"1\t{pos}\t{pos}\t{ref}\tG\t.\t.\tCLNSIG={classification};CLNREVSTAT={review};GENEINFO=G1:1{extra}"
    lines.extend([record(401,"Pathogenic"), record(405,"Benign"), record(409,"Uncertain_significance"),
                  record(413,"Benign","criteria_provided,_single_submitter"),
                  record(417,"Pathogenic",ref="C"), record(421,"Likely_benign"),
                  record(401,"Benign")])
    vcf.write_text("\n".join(lines)+"\n")
    out = tmp_path / "selected.csv"
    report = module.prepare(vcf, fasta, out, "2026-11-01", target=1)
    assert report["eligible"] == 2
    assert report["selected"] == 1
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["pos"] in {"405","421"}
    assert rows[0]["synthetic"] == "false"
    assert len(rows[0]["source_sha256"]) == 64
