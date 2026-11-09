"""Report descriptive ClinVar MC strata after the completed EVEE pilot."""

from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evee_readout_pilot_20261109_run02"
DATA = ROOT / "outputs/sh2_two_stage_20261104/data"
MIN_PER_CLASS = 10
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import average_precision_score, roc_auc_score
from evo2_mvp.evee_readout_pilot import read_locked_candidates
from evo2_mvp.utils import fingerprint, save_json, sha256_file


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completed_inputs():
    execution = read_json(RUN / "notebook_execution.json")
    require(execution.get("status") == "completed", "Notebook must complete before reading historical test results")
    notebook = read_json(Path(execution["executed_notebook"]))
    cells = [index for index, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"]
    require(execution.get("successful_cells") == cells, "Notebook code-cell completion record differs")
    require(all(notebook["cells"][i].get("execution_count") is not None
                and all(o.get("output_type") != "error" for o in notebook["cells"][i].get("outputs", []))
                for i in cells), "Executed Notebook contains an unexecuted or failed cell")
    lock, models = read_locked_candidates(RUN)
    summary = read_json(RUN / "run_report.json")
    historical = read_json(RUN / "historical_test_description.json")
    require(summary.get("status") == "completed" and historical.get("status") == "historical_test_described"
            and historical.get("test_used_for_selection") is False
            and historical["validation_lock_fingerprint"] == lock["lock_fingerprint"]
            and set(historical["models"]) == set(models), "Historical evaluation is incomplete or mismatched")
    require(sha256_file(RUN / "pilot_lock.json") == summary["candidate_lock_sha256"]
            and sha256_file(RUN / "historical_test_description.json") == summary["historical_test_report_sha256"],
            "Completed run provenance differs")
    prediction_files = {"val": "validation_predictions.csv", "test": "historical_test_predictions.csv"}
    for split, stage in (("val", "fit"), ("test", "describe_test")):
        stage_record = read_json(RUN / f"{stage}.stage.json")
        path = RUN / prediction_files[split]
        require(stage_record.get("status") == "completed"
                and stage_record["signature"]["experiment_identity"] == summary["experiment_identity"]
                and sha256_file(path) == stage_record["artifacts_sha256"][path.relative_to(ROOT).as_posix()],
                "Prediction export differs from the completed stage: " + split)
    return lock, models, prediction_files


def source_inputs(lock):
    metadata_path = DATA / "metadata.json"
    require(sha256_file(metadata_path) == lock["prepared_metadata_sha256"], "Prepared data changed after lock")
    metadata = read_json(metadata_path)
    variants_path, splits_path = ROOT / "data/variants.csv", ROOT / "data/splits.json"
    require(sha256_file(variants_path) == lock["variants_sha256"] == metadata["source_hashes"]["variants_sha256"]
            and sha256_file(splits_path) == metadata["source_hashes"]["splits_sha256"], "Original CSV or split changed")
    splits = read_json(splits_path)
    splits = splits.get("splits", splits)
    require(fingerprint(splits) == lock["split_fingerprint"] and splits == metadata["split_ids"], "Clinical split mismatch")
    expected = set(splits["val"]) | set(splits["test"])
    vcf = ROOT / "data/clinvar.vcf.gz"
    source = read_json(ROOT / "data/variants.metadata.json")
    source_sha = sha256_file(vcf)
    require(source_sha == source["source_sha256"] and source["genome_build"] == "GRCh38",
            "ClinVar VCF differs from the original cohort source")
    variants = {}
    with variants_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["variant_id"]
            if key not in expected:
                continue
            chrom = row["chrom"].lower().removeprefix("chr")
            pos = str(int(row["pos"]))
            ref, alt = row["ref"].upper(), row["alt"].upper()
            ids = set(row["source_accession"].split(";"))
            require(row["source_sha256"] == source_sha and row["source"] == "ClinVar"
                    and row["data_release_date"] == source["data_release_date"]
                    and row["genome_build"] == "GRCh38" and ids and not ids & {"", "."},
                    "Invalid original source accession/provenance: " + key)
            require(key == f"{chrom}-{pos}-{ref}-{alt}" and key not in variants and row["label"] in {"0", "1"},
                    "Invalid variant identity/label: " + key)
            variants[key] = (chrom.encode(), pos.encode(), ref.encode(), alt.encode(), ids, int(row["label"]))
    require(set(variants) == expected, "Some prediction variants lack original ClinVar source rows")
    return vcf, source_sha, source, splits, variants


def parse_mc(info):
    fields = [part[3:] for part in info.rstrip().split(b";") if part.startswith(b"MC=")]
    if not fields or fields == [b"."] or fields == [b""]:
        return "missing_mc", {}
    if len(fields) != 1:
        return "malformed_mc", {}
    terms = defaultdict(set)
    for entry in fields[0].decode("utf-8").split(","):
        term, separator, name = unquote(entry).partition("|")
        if not separator or not re.fullmatch(r"SO:\d{7}", term) or not name.strip() or "|" in name:
            return "malformed_mc", {}
        terms[term].add(name.strip())
    return ("usable" if len(terms) == 1 else "multiple_so_terms"), terms


def consequences(vcf, variants, release):
    wanted = defaultdict(dict)
    for key, (chrom, pos, ref, alt, _, _) in variants.items():
        wanted[chrom].setdefault(pos, {})[ref, alt] = key
    hits = defaultdict(list)
    alleles_found = set()
    header = {}
    with gzip.open(vcf, "rb") as handle:
        for line in handle:
            if line.startswith(b"#CHROM\t"):
                break
            for key, prefix in (("reference", b"##reference="), ("date", b"##fileDate="), ("mc", b"##INFO=<ID=MC,")):
                if line.startswith(prefix):
                    header[key] = line.rstrip().decode()
        require(header.get("reference") == "##reference=GRCh38"
                and header.get("date") == "##fileDate=" + release
                and "Sequence Ontology ID|molecular_consequence" in header.get("mc", ""),
                "VCF MC header does not match the source protocol")
        for line in handle:
            first = line.find(b"\t")
            chrom = line[:first].removeprefix(b"chr")
            positions = wanted.get(chrom)
            if positions is None:
                continue
            second = line.find(b"\t", first + 1)
            locus = positions.get(line[first + 1:second])
            if locus is None:
                continue
            fields = line.split(b"\t", 7)
            key = locus.get((fields[3], fields[4]))
            if key is None:
                continue
            alleles_found.add(key)
            ids = set(fields[2].decode("ascii").split(";")) & variants[key][4]
            if ids:
                status, terms = parse_mc(fields[7])
                hits[key].append((ids, status, terms))
    statuses, groups, names = {}, {}, defaultdict(set)
    for key in variants:
        records = hits[key]
        if key not in alleles_found:
            statuses[key] = "no_matching_allele"
        elif set().union(*(ids for ids, _, _ in records)) != variants[key][4]:
            statuses[key] = "source_accession_mismatch"
        else:
            exclusions = {status for _, status, _ in records}
            excluded = next((reason for reason in ("malformed_mc", "multiple_so_terms", "missing_mc")
                             if reason in exclusions), None)
            terms = set().union(*(set(terms) for _, _, terms in records))
            if excluded:
                statuses[key] = excluded
            elif len(terms) != 1:
                statuses[key] = "conflicting_source_records"
            else:
                term = terms.pop()
                statuses[key] = "usable_single_so_term"
                groups[key] = term
                for _, _, observed in records:
                    names[term].update(observed[term])
    return statuses, groups, names, header["mc"]


def split_report(path, split_name, ids, models, variants, statuses, assignments, names):
    allowed = set(ids)
    scores = {name: {} for name in models}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name, key = row["model"], row["variant_id"]
            score = float(row["score"])
            require(row["split"] == split_name and name in scores and key in allowed
                    and key not in scores[name] and int(row["label"]) == variants[key][5]
                    and 0 <= score <= 1, "Prediction row does not match the locked split/model/source")
            scores[name][key] = score
    require(all(set(values) == allowed for values in scores.values()), "Prediction model is missing locked variants")
    counts = Counter(statuses[key] for key in ids)
    grouped = defaultdict(list)
    for key in ids:
        if key in assignments:
            grouped[assignments[key]].append(key)
    results = {}
    for term, keys in sorted(grouped.items()):
        labels = [variants[key][5] for key in keys]
        positive, negative = sum(labels), len(keys) - sum(labels)
        result = {"vcf_mc_names": sorted(names[term]), "n": len(keys), "positives": positive, "negatives": negative}
        if min(positive, negative) >= MIN_PER_CLASS:
            result["models"] = {}
            for name in sorted(models):
                values = [scores[name][key] for key in keys]
                result["models"][name] = {"auroc": float(roc_auc_score(labels, values)),
                                          "average_precision": float(average_precision_score(labels, values))}
        results[term] = result
    return {"n": len(ids), "coverage": dict(sorted(counts.items())), "groups": results}


def main():
    lock, models, predictions = completed_inputs()
    vcf, vcf_sha, source, splits, variants = source_inputs(lock)
    began = perf_counter()
    statuses, assignments, names, mc_header = consequences(vcf, variants, source["data_release_date"])
    scan_seconds = perf_counter() - began
    report = {
        "status": "completed",
        "scope": "exploratory_within_existing_clinvar_mc_terms",
        "method": "Descriptive stratification of locked prediction scores by existing ClinVar MC Sequence Ontology IDs.",
        "relation_to_evee": "Within-term AUROC and average precision are descriptive subgroup estimands. EVEE consequence deconfounding is a separate adjustment procedure.",
        "historical_scope": "previously_examined_test_set_descriptive_only",
        "test_used_for_selection": False,
        "minimum_positives_and_negatives_per_group": MIN_PER_CLASS,
        "annotation_rule": "Exact chrom, pos, ref, alt and source accession match. Every matched source record must give one distinct SO ID and agree with the other source records.",
        "source": {"name": "ClinVar", "release_date": source["data_release_date"], "build": "GRCh38",
                   "vcf_sha256": vcf_sha, "mc_header": mc_header, "vcf_scan_seconds": scan_seconds},
        "validation_lock_fingerprint": lock["lock_fingerprint"],
        "validation": split_report(RUN / predictions["val"], "val", splits["val"], models, variants, statuses, assignments, names),
        "historical_test": split_report(RUN / predictions["test"], "historical_test", splits["test"], models, variants, statuses, assignments, names),
        "clinical_use": "research_only",
    }
    save_json(RUN / "consequence_group_report.json", report)
    print(json.dumps({
        "vcf_scan_seconds": scan_seconds,
        **{split: {"coverage": report[split]["coverage"],
                  "scored_groups": [term for term, value in report[split]["groups"].items() if "models" in value]}
           for split in ("validation", "historical_test")},
    }, indent=2))


if __name__ == "__main__":
    main()
