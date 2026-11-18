"""Command-line interface: ``mss <command> --help``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from . import __version__

log = logging.getLogger("mss")

OFFLINE = "[offline]"
ONLINE = "[needs internet]"


# ------------------------------------------------------------------------------- helpers
def _die(msg: str) -> None:
    raise SystemExit(f"mss: error: {msg}")


def _need_file(path, what="file"):
    if path is None or not Path(path).is_file():
        _die(f"{what} not found: {path}")


def _write_tsv(df: pd.DataFrame, path, index=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index, float_format="%.6g")
    log.info("wrote %s (%d rows)", path, len(df))


def _write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
    log.info("wrote %s", path)


def _read_manifest(path) -> pd.DataFrame:
    _need_file(path, "manifest")
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "genome_fasta" not in m.columns:
        _die(f"{path}: manifest needs a 'genome_fasta' column")
    if "genome_id" not in m.columns:
        m["genome_id"] = [Path(p).name.split(".fna")[0].split(".fa")[0] for p in m["genome_fasta"]]
    if m["genome_id"].duplicated().any():
        _die(f"{path}: duplicated genome_id values: {m.loc[m.genome_id.duplicated(), 'genome_id'].head(3).tolist()}")
    base = Path(path).parent
    m["genome_fasta"] = [p if Path(p).is_absolute() or Path(p).exists() else str(base / p)
                         for p in m["genome_fasta"]]
    return m


def _features_for(manifest, args, feature_type=None):
    from .features import genome_kmer_table, load_embeddings

    emb = getattr(args, "embeddings", None)
    if isinstance(emb, str):
        emb = [emb]
    if emb or (feature_type and feature_type.startswith("embedding")):
        if not emb:
            _die("this model was trained on external embeddings; pass --embeddings")
        tables = []
        for path in emb:
            _need_file(path, "embeddings")
            tables.append(load_embeddings(path, args.embedding_key))
        X = tables[0]
        for path, T in zip(emb[1:], tables[1:]):
            clash = X.columns.intersection(T.columns)
            if len(clash):
                _die(f"{path}: feature columns already in an earlier table (e.g. {list(clash[:3])})")
            n0 = len(X)
            X = X.join(T, how="inner")
            log.info("joined %s: %d features; %d genomes in both (%d and %d before)",
                     Path(path).name, X.shape[1], len(X), n0, len(T))
        if args.embedding_key not in manifest.columns:
            _die(f"manifest has no column {args.embedding_key!r} to match embeddings")
        key_to_gid = dict(zip(manifest[args.embedding_key].astype(str), manifest["genome_id"]))
        missing = sorted(set(key_to_gid) - set(X.index))
        if missing:
            log.warning("%d manifest genomes lack embeddings (e.g. %s); they are skipped",
                        len(missing), missing[:3])
        X = X.loc[[k for k in key_to_gid if k in X.index]]
        X.index = pd.Index([key_to_gid[k] for k in X.index], name="genome_id")
        return X, "embedding:" + "+".join(Path(p).name for p in emb), pd.DataFrame()
    t0 = time.time()
    X, stats = genome_kmer_table(manifest, k=args.k, cache_dir=args.cache_dir, threads=args.threads)
    bad = stats[stats["error"] != ""]
    for r in bad.itertuples():
        log.warning("genome %s skipped: %s", r.genome_id, r.error)
    log.info("k-mer features for %d genomes in %.1fs", len(X), time.time() - t0)
    return X, f"canonical_kmer_k{args.k}", stats


def _add_feature_args(p):
    p.add_argument("--k", type=int, default=5, help="k-mer length (default 5)")
    p.add_argument("--cache-dir", default=".mss_cache/kmers",
                   help="per-genome k-mer cache directory (default .mss_cache/kmers)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="worker processes (default $SLURM_CPUS_PER_TASK or 1)")
    p.add_argument("--embeddings", nargs="+",
                   help="one or more precomputed feature TSVs (genome key + dna_*/protein_*/f_*/kmer_* columns), "
                        "used INSTEAD of on-the-fly k-mers; several are joined on the key (genomes in all of them), "
                        "e.g. features/kmer5.tsv.gz features/pfam_pa.tsv.gz for a k-mer + Pfam hybrid")
    p.add_argument("--embedding-key", default="genome_id",
                   help="manifest column matching the embedding key column (default genome_id)")


# ------------------------------------------------------------------------------- commands
def cmd_bacdive(a):
    from .bacdive import BacDiveClient, build_reference, summarize_reference
    from .ontology import load_oxygen_map

    ids: list[int] = list(a.ids or [])
    if a.ids_file:
        _need_file(a.ids_file)
        ids += [int(x) for x in Path(a.ids_file).read_text().split() if x.strip().isdigit()]
    names = list(a.taxon or [])
    if a.taxa_file:
        _need_file(a.taxa_file)
        names += [l.strip() for l in Path(a.taxa_file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not ids and not names and not a.id_range:
        _die("give --ids, --ids-file, --taxon, --taxa-file or --id-range")
    omap = load_oxygen_map(a.oxygen_map)
    client = BacDiveClient(cache_dir=a.cache_dir)
    for n in names:
        found = client.search_name(n)
        log.info("taxon %r -> %d BacDive IDs", n, len(found))
        ids += found
    import itertools
    streams = [client.fetch(ids)] if ids else []
    if a.id_range:
        lo, hi = a.id_range
        log.info("scanning BacDive IDs %d-%d (resumable via %s/_blocks)", lo, hi, a.cache_dir)
        streams.append(client.scan_range(lo, hi))
    seen: set[int] = set()

    def unique(it):
        for i, r in it:
            if i not in seen:
                seen.add(i)
                yield i, r
    ref = build_reference(unique(itertools.chain(*streams)), omap)
    _write_tsv(ref, a.out)
    summ = summarize_reference(ref)
    _write_json(summ, Path(a.out).with_suffix(".summary.json"))
    if summ["oxygen_unmapped_strings"]:
        log.warning("oxygen strings left UNMAPPED (label empty): %s. Map them explicitly with "
                    "--oxygen-map if appropriate.", summ["oxygen_unmapped_strings"])


def cmd_genome_manifest(a):
    from .genomes import load_gtdb, make_manifest

    _need_file(a.reference, "reference")
    ref = pd.read_csv(a.reference, sep="\t", dtype=str, keep_default_na=False)
    gtdb = None
    if a.gtdb_metadata:
        for f in a.gtdb_metadata:
            _need_file(f, "GTDB metadata")
        gtdb = load_gtdb(a.gtdb_metadata)
    elif a.gtdb_reps_only:
        _die("--gtdb-reps-only needs --gtdb-metadata")
    m, dup = make_manifest(ref, a.genome_dir, a.require_label, gtdb, a.gtdb_reps_only)
    if m.empty:
        _die("no reference rows have a genome accession")
    _write_tsv(m, a.out)
    if len(dup):
        _write_tsv(dup, Path(a.out).with_suffix(".duplicate_accessions.tsv"))


def cmd_download_genomes(a):
    from .genomes import download_genomes

    _need_file(a.manifest, "manifest")
    m = pd.read_csv(a.manifest, sep="\t", dtype=str, keep_default_na=False)
    if a.limit:
        m = m.head(a.limit)
    ok, fails = download_genomes(m, a.outdir, batch_size=a.batch_size, datasets_bin=a.datasets_bin,
                                 workers=a.workers)
    _write_tsv(ok, a.out)
    _write_tsv(fails, Path(a.out).with_suffix(".failures.tsv"))
    log.info("%d genomes ready, %d failed", len(ok), len(fails))


def cmd_pfam_annotate(a):
    from .pfam import annotate_genomes

    _need_file(a.hmm, "Pfam HMM file")
    m = _read_manifest(a.manifest)
    tid = a.task_id if a.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n = a.n_tasks if a.n_tasks is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if not 0 <= tid < n:
        _die(f"task id {tid} outside 0..{n - 1}")
    res = annotate_genomes(m, a.hmm, a.cache_dir, cpus=a.threads, task_id=tid, n_tasks=n, batch=a.batch)
    log.info("pfam-annotate: %s", res)


def cmd_pfam_table(a):
    from .pfam import pfam_table

    m = _read_manifest(a.manifest)
    X, stats = pfam_table(m, a.cache_dir, value=a.value, min_genomes=a.min_genomes)
    _write_tsv(stats, Path(a.out).with_suffix("").with_suffix(".stats.tsv"))
    bad = stats[stats.error != ""]
    if len(bad):
        log.warning("%d genomes left out (%s)", len(bad), bad.error.value_counts().to_dict())
    if (bad.error == "not annotated").any() and not a.allow_missing:
        _die(f"{(bad.error == 'not annotated').sum()} genomes are not annotated yet; finish pfam-annotate "
             "or pass --allow-missing")
    _write_tsv(X, a.out)
    log.info("Pfam table: %d genomes x %d Pfams (%s)", len(X), X.shape[1] - 1, a.value)


def cmd_features(a):
    m = _read_manifest(a.manifest)
    X, ftype, stats = _features_for(m, a)
    _write_tsv(stats, Path(a.out).with_suffix(".stats.tsv"))
    if a.out.endswith(".tsv") or a.out.endswith(".tsv.gz"):
        _write_tsv(X, a.out, index=True)
    else:
        _die("--out must end with .tsv or .tsv.gz")


def cmd_train(a):
    import joblib

    from .models import task_labels, train_heads

    m = _read_manifest(a.manifest)
    cf = {}
    for spec in a.class_filter or []:
        task, sep, classes = spec.partition("=")
        names = [c.strip() for c in classes.split(",") if c.strip()]
        if not sep or not names:
            _die(f"--class-filter {spec!r}: expected TASK=CLASS,CLASS (e.g. gram_stain=negative,positive)")
        if task not in a.tasks:
            _die(f"--class-filter task {task!r} is not in --tasks {a.tasks}")
        cf[task] = names
    if a.oxygen_classes:
        if "oxygen" in cf:
            _die("give the oxygen filter once: --oxygen-classes or --class-filter oxygen=...")
        cf["oxygen"] = a.oxygen_classes
    for task, names in cf.items():
        unknown = [n for n in names if n not in set(task_labels(m, task))]
        if unknown:
            _die(f"--class-filter {task}: {unknown} not found among the manifest's {task} labels")
    cf = cf or None
    if a.cv_group not in m.columns:
        _die(f"--cv-group column {a.cv_group!r} not in manifest")
    X, ftype, _ = _features_for(m, a)
    bundle, report = train_heads(X, m, ftype, tasks=a.tasks, cv_group=a.cv_group, seed=a.seed,
                                 class_filter=cf, report_by=a.report_by)
    bundle["k"] = a.k
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "phenotype_model.joblib")
    report["manifest"] = str(a.manifest)
    _write_json(report, out / "training_report.json")
    log.info("saved %s", out / "phenotype_model.joblib")


def cmd_predict_genomes(a):
    import joblib

    from .models import predict_heads

    _need_file(a.model, "model")
    bundle = joblib.load(a.model)
    if bundle.get("kind") != "mss.phenotype_heads":
        _die(f"{a.model} is not an MSS phenotype model")
    if a.manifest:
        m = _read_manifest(a.manifest)
    elif a.fasta:
        m = pd.DataFrame({"genome_fasta": a.fasta})
        m["genome_id"] = [Path(p).name.split(".fna")[0].split(".fa")[0] for p in a.fasta]
        m["species"] = ""
    else:
        _die("give --manifest or --fasta")
    a.k = bundle.get("k", 5)
    X, _, _ = _features_for(m, a, bundle["feature_type"])
    pred = predict_heads(bundle, X).reset_index()
    keep = [c for c in ("genome_id", "species", "genus", "family") if c in m.columns]
    pred = m[keep].merge(pred, on="genome_id", how="inner")
    _write_tsv(pred, a.out)


def cmd_taxa(a):
    from .taxa import compute_taxa, compute_traits, read_profile

    _need_file(a.profile, "profile")
    _need_file(a.reference, "reference")
    prof = read_profile(a.profile)
    ref = pd.read_csv(a.reference, sep="\t", dtype=str, keep_default_na=False)
    pred = None
    if a.predictions:
        _need_file(a.predictions, "predictions")
        pred = pd.read_csv(a.predictions, sep="\t")
    samples, audit = compute_taxa(prof, ref, pred)
    _write_tsv(samples, a.out)
    stem = str(Path(a.out).with_suffix(""))
    _write_tsv(audit, a.audit or stem + ".per_taxon.tsv")
    if not a.oxygen_only:
        wide, taudit, done = compute_traits(prof, ref, pred, a.traits)
        _write_tsv(wide, stem + ".traits.tsv")
        _write_tsv(taudit, stem + ".traits_per_taxon.tsv")
        log.info("traits computed: %s", ", ".join(done))


def cmd_train_raw(a):
    import joblib

    from .models import task_labels
    from .raw import train_raw

    if a.mixtures < 10:
        _die("--mixtures must be >= 10")
    m = _read_manifest(a.manifest)
    X, ftype, _ = _features_for(m, a)
    labels = task_labels(m, "oxygen")
    groups = m.set_index("genome_id")["genus"] if "genus" in m.columns else None
    bundle, report = train_raw(X, labels, groups, a.mixtures, seed=a.seed, k=a.k,
                               max_members=a.max_members,
                               depth_range=(a.min_kmers, a.max_kmers))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "raw_mixture.joblib")
    _write_json(report, out / "raw_training_report.json")


def _samples_from_args(a):
    if a.sample_sheet:
        _need_file(a.sample_sheet, "sample sheet")
        ss = pd.read_csv(a.sample_sheet, sep="\t", dtype=str)
        if not {"sample_id", "fastq"} <= set(ss.columns):
            _die("sample sheet needs columns sample_id, fastq (comma-separate R1,R2)")
        rows = [(r.sample_id, r.fastq.split(",")) for r in ss.itertuples()]
        idx = a.array_index
        if idx == "env":
            idx = os.environ.get("SLURM_ARRAY_TASK_ID")
            if idx is None:
                _die("--array-index env but SLURM_ARRAY_TASK_ID unset")
        if idx is not None:
            i = int(idx)
            if not 0 <= i < len(rows):
                _die(f"array index {i} outside sample sheet (0..{len(rows) - 1})")
            rows = [rows[i]]
        return rows
    if not a.fastq:
        _die("give --fastq or --sample-sheet")
    sid = a.sample_id or Path(a.fastq[0]).name.split(".f")[0]
    return [(sid, a.fastq)]


def cmd_raw(a):
    import joblib

    from .features import fastq_kmers
    from .ontology import OXYGEN_CLASSES
    from .raw import STATUS, predict_raw

    _need_file(a.model, "model")
    bundle = joblib.load(a.model)
    if bundle.get("kind") != "mss.raw_mixture":
        _die(f"{a.model} is not an MSS-Raw model")
    rows = []
    for sid, files in _samples_from_args(a):
        for f in files:
            _need_file(f, "FASTQ")
        t0 = time.time()
        prof = fastq_kmers(files, k=bundle["k"], max_reads=a.max_reads)
        frac = predict_raw(bundle, prof.frequencies())[0]
        row = {"sample_id": sid, "n_reads": prof.n_records, "n_kmers": prof.n_kmers}
        for c, v in zip(OXYGEN_CLASSES, frac):
            row[f"{c}_pct"] = 100 * v
        row["mss_anaerobic_index"] = 100 * frac[0]
        row["model_id"] = bundle["model_id"]
        row["status"] = STATUS
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        log.info("%s: %d reads, anaerobic index %.1f (%s)", sid, prof.n_records, 100 * frac[0], STATUS)
    _write_tsv(pd.DataFrame(rows), a.out)


# ------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    P = argparse.ArgumentParser(prog="mss", description=(
        "MSS research prototype: oxygen-tolerance phenotypes from genomes, taxonomic profiles "
        "and (experimental) raw reads. Not clinically validated."))
    P.add_argument("--version", action="version", version=f"mss {__version__}")
    P.add_argument("-v", "--verbose", action="store_true")
    sub = P.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bacdive", help=f"{ONLINE} fetch BacDive records -> standardized reference TSV")
    p.add_argument("--ids", type=int, nargs="+", help="BacDive IDs")
    p.add_argument("--ids-file", help="file with BacDive IDs (whitespace separated)")
    p.add_argument("--taxon", nargs="+", help="taxon names, e.g. 'Bacteroides fragilis' or 'Clostridium'")
    p.add_argument("--taxa-file", help="file with one taxon name per line")
    p.add_argument("--id-range", type=int, nargs=2, metavar=("START", "END"),
                   help="scan every BacDive ID in this range, e.g. 1 180000 for all of BacDive "
                        "(resumable; IDs that do not exist are skipped)")
    p.add_argument("--cache-dir", default="data/bacdive_cache", help="raw JSON cache (default data/bacdive_cache)")
    p.add_argument("--oxygen-map", help="TSV raw<TAB>category to extend/override oxygen string mapping")
    p.add_argument("--out", default="data/bacdive_reference.tsv")
    p.set_defaults(func=cmd_bacdive)

    p = sub.add_parser("genome-manifest", help=f"{OFFLINE} reference TSV -> genome download manifest")
    p.add_argument("--reference", required=True)
    p.add_argument("--genome-dir", default="genomes")
    p.add_argument("--require-label", nargs="+", metavar="TRAIT",
                   help="keep only strains with a measured value for at least one of these traits, "
                        "e.g. oxygen gram_stain motility sporulation cell_shape")
    p.add_argument("--gtdb-metadata", nargs="+", metavar="TSV",
                   help="GTDB metadata, e.g. bac120_metadata_r226.tsv.gz ar53_metadata_r226.tsv.gz "
                        "(adds GTDB taxonomy, representative flag, CheckM2)")
    p.add_argument("--gtdb-reps-only", action="store_true",
                   help="keep only GTDB species-representative genomes (OxyMetaG strategy)")
    p.add_argument("--out", default="data/genome_manifest.tsv")
    p.set_defaults(func=cmd_genome_manifest)

    p = sub.add_parser("download-genomes", help=f"{ONLINE} NCBI Datasets download -> labeled_genomes.tsv")
    p.add_argument("--manifest", required=True)
    p.add_argument("--outdir", default="genomes")
    p.add_argument("--out", default="data/labeled_genomes.tsv")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--limit", type=int, help="only the first N rows (testing)")
    p.add_argument("--datasets-bin", default="datasets")
    p.add_argument("--workers", type=int, default=8, help="concurrent NCBI Datasets batch downloads")
    p.set_defaults(func=cmd_download_genomes)

    p = sub.add_parser("pfam-annotate", help=f"{OFFLINE} gene calls + Pfam (GA cutoffs) per genome, cached; "
                                             "splits across a SLURM array")
    p.add_argument("--manifest", required=True)
    p.add_argument("--hmm", required=True, help="Pfam-A.hmm or Pfam-A.hmm.gz")
    p.add_argument("--cache-dir", default=".mss_cache/pfam")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--task-id", type=int, help="this task's index (default $SLURM_ARRAY_TASK_ID or 0)")
    p.add_argument("--n-tasks", type=int, help="number of tasks (default $SLURM_ARRAY_TASK_COUNT or 1)")
    p.add_argument("--batch", type=int, default=20, help="genomes searched together per hmmsearch call")
    p.set_defaults(func=cmd_pfam_annotate)

    p = sub.add_parser("pfam-table", help=f"{OFFLINE} join Pfam caches into a genome x Pfam feature table")
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache-dir", default=".mss_cache/pfam")
    p.add_argument("--out", required=True, help="feature table (.tsv or .tsv.gz), usable as --embeddings")
    p.add_argument("--value", choices=["presence", "count"], default="presence")
    p.add_argument("--min-genomes", type=int, default=1, help="keep Pfams found in at least this many genomes")
    p.add_argument("--allow-missing", action="store_true", help="write the table even if some genomes "
                                                                 "are not annotated yet")
    p.set_defaults(func=cmd_pfam_table)

    p = sub.add_parser("features", help=f"{OFFLINE} precompute/cache genome k-mer features")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="feature table (.tsv or .tsv.gz)")
    _add_feature_args(p)
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train", help=f"{OFFLINE} train phenotype heads (oxygen, motility, sporulation, Gram, cell shape)")
    p.add_argument("--manifest", required=True, help="labeled genome manifest (genome_fasta + labels)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--tasks", nargs="+",
                   default=["oxygen", "motility", "sporulation", "gram_stain", "cell_shape"],
                   choices=["oxygen", "motility", "sporulation", "gram_stain", "cell_shape", "ph_range"],
                   help="phenotype heads to train (tasks without enough labels are skipped)")
    p.add_argument("--cv-group", default="genus",
                   help="manifest column for leave-group-out CV, e.g. genus, family, gtdb_genus, gtdb_family")
    p.add_argument("--report-by", default=None,
                   help="manifest column for per-stratum CV accuracy, e.g. gtdb_phylum")
    p.add_argument("--oxygen-classes", nargs="+", choices=["anaerobe", "facultative", "aerobe",
                   "microaerophile", "aerotolerant"],
                   help="train the oxygen head on these classes only, e.g. 'anaerobe aerobe' for an "
                        "OxyMetaG-style clear-label head (default: all classes)")
    p.add_argument("--class-filter", action="append", metavar="TASK=CLASS,CLASS",
                   help="train TASK only on genomes whose label is one of these classes; other labels are "
                        "left out, not merged (repeatable), e.g. gram_stain=negative,positive")
    p.add_argument("--seed", type=int, default=0)
    _add_feature_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict-genomes", help=f"{OFFLINE} phenotype probabilities for genomes")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest")
    p.add_argument("--fasta", nargs="+")
    p.add_argument("--out", default="results/genome_phenotype_predictions.tsv")
    _add_feature_args(p)
    p.set_defaults(func=cmd_predict_genomes)

    p = sub.add_parser("taxa", help=f"{OFFLINE} MSS-Taxa: profile + reference -> oxygen composition + multi-trait table")
    p.add_argument("--profile", required=True, help="long TSV: sample_id, species, relative_abundance")
    p.add_argument("--reference", required=True, help="phenotype reference (bacdive_reference.tsv format)")
    p.add_argument("--predictions", help="optional predict-genomes output used ONLY for species without a measured label")
    p.add_argument("--out", required=True)
    p.add_argument("--audit", help="per-taxon audit TSV (default <out>.per_taxon.tsv)")
    p.add_argument("--traits", nargs="+", help="traits for <out>.traits.tsv (default: every trait column "
                   "present in the reference: oxygen gram_stain motility sporulation cell_shape ph_range "
                   "optimal_temperature_c growth_temperature_min_c growth_temperature_max_c optimal_ph "
                   "growth_ph_min growth_ph_max optimal_nacl_pct growth_nacl_max_pct)")
    p.add_argument("--oxygen-only", action="store_true", help="skip the multi-trait table")
    p.set_defaults(func=cmd_taxa)

    p = sub.add_parser("train-raw", help=f"{OFFLINE} EXPERIMENTAL: train k-mer mixture regressor")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mixtures", type=int, default=2000)
    p.add_argument("--max-members", type=int, default=10, help="max genomes per synthetic mixture")
    p.add_argument("--min-kmers", type=float, default=2e4, help="min simulated k-mer sampling depth")
    p.add_argument("--max-kmers", type=float, default=2e6, help="max simulated k-mer sampling depth")
    p.add_argument("--seed", type=int, default=0)
    _add_feature_args(p)
    p.set_defaults(func=cmd_train_raw)

    p = sub.add_parser("raw", help=f"{OFFLINE} EXPERIMENTAL: FASTQ -> oxygen-category composition")
    p.add_argument("--model", required=True)
    p.add_argument("--fastq", nargs="+", help="FASTQ(.gz) files of ONE sample (e.g. R1 R2)")
    p.add_argument("--sample-id")
    p.add_argument("--sample-sheet", help="TSV sample_id, fastq (comma-separated files)")
    p.add_argument("--array-index", help="process one sample-sheet row (0-based) or 'env' for SLURM_ARRAY_TASK_ID")
    p.add_argument("--max-reads", type=int, default=None, help="stop after N reads per sample")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_raw)
    return P


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    t0 = time.time()
    args.func(args)
    log.info("%s finished in %.1fs", args.command, time.time() - t0)
    return 0
