"""Resume SCALE paper pipeline after feature extraction completes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def count_tsv_rows(tsv_path: Path) -> int:
    if not tsv_path.is_file():
        return 0
    with tsv_path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def latest_checkpoint(ckpt_dir: Path) -> Path:
    bins = list(ckpt_dir.glob("pytorch_model_*.bin"))
    if not bins:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return max(bins, key=lambda path: path.stat().st_mtime)


def feature_type_flags(feature_types: list[str]) -> list[str]:
    """Map feature type names to argparse flags for retrieval/eval scripts."""
    allowed = {
        "t", "p", "i", "v", "a",
        "tp", "ti", "tv", "pi", "pv", "iv", "ta", "pa", "ia", "va",
        "tpi", "tpv", "tiv", "piv", "tpiv", "tpiva", "dense",
    }
    flags: list[str] = []
    for name in feature_types:
        if name not in allowed:
            raise ValueError(f"Unknown feature type: {name!r} (allowed: {sorted(allowed)})")
        flags.append(f"--{name}")
    return flags


def ensure_product_catalog(work: Path, py: str, repo: Path, env: dict[str, str]) -> None:
    """Ensure id_label.json contains super_category for attribute reranking."""
    id_label_path = work / "id_label.json"
    if not id_label_path.is_file():
        return
    id_label = json.loads(id_label_path.read_text(encoding="utf-8"))
    sample = next(iter(id_label.values()), {})
    if sample.get("super_category"):
        return
    dataset_dir = repo / "app" / "datasets" / "downloaded_m5product_balanced"
    if not (dataset_dir / "metadata.json").is_file():
        print("Warning: super_category missing and dataset metadata not found; rerank may be weak.")
        return
    os_env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), **env}
    subprocess.check_call(
        [
            py,
            str(repo / "app" / "SCALE" / "preprocess" / "enrich_product_catalog.py"),
            "--id-label", str(id_label_path),
            "--dataset-dir", str(dataset_dir),
        ],
        env=os_env,
    )


def run_paper_retrieval_eval(
    py: str,
    scale: Path,
    work: Path,
    feature_types: list[str],
    env: dict[str, str],
    *,
    pipeline: str = "baseline",
    candidate_n: int = 100,
    rerank_lambda: float = 0.7,
    query_mode: str = "multimodal",
) -> dict:
    """Run retrieval + evaluate_unit_v2 (baseline or improved reranking)."""
    from indexing.pipeline_config import PipelineMode, PipelinePaths

    mode = PipelineMode.from_value(pipeline)
    paths = PipelinePaths(work, mode)
    os_env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), **env}
    query_dir = work / "features" / "test"
    gallery_dir = work / "features" / "gallery"
    retrieval_dir = paths.retrieval_results
    metric_dir = paths.retrieval_metric
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    type_flags = feature_type_flags(feature_types)
    if mode is PipelineMode.IMPROVED:
        repo = scale.parent.parent
        ensure_product_catalog(work, py, repo, env)
        app_dir = scale.parent
        for feature_type in feature_types:
            subprocess.check_call(
                [
                    py,
                    str(app_dir / "evaluation" / "run_improved_retrieval.py"),
                    "--query-feature-path", str(query_dir),
                    "--gallery-feature-path", str(gallery_dir),
                    "--catalog", str(work / "id_label.json"),
                    "--retrieval-results-path", str(retrieval_dir),
                    "--feature-type", feature_type,
                    "--candidate-n", str(candidate_n),
                    "--output-topk", "10",
                    "--lambda-emb", str(rerank_lambda),
                    "--query-mode", query_mode,
                ],
                cwd=str(app_dir),
                env=os_env,
            )
    else:
        subprocess.check_call(
            [
                py,
                str(scale / "retrieval_unit_id_list_v2.py"),
                "--query_feature_path", str(query_dir),
                "--gallery_feature_path", str(gallery_dir),
                "--retrieval_results_path", str(retrieval_dir),
                "--max_topk", "10",
                *type_flags,
            ],
            cwd=str(scale.parent),
            env=os_env,
        )

    subprocess.check_call(
        [
            py,
            str(scale / "evaluate_unit_v2.py"),
            "--retrieval_result_dir", str(retrieval_dir),
            "--GT_file", str(work / "id_label.json"),
            "--gallery-ids", str(gallery_dir / "id.npy"),
            "--output_metric_dir", str(metric_dir),
            *type_flags,
        ],
        cwd=str(scale.parent),
        env=os_env,
    )

    metric_path = metric_dir / "metric_results.json"
    results = json.loads(metric_path.read_text(encoding="utf-8"))
    benchmark_path = paths.evaluation_benchmark
    benchmark_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Paper benchmark metrics ({mode.value}): {benchmark_path}")
    for feature_type in feature_types:
        if feature_type not in results:
            continue
        print(f"  [{feature_type}]")
        for topk in ("top1", "top5", "top10"):
            if topk in results[feature_type]:
                row = results[feature_type][topk]
                print(
                    f"    {topk}: mAP={row['mAP']:.2f}  "
                    f"Prec={row['Prec']:.2f}  HitRate={row['mHitRate']:.2f}"
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/scale_paper"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-tsv-rows", type=int, default=9000)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Effective train batch size (micro-batch = batch_size // grad_accum; need micro-batch >= 2 for CLR)")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use Apex FP16 (requires nvidia-apex; unavailable on most Windows setups)")
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip waiting for region TSV extraction to finish",
    )
    parser.add_argument(
        "--skip-lmdb",
        action="store_true",
        help="Skip rebuilding LMDB splits (use when already built)",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretrain and use latest checkpoint in checkpoints/<save_name>/",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip feature extraction (use existing features/test and features/gallery)",
    )
    parser.add_argument(
        "--eval-feature-types",
        nargs="+",
        default=["tpiva"],
        help="Feature types for paper retrieval eval (default: tpiva = 5 modalities)",
    )
    parser.add_argument(
        "--pipeline",
        choices=("baseline", "improved"),
        default="baseline",
        help="Retrieval pipeline: baseline (embedding only) or improved (HNSW coarse + attribute rerank)",
    )
    parser.add_argument(
        "--candidate-n",
        type=int,
        default=100,
        help="Coarse candidate count N for improved pipeline (stage 1)",
    )
    parser.add_argument(
        "--rerank-lambda",
        type=float,
        default=0.7,
        help="Blend weight lambda for S_tong = lambda*S_emb + (1-lambda)*S_attr",
    )
    parser.add_argument(
        "--query-mode",
        choices=("multimodal", "image_only"),
        default="multimodal",
        help="Improved eval query mode: multimodal (scenario A) or image_only (scenario B)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    work = (repo / args.work_dir).resolve()
    scale = repo / "app" / "SCALE"
    py = args.python
    env = {"PYTHONPATH": f"{scale};{repo / 'app'}", "PYTHONUNBUFFERED": "1"}

    tsv = work / "tsv_features" / "features.tsv"
    if not args.skip_wait:
        while count_tsv_rows(tsv) < args.min_tsv_rows:
            rows = count_tsv_rows(tsv)
            print(f"Waiting for region TSV: {rows}/{args.min_tsv_rows}")
            time.sleep(60)

        stable_checks = 0
        last_rows = count_tsv_rows(tsv)
        while stable_checks < 3:
            time.sleep(120)
            rows = count_tsv_rows(tsv)
            if rows == last_rows:
                stable_checks += 1
            else:
                stable_checks = 0
                last_rows = rows
            print(f"Waiting for region extraction to finish: {rows} rows (stable {stable_checks}/3)")
    else:
        rows = count_tsv_rows(tsv)
        print(f"Skipping TSV wait; using {rows} rows in {tsv}")

    if not args.skip_lmdb:
        for split in ("train", "test", "gallery"):
            subprocess.check_call(
                [
                    py,
                    str(scale / "tools" / "bp_feature" / "convert" / "convert_from_config.py"),
                    "--tsv-dir", str(work / "tsv_features"),
                    "--output-lmdb", str(work / "lmdb_features" / f"{split}_feature.lmdb"),
                    "--ids-file", str(work / f"{split}_ids.json"),
                    "--tsv-name", "features.tsv",
                    "--map-size-gb", "8",
                ],
                env={**dict(**{k: v for k, v in __import__("os").environ.items()}), **env},
                cwd=str(repo),
            )
    else:
        print("Skipping LMDB rebuild")

    examples = scale / "examples" / "SCALE"
    pretrain_cmd = [
        py, "-u", "pretrain_task.py",
        "--from_pretrained", "bert-base-chinese",
        "--bert_model", "bert-base-chinese",
        "--config_file", "..\\..\\config\\bert_base_6layer_6conect_capture_itp3va.json",
        "--predict_feature",
        "--learning_rate", "1e-4",
        "--video_feature_dir", str(work / "video_feature"),
        "--audio_feature_dir", str(work / "audio_feature"),
        "--lmdb_file", str(work / "lmdb_features" / "train_feature.lmdb"),
        "--caption_path", str(work / "id_label.json"),
        "--output_dir", str(work / "checkpoints"),
        "--save_name", "scale_paper_simcl",
        "--train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--num_train_epochs", str(args.train_epochs),
        "--max_seq_length", "36",
        "--video_len", "12",
        "--pv_seq_len", "64",
        "--audio_len", "12",
        "--num_workers", "0",
        "--MLM", "--MRM", "--MEM", "--MFM", "--MAM", "--CLR",
    ]
    if not args.skip_pretrain:
        if args.fp16:
            pretrain_cmd.append("--fp16")
        subprocess.check_call(
            pretrain_cmd,
            cwd=str(examples),
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}), **env},
        )
    else:
        print("Skipping pretrain; using latest checkpoint")
    ckpt = latest_checkpoint(work / "checkpoints" / "scale_paper_simcl")
    print(f"Using checkpoint: {ckpt}")
    if not args.skip_extract:
        for split in ("test", "gallery"):
            feat_dir = work / "features" / split
            feat_dir.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                [
                    py, "eval/extract_features.py",
                    "--from_pretrained", str(ckpt),
                    "--bert_model", "bert-base-chinese",
                    "--config_file", "..\\..\\config\\bert_base_6layer_6conect_capture_itp3va.json",
                    "--predict_feature",
                    "--lmdb_file", str(work / "lmdb_features" / f"{split}_feature.lmdb"),
                    "--caption_path", str(work / "id_label.json"),
                    "--video_feature_dir", str(work / "video_feature"),
                    "--audio_feature_dir", str(work / "audio_feature"),
                    "--feature_dir", str(feat_dir),
                    "--train_batch_size", "8",
                    "--max_seq_length", "36",
                    "--video_len", "12",
                    "--pv_seq_len", "64",
                    "--audio_len", "12",
                    "--num_workers", "0",
                    "--split", split,
                ],
                cwd=str(examples),
                env={**dict(**{k: v for k, v in __import__("os").environ.items()}), **env},
            )
    else:
        print("Skipping feature extraction")

    run_paper_retrieval_eval(
        py,
        scale,
        work,
        args.eval_feature_types,
        env,
        pipeline=args.pipeline,
        candidate_n=args.candidate_n,
        rerank_lambda=args.rerank_lambda,
        query_mode=args.query_mode,
    )

    from indexing.pipeline_config import PipelinePaths, PipelineMode

    paths = PipelinePaths(work, PipelineMode.from_value(args.pipeline))
    subprocess.check_call(
        [
            py, str(repo / "app" / "indexing" / "build_index.py"),
            "--embeddings", str(work / "features" / "gallery" / "tpiva_feature_np.npy"),
            "--ids", str(work / "features" / "gallery" / "id.npy"),
            "--output-dir", str(paths.index_dir),
            "--index-type", "hnsw",
        ],
        cwd=str(repo),
    )
    print(f"Pipeline resume complete ({args.pipeline}).")
    print(f"  Benchmark: {paths.evaluation_benchmark}")


if __name__ == "__main__":
    main()
