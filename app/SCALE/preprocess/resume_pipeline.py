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
    bins = sorted(
        ckpt_dir.glob("pytorch_model_*.bin"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not bins:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return bins[-1]


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
    if args.fp16:
        pretrain_cmd.append("--fp16")
    subprocess.check_call(
        pretrain_cmd,
        cwd=str(examples),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), **env},
    )
    ckpt = latest_checkpoint(work / "checkpoints" / "scale_paper_simcl")
    print(f"Using checkpoint: {ckpt}")
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

    subprocess.check_call(
        [
            py, str(scale / "preprocess" / "run_retrieval_eval.py"),
            "--query-features", str(work / "features" / "test" / "tpiva_feature_np.npy"),
            "--query-ids", str(work / "features" / "test" / "id.npy"),
            "--gallery-features", str(work / "features" / "gallery" / "tpiva_feature_np.npy"),
            "--gallery-ids", str(work / "features" / "gallery" / "id.npy"),
            "--id-label", str(work / "id_label.json"),
            "--output", str(work / "evaluation.json"),
        ],
        cwd=str(repo),
    )

    subprocess.check_call(
        [
            py, str(repo / "app" / "indexing" / "build_index.py"),
            "--embeddings", str(work / "features" / "gallery" / "tpiva_feature_np.npy"),
            "--ids", str(work / "features" / "gallery" / "id.npy"),
            "--output-dir", str(work / "index_hnsw"),
            "--index-type", "hnsw",
        ],
        cwd=str(repo),
    )
    print("Pipeline resume complete.")


if __name__ == "__main__":
    main()
