#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_ROOT))
from common.gpu_fallback import run_torch_with_cpu_fallback

import numpy as np
import polars as pl
import torch

from config import (
    ARTIFACTS_DIR,
    MAXLEN,
    PRED_MAX_USERS,
    PREDICT_BATCH,
    POLARS_MAX_THREADS,
    SCORE_CHUNK,
    SUBMISSION_K,
    TIME_SPAN,
    resolve_data_dir,
)
from data import (
    Vocab,
    build_seen_items,
    compute_time_matrix,
    iter_eval_users,
    load_sequences,
    load_user_sequences,
    pad_sequence_pair,
    pad_submission,
    verify_data_layout,
)
from model import TiSASRec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="datafest root (default: рядом с Tisasrec/ или DATA_DIR)",
    )
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def load_model(artifacts_dir: Path, device: torch.device) -> TiSASRec:
    ckpt = torch.load(artifacts_dir / "model.pt", map_location=device, weights_only=False)
    model = TiSASRec(
        num_items=int(ckpt["num_items"]),
        hidden_units=int(ckpt["hidden_units"]),
        maxlen=int(ckpt["maxlen"]),
        time_span=int(ckpt["time_span"]),
        num_blocks=int(ckpt["num_blocks"]),
        num_heads=int(ckpt["num_heads"]),
        dropout_rate=float(ckpt["dropout_rate"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def score_users(
    model: TiSASRec,
    user_ids: list[int],
    sequences: dict[int, tuple[list[int], list[int]]],
    vocab: Vocab,
    seen: dict[int, set[int]],
    device: torch.device,
    time_span: int = TIME_SPAN,
    k: int = SUBMISSION_K,
) -> list[tuple[int, int]]:
    all_idx = torch.arange(1, vocab.size + 1, device=device, dtype=torch.long)
    all_items = [vocab.idx2item[i] for i in range(1, vocab.size + 1)]
    rows: list[tuple[int, int]] = []

    for start in range(0, len(user_ids), PREDICT_BATCH):
        batch_uids = user_ids[start : start + PREDICT_BATCH]
        seq_tensors: list[list[int]] = []
        time_matrices: list[np.ndarray] = []
        for uid in batch_uids:
            items, times = sequences.get(uid, ([], []))
            seq, time_seq = pad_sequence_pair(items, times, MAXLEN)
            seq_tensors.append(seq)
            time_matrices.append(compute_time_matrix(time_seq, time_span, MAXLEN))

        seq_batch = torch.tensor(seq_tensors, dtype=torch.long, device=device)
        time_batch = torch.tensor(np.stack(time_matrices), dtype=torch.long, device=device)

        user_scores = np.full((len(batch_uids), vocab.size), -np.inf, dtype=np.float32)
        for c_start in range(0, len(all_idx), SCORE_CHUNK):
            c_end = min(c_start + SCORE_CHUNK, len(all_idx))
            cand = all_idx[c_start:c_end].unsqueeze(0).expand(len(batch_uids), -1)
            chunk_scores = model.predict_next(seq_batch, time_batch, cand).cpu().numpy()
            user_scores[:, c_start:c_end] = chunk_scores

        for i, uid in enumerate(batch_uids):
            seen_raw = seen.get(uid, set())
            ranked_idx = np.argsort(-user_scores[i])
            picked: list[int] = []
            for j in ranked_idx:
                raw_item = all_items[j]
                if raw_item in seen_raw:
                    continue
                picked.append(raw_item)
                if len(picked) >= k:
                    break
            rows.extend((uid, iid) for iid in picked)

    return rows


def _run(args: argparse.Namespace, device: torch.device) -> None:
    logging.info("Device: %s", device)

    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else resolve_data_dir()
    )
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    out_csv = args.out.expanduser().resolve()

    verify_data_layout(data_dir)
    logging.info("data_dir=%s", data_dir)

    vocab = Vocab.load(artifacts_dir / "vocab.json")
    model = load_model(artifacts_dir, device)
    time_span = model.time_span

    eval_users = iter_eval_users(data_dir)
    if PRED_MAX_USERS > 0:
        eval_users = eval_users.head(PRED_MAX_USERS)
        logging.info("Limited predict to %s users", eval_users.height)
    user_ids = [int(x) for x in eval_users["user_id"].to_list()]

    seq_path = artifacts_dir / "sequences.npz"
    if seq_path.exists():
        logging.info("Loading cached sequences from %s", seq_path)
        sequences = load_sequences(seq_path)
        missing = [u for u in user_ids if u not in sequences]
        if missing:
            logging.info("Rebuilding sequences for %s users missing from cache", len(missing))
            extra_users = pl.DataFrame({"user_id": missing}, schema={"user_id": pl.UInt32})
            sequences.update(load_user_sequences(data_dir, vocab, extra_users))
    else:
        logging.info("Building sequences from eval events…")
        sequences = load_user_sequences(data_dir, vocab, eval_users)

    logging.info("Building seen-item sets…")
    seen = build_seen_items(data_dir, eval_users)

    logging.info("Scoring %s users…", len(user_ids))
    rows = score_users(
        model, user_ids, sequences, vocab, seen, device, time_span=time_span, k=SUBMISSION_K
    )

    out = pad_submission(rows, eval_users, vocab.popular_items, k=SUBMISSION_K)
    out.write_csv(out_csv)
    n_users = out["user_id"].n_unique()
    logging.info(
        "Wrote %s rows, %s users → %s",
        out.height,
        n_users,
        out_csv,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.environ["POLARS_MAX_THREADS"] = str(POLARS_MAX_THREADS)
    run_torch_with_cpu_fallback(
        lambda device: _run(args, device),
        cpu_only=args.cpu_only,
        label="Tisasrec/predict",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
