#!/usr/bin/env python3
"""Обучение TiSASRec и сохранение артефактов."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_ROOT))
from common.gpu_fallback import run_torch_with_cpu_fallback

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    CUTOFF_MS,
    DROPOUT_RATE,
    EPOCHS,
    GRAD_CLIP,
    HIDDEN_UNITS,
    LR,
    MAXLEN,
    NUM_BLOCKS,
    NUM_HEADS,
    NUM_NEG,
    NUM_WORKERS,
    POLARS_MAX_THREADS,
    RANDOM_SEED,
    TIME_SPAN,
    TRAIN_MAX_STEPS,
    TRAIN_MAX_USERS,
    USE_TRAIN_FOR_SEQUENCES,
    USE_TRAIN_FOR_VOCAB,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    resolve_data_dir,
)
from data import (
    TiSASRecTrainDataset,
    build_vocab,
    iter_eval_users,
    load_contact_eids,
    load_user_sequences,
    save_sequences,
    verify_data_layout,
)
from model import TiSASRec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="datafest root (default: рядом с Tisasrec/ или DATA_DIR)",
    )
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _run(args: argparse.Namespace, device: torch.device) -> None:
    logging.info("Device: %s", device)

    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else resolve_data_dir()
    )
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    verify_data_layout(data_dir)
    logging.info("data_dir=%s cutoff_ms=%s", data_dir, CUTOFF_MS)

    contact = load_contact_eids(data_dir / "contact_eids.csv")
    logging.info("Building vocabulary (top %s items)…", args.vocab_size)
    vocab = build_vocab(data_dir, contact, vocab_size=args.vocab_size)
    vocab.save(artifacts_dir / "vocab.json")

    eval_users = iter_eval_users(data_dir)
    if TRAIN_MAX_USERS > 0:
        eval_users = eval_users.head(TRAIN_MAX_USERS)
        logging.info("Limited to %s users for training", eval_users.height)

    logging.info("Building user sequences with timestamps…")
    sequences = load_user_sequences(data_dir, vocab, eval_users)
    save_sequences(artifacts_dir / "sequences.npz", sequences)

    dataset = TiSASRecTrainDataset(
        sequences,
        num_items=vocab.size,
        maxlen=MAXLEN,
        time_span=TIME_SPAN,
        num_neg=NUM_NEG,
    )
    nw = NUM_WORKERS if NUM_WORKERS > 0 else 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        persistent_workers=nw > 0,
    )
    logging.info("Training samples: %s", len(dataset))

    model = TiSASRec(
        num_items=vocab.size,
        hidden_units=HIDDEN_UNITS,
        maxlen=MAXLEN,
        time_span=TIME_SPAN,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.98),
        weight_decay=WEIGHT_DECAY,
    )

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            seq, time_seq, time_matrix, pos, neg = [x.to(device) for x in batch]
            loss = model(seq, time_matrix, pos, neg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            global_step += 1
            if TRAIN_MAX_STEPS > 0 and global_step >= TRAIN_MAX_STEPS:
                break
        avg = epoch_loss / max(n_batches, 1)
        logging.info("Epoch %s/%s loss=%.4f batches=%s", epoch + 1, args.epochs, avg, n_batches)
        if TRAIN_MAX_STEPS > 0 and global_step >= TRAIN_MAX_STEPS:
            break

    model_path = artifacts_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_items": vocab.size,
            "hidden_units": HIDDEN_UNITS,
            "maxlen": MAXLEN,
            "time_span": TIME_SPAN,
            "num_blocks": NUM_BLOCKS,
            "num_heads": NUM_HEADS,
            "dropout_rate": DROPOUT_RATE,
        },
        model_path,
    )

    meta = {
        "data_dir": str(data_dir),
        "cutoff_ms": CUTOFF_MS,
        "use_train_vocab": USE_TRAIN_FOR_VOCAB,
        "use_train_sequences": USE_TRAIN_FOR_SEQUENCES,
        "vocab_size": vocab.size,
        "maxlen": MAXLEN,
        "time_span": TIME_SPAN,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_neg": NUM_NEG,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_train_samples": len(dataset),
        "n_users": len(sequences),
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("Saved model → %s", model_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.environ["POLARS_MAX_THREADS"] = str(POLARS_MAX_THREADS)
    set_seed(RANDOM_SEED)
    run_torch_with_cpu_fallback(
        lambda device: _run(args, device),
        cpu_only=args.cpu_only,
        label="Tisasrec/train",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
