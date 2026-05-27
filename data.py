import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from config import (
    ALPHA_CONTACT,
    ALPHA_VIEW,
    CUTOFF_MS,
    MAXLEN,
    RANDOM_SEED,
    SUBMISSION_K,
    TIME_SPAN,
    USE_TRAIN_FOR_SEQUENCES,
    VOCAB_SIZE,
)

logger = logging.getLogger(__name__)

UserSequence = tuple[list[int], list[int]]


@dataclass(frozen=True)
class Vocab:
    item2idx: dict[int, int]
    idx2item: list[int]
    popular_items: list[int]

    @property
    def size(self) -> int:
        return len(self.idx2item) - 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "item2idx": {str(k): v for k, v in self.item2idx.items()},
            "idx2item": self.idx2item,
            "popular_items": self.popular_items,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        payload = json.loads(path.read_text(encoding="utf-8"))
        item2idx = {int(k): int(v) for k, v in payload["item2idx"].items()}
        return cls(
            item2idx=item2idx,
            idx2item=[int(x) for x in payload["idx2item"]],
            popular_items=[int(x) for x in payload["popular_items"]],
        )


def eval_user_events_path(data_dir: Path) -> Path:
    for name in ("eval_user_events.pq", "eval_user_events.parquet"):
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"eval_user_events not found in {data_dir}")


def load_contact_eids(path: Path) -> set[int]:
    df = pl.read_csv(path)
    col = "mapped_eid" if "mapped_eid" in df.columns else df.columns[0]
    return {int(x) for x in df[col].to_list()}


def train_parquet_paths(
    data_dir: Path,
    *,
    user_ids: pl.Series | None = None,
) -> list[Path]:
    """Все шарды train_data или только part_{user_id % 100} для заданных users."""
    d = data_dir / "train_data"
    if not d.is_dir():
        raise FileNotFoundError(f"train_data/ not found in {data_dir}")

    if user_ids is not None and user_ids.len() > 0:
        from config import TRAIN_SHARD_MOD

        mods = {int(u) % TRAIN_SHARD_MOD for u in user_ids.to_list()}
        paths: list[Path] = []
        for m in sorted(mods):
            found: Path | None = None
            for fmt in (f"part_{m:03d}.parquet", f"part_{m:02d}.parquet", f"part_{m}.parquet"):
                p = d / fmt
                if p.is_file():
                    found = p
                    break
            if found is not None:
                paths.append(found)
        if paths:
            logger.info(
                "train_data: %s/%s shards for eval users (user_id %% %s)",
                len(paths),
                TRAIN_SHARD_MOD,
                TRAIN_SHARD_MOD,
            )
            return paths

    paths = sorted(d.glob("part_*.parquet"))
    if not paths:
        paths = sorted(d.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet in {d}")
    return paths


def verify_data_layout(data_dir: Path) -> None:
    """Проверка ожидаемых файлов datafest (лог, без падения)."""
    required = ("eval_users.csv", "contact_eids.csv")
    optional = ("item_features.parquet", "eval_user_events.pq", "eval_user_events.parquet")
    for name in required:
        ok = (data_dir / name).is_file()
        logger.info("data %s: %s", name, "OK" if ok else "MISSING")
    for name in optional:
        if (data_dir / name).is_file():
            logger.info("data %s: OK", name)
    try:
        n_shards = len(train_parquet_paths(data_dir))
        logger.info("data train_data: %s parquet shard(s)", n_shards)
    except FileNotFoundError as exc:
        logger.warning("data train_data: %s", exc)
    try:
        eval_user_events_path(data_dir)
        logger.info("data eval_user_events: OK")
    except FileNotFoundError:
        logger.warning("data eval_user_events: MISSING")


def build_item_counts(
    data_dir: Path,
    contact_eids: set[int],
    *,
    use_train: bool = True,
) -> pl.DataFrame:
    ev_path = eval_user_events_path(data_dir)
    contact_list = list(contact_eids)

    ev = (
        pl.scan_parquet(str(ev_path))
        .filter(pl.col("timestamp") < CUTOFF_MS)
        .select(["item_id", "eid"])
    )
    ev_counts = (
        ev.with_columns(
            pl.when(pl.col("eid").is_in(contact_list))
            .then(pl.lit(ALPHA_CONTACT))
            .otherwise(pl.lit(ALPHA_VIEW))
            .alias("w")
        )
        .group_by("item_id")
        .agg(pl.col("w").sum().alias("cnt"))
        .collect(streaming=True)
    )

    if not use_train:
        return ev_counts.sort("cnt", descending=True)

    parts: list[pl.DataFrame] = [ev_counts]
    for path in train_parquet_paths(data_dir):
        chunk = (
            pl.scan_parquet(str(path))
            .filter(pl.col("timestamp") < CUTOFF_MS)
            .select(["item_id", "eid"])
            .with_columns(
                pl.when(pl.col("eid").is_in(contact_list))
                .then(pl.lit(ALPHA_CONTACT))
                .otherwise(pl.lit(ALPHA_VIEW))
                .alias("w")
            )
            .group_by("item_id")
            .agg(pl.col("w").sum().alias("cnt"))
            .collect(streaming=True)
        )
        if not chunk.is_empty():
            parts.append(chunk)

    merged = (
        pl.concat(parts, how="vertical")
        .group_by("item_id")
        .agg(pl.col("cnt").sum())
        .sort("cnt", descending=True)
    )
    return merged


def build_vocab(
    data_dir: Path,
    contact_eids: set[int],
    vocab_size: int = VOCAB_SIZE,
    *,
    use_train: bool | None = None,
) -> Vocab:
    if use_train is None:
        from config import USE_TRAIN_FOR_VOCAB

        use_train = USE_TRAIN_FOR_VOCAB
    counts = build_item_counts(data_dir, contact_eids, use_train=use_train)
    top = counts.head(vocab_size)
    idx2item = [0]
    item2idx: dict[int, int] = {0: 0}
    for row in top.iter_rows():
        item_id = int(row[0])
        item2idx[item_id] = len(idx2item)
        idx2item.append(item_id)
    popular = [int(x) for x in top["item_id"].head(SUBMISSION_K * 4).to_list()]
    logger.info("Vocab size=%s (incl. padding)", len(idx2item) - 1)
    return Vocab(item2idx=item2idx, idx2item=idx2item, popular_items=popular)


def normalize_user_times(timestamps: list[int]) -> list[int]:
    """Per-user time normalization (TiSASRec util.cleanAndsort)."""
    if not timestamps:
        return []
    t_min = min(timestamps)
    diffs = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
        if timestamps[i + 1] != timestamps[i]
    ]
    time_scale = min(diffs) if diffs else 1
    if time_scale == 0:
        time_scale = 1
    return [int(round((t - t_min) / time_scale) + 1) for t in timestamps]


def compute_time_matrix(
    time_seq: list[int],
    time_span: int = TIME_SPAN,
    maxlen: int = MAXLEN,
) -> np.ndarray:
    """Pairwise discretized time intervals, capped at time_span."""
    mat = np.zeros((maxlen, maxlen), dtype=np.int64)
    n = min(len(time_seq), maxlen)
    for i in range(n):
        for j in range(n):
            ti, tj = time_seq[i], time_seq[j]
            if ti == 0 or tj == 0:
                continue
            mat[i, j] = min(abs(int(ti) - int(tj)), time_span)
    return mat


def _map_sequence_with_times(
    items: list[int],
    timestamps: list[int],
    vocab: Vocab,
) -> UserSequence:
    norm_times = normalize_user_times(timestamps)
    out_items: list[int] = []
    out_times: list[int] = []
    for item_id, ts in zip(items, norm_times):
        idx = vocab.item2idx.get(int(item_id))
        if idx is not None and idx > 0:
            out_items.append(idx)
            out_times.append(int(ts))
    return out_items, out_times


def events_lazy_for_eval_users(
    data_dir: Path,
    eval_users: pl.DataFrame,
    *,
    use_train: bool | None = None,
) -> pl.LazyFrame:
    """eval_user_events + train_data (до cutoff) для пользователей из eval_users."""
    if use_train is None:
        use_train = USE_TRAIN_FOR_SEQUENCES

    users = eval_users.select(pl.col("user_id").cast(pl.UInt32).unique())
    parts: list[pl.LazyFrame] = [
        pl.scan_parquet(str(eval_user_events_path(data_dir)))
        .filter(pl.col("timestamp") < CUTOFF_MS)
        .select(
            pl.col("user_id").cast(pl.UInt32),
            pl.col("item_id").cast(pl.UInt32),
            pl.col("timestamp"),
        ),
    ]
    if use_train:
        user_ids = users.collect()["user_id"]
        for path in train_parquet_paths(data_dir, user_ids=user_ids):
            parts.append(
                pl.scan_parquet(str(path))
                .filter(pl.col("timestamp") < CUTOFF_MS)
                .select(
                    pl.col("user_id").cast(pl.UInt32),
                    pl.col("item_id").cast(pl.UInt32),
                    pl.col("timestamp"),
                ),
            )
        logger.info(
            "Sequences: eval_user_events + %s train_data shards (cutoff_ms=%s)",
            len(parts) - 1,
            CUTOFF_MS,
        )
    else:
        logger.info("Sequences: eval_user_events only (cutoff_ms=%s)", CUTOFF_MS)

    lf = (
        pl.concat(parts, how="vertical")
        .join(users.lazy(), on="user_id", how="inner")
        .unique(subset=["user_id", "item_id", "timestamp"], keep="first")
        .sort(["user_id", "timestamp"])
    )
    return lf


def load_user_sequences(
    data_dir: Path,
    vocab: Vocab,
    eval_users: pl.DataFrame | None = None,
    *,
    use_train: bool | None = None,
) -> dict[int, UserSequence]:
    """User -> (mapped item indices, normalized time indices)."""
    if eval_users is None:
        eval_users = iter_eval_users(data_dir)
    df = events_lazy_for_eval_users(
        data_dir, eval_users, use_train=use_train,
    ).collect(streaming=True)
    sequences: dict[int, UserSequence] = {}
    for grp in df.partition_by("user_id", maintain_order=True):
        uid = int(grp["user_id"][0])
        raw_items = [int(x) for x in grp["item_id"].to_list()]
        raw_ts = [int(x) for x in grp["timestamp"].to_list()]
        items, times = _map_sequence_with_times(raw_items, raw_ts, vocab)
        if items:
            sequences[uid] = (items[-MAXLEN:], times[-MAXLEN:])
    logger.info("Loaded sequences for %s users", len(sequences))
    return sequences


def save_sequences(path: Path, sequences: dict[int, UserSequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    users = np.array(list(sequences.keys()), dtype=np.uint32)
    lengths = np.array([len(sequences[int(u)][0]) for u in users], dtype=np.int32)
    items_flat: list[int] = []
    times_flat: list[int] = []
    for u in users:
        items, times = sequences[int(u)]
        items_flat.extend(items)
        times_flat.extend(times)
    np.savez_compressed(
        path,
        users=users,
        lengths=lengths,
        items=np.array(items_flat, dtype=np.int32),
        times=np.array(times_flat, dtype=np.int32),
    )


def load_sequences(path: Path) -> dict[int, UserSequence]:
    data = np.load(path)
    users = data["users"]
    lengths = data["lengths"]
    items_flat = data["items"]
    times_flat = data["times"]
    sequences: dict[int, UserSequence] = {}
    offset = 0
    for uid, ln in zip(users, lengths):
        ln = int(ln)
        sequences[int(uid)] = (
            items_flat[offset : offset + ln].tolist(),
            times_flat[offset : offset + ln].tolist(),
        )
        offset += ln
    return sequences


def _build_padded_sample(
    prefix_items: list[int],
    prefix_times: list[int],
    pos_item: int,
    pos_time: int,
    maxlen: int,
) -> tuple[list[int], list[int], list[int]]:
    full_items = (prefix_items + [pos_item])[-maxlen:]
    full_times = (prefix_times + [pos_time])[-maxlen:]
    n = len(full_items)
    pad = maxlen - n
    seq = [0] * pad + full_items[:-1]
    time_seq = [0] * pad + full_times[:-1]
    if len(seq) < maxlen:
        seq.append(0)
        time_seq.append(0)
    pos = [0] * pad + full_items
    return seq, time_seq, pos


class TiSASRecTrainDataset(Dataset):
    """Leave-one-out samples with time matrix for TiSASRec."""

    def __init__(
        self,
        sequences: dict[int, UserSequence],
        num_items: int,
        maxlen: int = MAXLEN,
        time_span: int = TIME_SPAN,
        num_neg: int = 1,
        seed: int = RANDOM_SEED,
        max_positions_per_user: int | None = None,
    ) -> None:
        from config import MAX_POSITIONS_PER_USER

        if max_positions_per_user is None:
            max_positions_per_user = MAX_POSITIONS_PER_USER
        self.maxlen = maxlen
        self.time_span = time_span
        self.num_items = num_items
        self.num_neg = num_neg
        self.rng = random.Random(seed)
        self.samples: list[tuple[list[int], list[int], int, int]] = []
        for items, times in sequences.values():
            if len(items) < 2:
                continue
            if max_positions_per_user <= 0:
                start = 1
            else:
                start = max(1, len(items) - max_positions_per_user)
            for t in range(start, len(items)):
                self.samples.append((items[:t], times[:t], items[t], times[t]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_items, prefix_times, pos_item, pos_time = self.samples[idx]
        seq, time_seq, pos = _build_padded_sample(
            prefix_items, prefix_times, pos_item, pos_time, self.maxlen
        )
        time_matrix = compute_time_matrix(time_seq, self.time_span, self.maxlen)

        neg: list[int] = []
        for p in pos:
            if p == 0:
                neg.append(0)
                continue
            n = self.rng.randint(1, self.num_items)
            while n == p:
                n = self.rng.randint(1, self.num_items)
            neg.append(n)

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(time_seq, dtype=torch.long),
            torch.tensor(time_matrix, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def iter_eval_users(data_dir: Path) -> pl.DataFrame:
    return pl.read_csv(data_dir / "eval_users.csv").select(
        pl.col("user_id").cast(pl.UInt32).unique()
    )


def build_seen_items(
    data_dir: Path,
    eval_users: pl.DataFrame,
    *,
    use_train: bool | None = None,
) -> dict[int, set[int]]:
    df = (
        events_lazy_for_eval_users(data_dir, eval_users, use_train=use_train)
        .select(["user_id", "item_id"])
        .unique()
        .collect(streaming=True)
    )
    seen: dict[int, set[int]] = {}
    for row in df.iter_rows():
        uid, iid = int(row[0]), int(row[1])
        seen.setdefault(uid, set()).add(iid)
    return seen


def pad_submission(
    rows: list[tuple[int, int]],
    eval_users: pl.DataFrame,
    popular_items: list[int],
    k: int = 160,
) -> pl.DataFrame:
    from config import SUBMISSION_K

    k = k or SUBMISSION_K
    by_user: dict[int, list[int]] = {}
    for uid, iid in rows:
        by_user.setdefault(uid, []).append(iid)

    out_rows: list[tuple[int, int]] = []
    for uid in eval_users["user_id"].to_list():
        uid = int(uid)
        taken = set(by_user.get(uid, []))
        items = list(by_user.get(uid, []))
        for iid in popular_items:
            if len(items) >= k:
                break
            if iid not in taken:
                items.append(iid)
                taken.add(iid)
        for iid in items[:k]:
            out_rows.append((uid, iid))

    return pl.DataFrame(out_rows, schema={"user_id": pl.UInt32, "item_id": pl.UInt32}, orient="row")


def pad_sequence_pair(items: list[int], times: list[int], maxlen: int) -> tuple[list[int], list[int]]:
    items = items[-maxlen:]
    times = times[-maxlen:]
    pad = maxlen - len(items)
    return [0] * pad + items, [0] * pad + times
