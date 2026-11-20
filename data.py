"""Read a five-minute Borg job panel and align it to a common UTC clock."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RESOURCES = ("cpu", "memory")
WINDOW_MINUTES = 5
HOUR = 60 // WINDOW_MINUTES
DAY = 24 * HOUR
WINDOW = pd.Timedelta(minutes=WINDOW_MINUTES)
AVERAGES = ["cpu_avg", "memory_avg"]
PEAKS = ["cpu_peak", "memory_peak"]
REQUESTS = ["cpu_request", "memory_request"]
METADATA = REQUESTS + ["priority", "scheduling_class"]
REQUIRED = ["job_id", "timestamp", *AVERAGES, *PEAKS, *REQUESTS]
OPTIONAL = ["priority", "scheduling_class", "task_count", "sample_rank"]


@dataclass(frozen=True)
class Split:
    start: pd.Timestamp
    train_days: int
    tune_days: int
    test_days: int

    def __post_init__(self):
        start = pd.to_datetime(self.start, utc=True)
        if pd.isna(start) or start != start.floor("h"):
            raise ValueError("The experiment start must be aligned to a UTC hour.")
        if min(self.train_days, self.tune_days, self.test_days) < 1:
            raise ValueError("Every split must be at least one day long.")
        object.__setattr__(self, "start", start)

    @property
    def train_end(self):
        return self.train_days * DAY

    @property
    def tune_end(self):
        return self.train_end + self.tune_days * DAY

    @property
    def end(self):
        return self.tune_end + self.test_days * DAY

    def timestamp(self, step):
        return self.start + int(step) * WINDOW


@dataclass
class Job:
    job_id: str
    index: pd.DatetimeIndex
    average: np.ndarray
    peak: np.ndarray
    task_count: np.ndarray
    request: np.ndarray
    priority: float
    scheduling_class: float
    observed: np.ndarray
    filled_peak: np.ndarray
    first_observation: int
    complete_hours: np.ndarray
    ready_hours: np.ndarray

    def steps(self, start, end, complete_only=False):
        """Return hourly origins with at least one hour of observed history."""
        hours = np.arange(start // HOUR, end // HOUR)
        eligible = self.ready_hours[hours]
        if complete_only:
            eligible = eligible & self.complete_hours[hours]
        return hours[eligible] * HOUR

    def context(self, step, length):
        start = max(self.first_observation, step - length)
        return self.filled_peak[start:step]


def load_jobs(path: Path, split: Split, max_jobs=None):
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"} or path.is_dir():
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"job_id": "string"},
                            usecols=lambda name: name in REQUIRED + OPTIONAL)
    else:
        raise ValueError("Supply a CSV, a Parquet file, or a directory of Parquet shards.")
    return prepare_jobs(frame, split, max_jobs)


def prepare_jobs(frame: pd.DataFrame, split: Split, max_jobs=None):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing trace columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, [col for col in REQUIRED + OPTIONAL if col in frame]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["timestamp"].isna().any() or frame["job_id"].isna().any():
        raise ValueError("Every trace row needs a timestamp and job_id.")
    frame = frame.loc[frame["timestamp"].between(
        split.start, split.timestamp(split.end), inclusive="left"
    )]
    if frame.empty:
        raise ValueError("The data has no rows inside the specified experiment dates.")
    if not frame["timestamp"].eq(frame["timestamp"].dt.floor("5min")).all():
        raise ValueError("Use UTC five-minute window-start timestamps.")
    frame["job_id"] = frame["job_id"].astype("string")
    if max_jobs is not None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive.")
        sample = frame[["job_id"] + (["sample_rank"] if "sample_rank" in frame else [])]
        sample = sample.drop_duplicates("job_id").sort_values(
            list(reversed(sample.columns))
        ).head(max_jobs)
        frame = frame.loc[frame["job_id"].isin(sample["job_id"])]
    if frame.duplicated(["job_id", "timestamp"]).any():
        raise ValueError("Each job_id/timestamp pair must appear once. Aggregate tasks first.")
    for column in ["priority", "scheduling_class", "task_count"]:
        if column not in frame:
            frame[column] = 1 if column == "task_count" else np.nan
    float_columns = AVERAGES + PEAKS + METADATA + ["task_count"]
    frame[float_columns] = frame[float_columns].astype("float32")
    usage_and_requests = frame[AVERAGES + PEAKS + REQUESTS].to_numpy()
    if not np.isfinite(usage_and_requests).all() or (
        (usage_and_requests < 0) | (usage_and_requests > 1)
    ).any():
        raise ValueError("Usage and requested resources must be finite normalized values in [0, 1].")
    counts = frame["task_count"].to_numpy()
    if not (np.isfinite(counts) & (counts > 0) & (counts == np.floor(counts))).all():
        raise ValueError("task_count must be a positive integer for each observed row.")
    if np.isinf(frame[["priority", "scheduling_class"]].to_numpy()).any():
        raise ValueError("Priority and scheduling class must be finite or missing.")

    index = pd.date_range(split.start, periods=split.end, freq=WINDOW)
    jobs = []
    for job_id, group in frame.groupby("job_id", sort=True):
        group = group.set_index("timestamp").sort_index()
        if any(group[name].nunique(dropna=False) != 1 for name in METADATA):
            raise ValueError(f"Job {job_id} has inconsistent initial metadata.")
        metadata = group.iloc[0]
        aligned = group[AVERAGES + PEAKS + ["task_count"]].reindex(index)
        peak = aligned[PEAKS].to_numpy()
        observed = np.isfinite(peak[:, 0])
        cumulative = np.r_[0, observed.cumsum()]
        jobs.append(Job(
            str(job_id), index, aligned[AVERAGES].to_numpy(), peak,
            aligned["task_count"].to_numpy(), metadata[REQUESTS].to_numpy(dtype=float),
            float(metadata["priority"]), float(metadata["scheduling_class"]),
            observed, aligned[PEAKS].ffill().to_numpy(), int(np.flatnonzero(observed)[0]),
            observed.reshape(-1, HOUR).all(axis=1), cumulative[:-1:HOUR] >= HOUR,
        ))
    return jobs


def demo_data(split: Split, seed=42):
    """Small deterministic workloads for checking the pipeline without Borg access."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(split.start, periods=split.end, freq=WINDOW)
    phase = 2 * np.pi * np.arange(split.end) / DAY
    jobs = []
    for i in range(2):
        cpu = (i + 1) * (0.035 + 0.015 * (1 + np.sin(phase - i)))
        cpu += rng.uniform(0, 0.007, len(index))
        memory = (i + 1) * (0.07 + 0.017 * (1 + np.cos(phase / 2 + i)))
        spikes = rng.random(len(index)) < 0.015
        jobs.append(pd.DataFrame({
            "job_id": f"demo-{i + 1}", "timestamp": index, "cpu_avg": cpu,
            "memory_avg": memory, "cpu_peak": cpu * 1.4 + spikes * 0.08,
            "memory_peak": memory * 1.15 + spikes * 0.08,
            "cpu_request": 0.22 * (i + 1), "memory_request": 0.35 * (i + 1),
            "task_count": 3 + i, "priority": 120 - 10 * i, "scheduling_class": 3 - 3 * i,
        }))
    return pd.concat(jobs, ignore_index=True)
