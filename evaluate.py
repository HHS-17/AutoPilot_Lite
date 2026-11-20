"""Run chronological calibration and an hourly held-out replay."""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from data import DAY, HOUR, RESOURCES, Split, demo_data, load_jobs, prepare_jobs
from models import (
    CHRONOS_CHECKPOINT, TOTO_CHECKPOINT, Autopilot, AutopilotCosts, ChronosBolt,
    FixedLimits, LightGBM, Toto,
)


METHODS = ("baseline", "autopilot", "lightgbm", "chronos", "toto")
OVERRUN_PROXIES = {"cpu": "throttling", "memory": "oom"}
MARGINS = (0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.0)
AUTOPILOT_GRID = (
    AutopilotCosts(overrun=10.0, change=0.01, switch=0.01),
    AutopilotCosts(),
    AutopilotCosts(overrun=60.0, change=0.01, switch=0.01),
    AutopilotCosts(overrun=60.0, change=0.05, switch=0.05),
)


@dataclass
class Stabilizer:
    drop_fraction: float
    wait_hours: int
    limit: float | None = None
    low_hours: int = 0
    low_peak: float = 0.0

    def apply(self, proposed, previous_peak):
        if self.limit is None or proposed >= self.limit:
            changed = self.limit is not None and proposed > self.limit
            self.limit = proposed
            self.low_hours = 0
        elif proposed >= self.limit * (1 - self.drop_fraction) or (
            previous_peak is None or previous_peak > proposed
        ):
            changed = False
            self.low_hours = 0
        else:
            if self.low_hours and self.low_peak <= proposed:
                self.low_hours += 1
                self.low_peak = max(self.low_peak, previous_peak)
            else:
                self.low_hours = 1
                self.low_peak = previous_peak
            changed = self.low_hours >= self.wait_hours
            if changed:
                self.limit = proposed
                self.low_hours = 0
        return self.limit, changed


def replay(job, steps, raw_limits, timing, margins, drop_fraction, wait_hours):
    """A proposed limit sees only completed history. Actual demand is scored afterward."""
    proposed = raw_limits * (1 + np.asarray(margins))
    if not np.isfinite(proposed).all() or (proposed < 0).any():
        raise ValueError(f"Invalid recommendation for job {job.job_id}.")
    windows = steps[:, None] + np.arange(HOUR)
    previous_peak = job.peak[windows - HOUR].max(axis=1)
    complete = job.complete_hours[steps // HOUR]
    frames = []
    for i, resource in enumerate(RESOURCES):
        stabilizer = Stabilizer(drop_fraction, wait_hours)
        decisions = [
            stabilizer.apply(limit, last if np.isfinite(last) else None)
            for limit, last in zip(proposed[:, i], previous_peak[:, i])
        ]
        applied = np.asarray([limit for limit, _ in decisions])
        changed = np.asarray([changed for _, changed in decisions], dtype=bool)
        peaks = job.peak[windows, i]
        average = job.average[windows, i]
        count = job.task_count[windows]
        reserved = (applied[:, None] * count).sum(axis=1) / HOUR
        difference = applied[:, None] - average
        slack = (np.maximum(difference, 0) * count).sum(axis=1) / HOUR
        net_slack = (difference * count).sum(axis=1) / HOUR
        frames.append(pd.DataFrame({
            "job_id": job.job_id, "timestamp": job.index[steps], "step": steps,
            "scheduling_class": job.scheduling_class,
            "resource": resource, "proposed_limit": proposed[:, i], "limit": applied,
            "scored": complete, "actual_peak": peaks.max(axis=1),
            "overrun_windows": np.where(complete, (peaks > applied[:, None]).sum(axis=1), 0),
            "observed_windows": complete.astype(int) * HOUR,
            "reserved_resource_hours": np.where(complete, reserved, np.nan),
            "absolute_slack_resource_hours": np.where(complete, slack, np.nan),
            "net_slack_resource_hours": np.where(complete, net_slack, np.nan),
            "changed": changed, "comparable_limit": np.arange(len(steps)) > 0,
            "prediction_ms": timing,
        }))
    return pd.concat(frames, ignore_index=True)


def summarize(rows):
    windows = int(rows["observed_windows"].sum())
    overruns = int(rows["overrun_windows"].sum())
    reserved = float(rows["reserved_resource_hours"].sum())
    slack = float(rows["absolute_slack_resource_hours"].sum())
    changes = int(rows["changed"].sum())
    comparable = int(rows["comparable_limit"].sum())
    return {
        "recommendations": len(rows),
        "scored_hours": int(rows["scored"].sum()),
        "observed_windows": windows,
        "overrun_windows": overruns,
        "overrun_rate": overruns / windows if windows else np.nan,
        "reserved_resource_hours": reserved if windows else np.nan,
        "absolute_slack_resource_hours": slack if windows else np.nan,
        "net_slack_resource_hours": rows["net_slack_resource_hours"].sum() if windows else np.nan,
        "relative_slack": (slack / reserved if reserved else 0.0) if windows else np.nan,
        "limit_changes": changes,
        "limit_change_rate": changes / comparable if comparable else np.nan,
        "average_prediction_ms": rows["prediction_ms"].mean(),
        "p95_prediction_ms": rows["prediction_ms"].quantile(0.95),
    }


def choose_setting(summaries, budget):
    evaluated = [
        i for i, summary in enumerate(summaries) if np.isfinite(summary["overrun_rate"])
    ]
    if not evaluated:
        raise ValueError("No complete tuning hours were available to choose a setting.")
    feasible = [i for i in evaluated if summaries[i]["overrun_rate"] <= budget]
    if feasible:
        return min(feasible, key=lambda i: (summaries[i]["relative_slack"],
                                           summaries[i]["overrun_rate"]))
    return min(evaluated, key=lambda i: (summaries[i]["overrun_rate"],
                                        summaries[i]["relative_slack"]))


def predictions(model, jobs, begin, end):
    cached = {}
    for job in jobs:
        steps = job.steps(begin, end)
        if len(steps):
            cached[job.job_id] = (steps, *model.predict(job, steps))
    return cached


def score_predictions(jobs, forecast, margins, args, start=0):
    scored = []
    for job in jobs:
        if job.job_id in forecast:
            rows = replay(job, *forecast[job.job_id], margins, args.drop_fraction, args.wait_hours)
            scored.append(rows.loc[rows["step"] >= start])
    return pd.concat(scored, ignore_index=True)


def tuning_metrics(jobs, forecast, margins, args):
    rows = score_predictions(jobs, forecast, margins, args)
    return {resource: summarize(group) for resource, group in rows.groupby("resource", sort=False)}


def calibrate(name, candidates, budgets):
    chosen = []
    records = []
    for resource, budget in zip(RESOURCES, budgets):
        summaries = [results[resource] for _, results in candidates]
        winner = choose_setting(summaries, budget)
        chosen.append(winner)
        for i, ((setting, _), summary) in enumerate(zip(candidates, summaries)):
            records.append({
                "model": name, "resource": resource, **setting,
                "overrun_budget": budget, "selected": i == winner, **summary,
            })
    return chosen, records


def evaluate_method(name, jobs, split, args, device):
    train_end, tune_end, end = split.train_end, split.tune_end, split.end
    budgets = (args.cpu_budget, args.memory_budget)
    margins = [0.0, 0.0]
    tuning, selected = [], {}
    importance = None
    if name == "autopilot":
        candidates = []
        for costs in AUTOPILOT_GRID:
            tune = predictions(Autopilot(costs, costs), jobs, train_end, tune_end)
            scores = tuning_metrics(jobs, tune, margins, args)
            candidates.append((asdict(costs), scores))
        winners, tuning = calibrate(name, candidates, budgets)
        cpu, memory = (AUTOPILOT_GRID[i] for i in winners)
        selected = {"cpu": asdict(cpu), "memory": asdict(memory)}
        forecast = predictions(Autopilot(cpu, memory), jobs, train_end, end)
    elif name == "baseline":
        forecast = predictions(FixedLimits(), jobs, train_end, end)
    else:
        if name == "lightgbm":
            model = LightGBM(args.seed, args.threads)
            model.fit(jobs, train_end)
            importance = model.importance()
        elif name == "chronos":
            model = ChronosBolt(device, args.context, args.batch_size, args.local_models_only)
        else:
            model = Toto(device, args.context, args.batch_size, args.toto_samples,
                         args.toto_samples_per_batch, args.local_models_only)

        tune = predictions(model, jobs, train_end, tune_end)
        candidates = [
            ({"margin": margin}, tuning_metrics(jobs, tune, [margin, margin], args))
            for margin in MARGINS
        ]
        winners, tuning = calibrate(name, candidates, budgets)
        margins = [MARGINS[i] for i in winners]
        selected = dict(zip(RESOURCES, margins))
        test = predictions(model, jobs, tune_end, end)
        forecast = {**tune, **test}
        for job_id in tune.keys() & test.keys():
            forecast[job_id] = tuple(
                np.concatenate((before, after), axis=0)
                for before, after in zip(tune[job_id], test[job_id])
            )
    scored = score_predictions(jobs, forecast, margins, args, start=tune_end)
    return scored, tuning, selected, importance


def job_day_metrics(name, rows, jobs):
    """Estimate the paper's job-day slack measures using aggregated trace usage."""
    complete = rows.loc[rows["scored"]].copy()
    complete["trace_day"] = complete["step"] // DAY + 1
    for (job_id, resource, day), group in complete.groupby(
        ["job_id", "resource", "trace_day"], sort=False
    ):
        i = RESOURCES.index(resource)
        minutes = group["step"].to_numpy()[:, None] + np.arange(HOUR)
        p95_usage = float(np.quantile(jobs[job_id].average[minutes, i], 0.95))
        average_limit = group["limit"].mean()
        yield {
            "model": name, "job_id": job_id, "resource": resource, "trace_day": day,
            "scored_hours": len(group), "p95_five_minute_average_usage": p95_usage,
            "average_limit": average_limit,
            "relative_slack_p95": (average_limit - p95_usage) / average_limit if average_limit else np.nan,
            "mean_net_slack_resource": group["net_slack_resource_hours"].sum() / len(group),
        }


def write_plots(output, per_job, summary, manifest=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    labels = {"baseline": "Fixed baseline", "autopilot": "Autopilot ML",
              "lightgbm": "LightGBM", "chronos": "Chronos-Bolt", "toto": "Toto"}
    colors = {"baseline": "#6b7785", "autopilot": "#285e8e", "lightgbm": "#298562",
              "chronos": "#a65f24", "toto": "#7148a5"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for resource, axis in zip(RESOURCES, axes):
        for name, group in per_job[per_job["resource"] == resource].groupby("model", sort=False):
            values = np.sort(group["relative_slack"].dropna().to_numpy())
            if len(values):
                axis.step(np.r_[values[0], values], np.arange(len(values) + 1) / len(values),
                          where="post", label=labels[name], color=colors[name], linewidth=2)
        axis.set(title=resource.upper() if resource == "cpu" else "Memory",
                 xlabel="Relative slack", ylabel="Cumulative share of jobs", xlim=(0, 1), ylim=(0, 1))
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="lower right")
    figure.savefig(output / "slack_cdf.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for resource, axis in zip(RESOURCES, axes):
        for row in summary.loc[summary["resource"] == resource].itertuples():
            axis.scatter(row.relative_slack, row.overrun_rate, s=55, label=labels[row.model],
                         color=colors[row.model])
        axis.set(title=resource.upper() if resource == "cpu" else "Memory",
                 xlabel="Relative slack",
                 ylabel="Throttling proxy rate" if resource == "cpu" else "OOM proxy rate")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(output / "tradeoff.png", dpi=160)
    plt.close(figure)

    def plot_metric(axis, values, title, unit, value_format, scale="linear"):
        values = values.reindex(METHODS).dropna()
        positions = np.arange(len(values))
        if scale == "symlog":
            # The linear section includes zero and keeps small rates visible.
            axis.set_xscale("symlog", linthresh=0.01)
            axis.set_xticks([0, 0.01, 0.1, 1, 10, 100])
        bars = axis.barh(positions, values.to_numpy(), height=0.62,
                         color=[colors[name] for name in values.index])
        axis.set_yticks(positions, [labels[name] for name in values.index])
        axis.invert_yaxis()
        axis.set(title=title, xlabel=unit)
        axis.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _: f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"
        ))
        axis.margins(x=0.25)
        axis.set_xlim(left=0)
        if scale == "symlog":
            axis.set_xlim(0, 100)
        axis.bar_label(bars, labels=[format(value, value_format) for value in values], padding=4)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.22)
        axis.tick_params(axis="y", length=0)
        axis.spines[["top", "right"]].set_visible(False)

    metrics = {resource: rows.set_index("model")
               for resource, rows in summary.groupby("resource")}
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 6.7), layout="constrained")
    figure.suptitle("Slack over the test period", fontsize=16)
    for column, (resource, label) in enumerate(zip(RESOURCES, ("CPU", "memory"))):
        rows = metrics[resource]
        plot_metric(axes[0, column], rows["relative_slack"] * 100, f"Relative {label} slack",
                    "Unused reserved capacity (%)", ".2f")
        axes[0, column].set_xlim(0, 110)
        plot_metric(axes[1, column], rows["absolute_slack_resource_hours"], f"Absolute {label} slack",
                    f"Unused normalized {label}-hours", ",.0f")
    figure.savefig(output / "slack_by_method.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 6.7), layout="constrained")
    figure.suptitle("Overrun risk and limit changes", fontsize=16)
    for column, (resource, label, proxy) in enumerate(zip(
        RESOURCES, ("CPU", "Memory"), ("Throttling", "OOM")
    )):
        rows = metrics[resource]
        plot_metric(axes[0, column], rows["overrun_rate"] * 100, f"{proxy} proxy rate",
                    "Scored five-minute windows (%) · log above 0.01%", ".3f", scale="symlog")
        plot_metric(axes[1, column], rows["limit_changes"], f"{label} limit changes",
                    "Number of changes during the test period", ",.0f")
    figure.savefig(output / "overruns_and_limit_changes.png", dpi=180)
    plt.close(figure)

    # The evaluator records the same joint CPU/memory inference time on both rows.
    timing = metrics["cpu"]
    panels = [
        (timing["average_prediction_ms"], "Mean prediction time", "ms per job-hour", ".3f"),
        (timing["p95_prediction_ms"], "P95 prediction time", "ms per job-hour", ".3f"),
    ]
    if manifest:
        runs = [manifest] if "methods" in manifest else manifest.values()
        runtime = {name: details["elapsed_seconds"] / 60
                   for run in runs for name, details in run["methods"].items()}
        panels.append((pd.Series(runtime), "Elapsed method runtime", "Minutes for the full run", ".2f"))
    figure, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6),
                                layout="constrained", squeeze=False)
    figure.suptitle("Compute cost by method", fontsize=16)
    for axis, panel in zip(axes[0], panels):
        plot_metric(axis, *panel)
    figure.savefig(output / "compute_cost.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Job-panel CSV, Parquet file or Parquet directory.")
    source.add_argument("--demo", action="store_true", help="Use two synthetic jobs and 1/1/1 days.")
    source.add_argument("--plot-from", type=Path, help="Recreate plots from an existing results directory.")
    parser.add_argument("--models", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output", type=Path, help="New or empty results directory.")
    parser.add_argument("--start", default="2019-05-01T07:00:00Z", help="Trace start timestamp.")
    parser.add_argument("--train-days", type=int)
    parser.add_argument("--tune-days", type=int)
    parser.add_argument("--test-days", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context", type=int, default=2048, help="Foundation-model context in windows.")
    parser.add_argument("--batch-size", type=int, default=4, help="Hours per neural-model inference batch.")
    parser.add_argument("--toto-samples", type=int, default=256)
    parser.add_argument("--toto-samples-per-batch", type=int, default=16)
    parser.add_argument("--local-models-only", action="store_true", help="Use cached weights only.")
    parser.add_argument("--cpu-budget", type=float, default=0.05)
    parser.add_argument("--memory-budget", type=float, default=0.02)
    parser.add_argument("--drop-fraction", type=float, default=0.15)
    parser.add_argument("--wait-hours", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.toto_samples, args.toto_samples_per_batch,
           args.wait_hours) < 1 or (args.max_jobs is not None and args.max_jobs < 1):
        parser.error("Batch sizes, samples, jobs, threads and wait hours must be positive.")
    if args.toto_samples % args.toto_samples_per_batch:
        parser.error("--toto-samples must be divisible by --toto-samples-per-batch.")
    if not HOUR <= args.context <= 2048:
        parser.error("--context must be between 12 and 2048 five-minute windows.")
    if not all(0 <= value <= 1 for value in (args.cpu_budget, args.memory_budget, args.drop_fraction)):
        parser.error("Budgets and downscale threshold must be in [0, 1].")
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    return args


def main():
    args = parse_args()
    if args.plot_from:
        folder = args.plot_from
        manifest_path = folder / "run.json"
        manifest = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        write_plots(folder, pd.read_csv(folder / "per_job.csv"), pd.read_csv(folder / "summary.csv"),
                    manifest)
        print(f"Plots written to {folder.resolve()}")
        return

    days = (1, 1, 1) if args.demo else (14, 7, 10)
    actual_days = [default if provided is None else provided for default, provided in
                   zip(days, (args.train_days, args.tune_days, args.test_days))]
    split = Split(args.start, *actual_days)
    device = "cpu"
    if {"chronos", "toto"}.intersection(args.models):
        import torch

        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested. Install a CUDA-enabled PyTorch build and driver.")
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
            device = "cuda"

    jobs = (
        prepare_jobs(demo_data(split, args.seed), split, args.max_jobs) if args.demo
        else load_jobs(args.data, split, args.max_jobs)
    )
    for label, start, stop in (("tuning", split.train_end, split.tune_end),
                               ("test", split.tune_end, split.end)):
        if not any(len(job.steps(start, stop, complete_only=True)) for job in jobs):
            raise ValueError(f"No complete {label} hours with history. Check the dates and input data.")
    output = args.output or Path("results") / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Use a new or empty output directory. Existing files found in {output}.")
    manifest = {
        "options": vars(args),
        "periods_utc": {
            "training_start": split.start, "tuning_start": split.timestamp(split.train_end),
            "test_start": split.timestamp(split.tune_end), "test_end_exclusive": split.timestamp(split.end),
        },
        "job_ids": [job.job_id for job in jobs], "neural_device": device,
        "checkpoints": {"chronos": CHRONOS_CHECKPOINT, "toto": TOTO_CHECKPOINT},
        "methods": {},
    }
    summaries, per_job, per_day, tuning = [], [], [], []
    jobs_by_id = {job.job_id: job for job in jobs}
    for name in dict.fromkeys(args.models):
        print(f"Evaluating {name} on {len(jobs)} jobs...", flush=True)
        started = perf_counter()
        rows, settings, selected, importance = evaluate_method(name, jobs, split, args, device)
        rows.drop(columns="step").to_csv(output / f"hourly_{name}.csv", index=False)
        for resource, group in rows.groupby("resource", sort=False):
            summaries.append({"model": name, "resource": resource,
                              "overrun_proxy": OVERRUN_PROXIES[resource], **summarize(group)})
        for (job_id, resource), group in rows.groupby(["job_id", "resource"], sort=False):
            per_job.append({"model": name, "job_id": job_id, "resource": resource,
                            "scheduling_class": group["scheduling_class"].iat[0], **summarize(group)})
        per_day.extend(job_day_metrics(name, rows, jobs_by_id))
        tuning.extend(settings)
        manifest["methods"][name] = {"selected": selected, "elapsed_seconds": perf_counter() - started}
        if importance is not None:
            importance.to_csv(output / "lightgbm_importance.csv", index=False)
        pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
        pd.DataFrame(per_job).to_csv(output / "per_job.csv", index=False)
        pd.DataFrame(per_day).to_csv(output / "per_job_day.csv", index=False)
        pd.DataFrame(tuning, columns=None if tuning else [
            "model", "resource", "overrun_budget", "selected"
        ]).to_csv(output / "tuning.csv", index=False)
        (output / "run.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_plots(output, pd.DataFrame(per_job), pd.DataFrame(summaries), manifest)
    columns = ["model", "resource", "relative_slack", "overrun_rate", "limit_changes", "average_prediction_ms"]
    print(pd.DataFrame(summaries)[columns].to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()
