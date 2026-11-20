"""Four recommenders and the fixed user-request baseline."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from data import HOUR, RESOURCES, WINDOW_MINUTES, Job


CHRONOS_CHECKPOINT = "amazon/chronos-bolt-base"
TOTO_CHECKPOINT = "Datadog/Toto-Open-Base-1.0"


class FixedLimits:
    def predict(self, job: Job, steps: np.ndarray):
        return np.broadcast_to(job.request, (len(steps), 2)), np.zeros(len(steps))


@dataclass(frozen=True)
class AutopilotCosts:
    overrun: float = 25.0
    underrun: float = 1.0
    change: float = 0.02
    switch: float = 0.02


class _AutopilotEnsemble:
    """Paper equations 4–9 on a histogram of the observed five-minute proxies."""

    BINS = np.r_[0.0, np.geomspace(1e-7, 1.0, 255)]
    HALF_LIVES = (6.0, 24.0, 48.0)
    MARGIN_FRACTIONS = (0.0, 0.05, 0.15)

    def __init__(self, request, costs):
        self.costs = costs
        self.decay = 1 - np.exp2(-1 / np.asarray(self.HALF_LIVES))
        self.margins = np.asarray(self.MARGIN_FRACTIONS) * request
        self.histogram = np.zeros((len(self.decay), len(self.BINS)))
        size = len(self.decay) * len(self.margins)
        self.arm_scores = np.zeros(size)
        self.previous_indices = None
        self.previous_limits = None
        self.previous_changes = None
        self.selected_arm = None
        self.score_decay = 1 - np.exp2(-1 / 24)

    def update(self, usage):
        """Score the last limits on the completed hour, then recommend the next."""
        samples = usage[np.isfinite(usage)]
        self.histogram *= 1 - self.decay[:, None]
        if samples.size:
            buckets = np.searchsorted(self.BINS, samples, side="left")
            counts = np.bincount(buckets, minlength=len(self.BINS)) / len(samples)
            self.histogram += self.decay[:, None] * counts
            if self.previous_limits is not None:
                limits = self.previous_limits[:, None]
                realized = (
                    self.costs.overrun * (samples > limits).mean(axis=1)
                    + self.costs.underrun * (samples < limits).mean(axis=1)
                    + self.costs.change * self.previous_changes
                )
                self.arm_scores += self.score_decay * (realized - self.arm_scores)

        if self.histogram.sum() == 0:
            return np.nan
        cumulative = self.histogram.cumsum(axis=1)
        overruns = cumulative[:, -1:] - cumulative
        underruns = np.c_[np.zeros(len(self.decay)), cumulative[:, :-1]]
        candidate_costs = self.costs.overrun * overruns + self.costs.underrun * underruns
        # Each margin variant of each half-life keeps its own previous decision.
        candidate_costs = candidate_costs.repeat(len(self.margins), axis=0)
        if self.previous_indices is not None:
            candidate_costs += self.costs.change * (
                np.arange(len(self.BINS))[None, :] != self.previous_indices[:, None]
            )
        indices = candidate_costs.argmin(axis=1)
        limits = self.BINS[indices] + np.tile(self.margins, len(self.decay))
        scores = self.arm_scores.copy()
        changes = np.zeros(len(limits), dtype=bool)
        if self.selected_arm is not None:
            changes = limits != self.previous_limits
            scores += self.costs.switch * (np.arange(len(limits)) != self.selected_arm)
            scores += self.costs.change * (limits != self.previous_limits[self.selected_arm])
        self.selected_arm = int(scores.argmin())
        self.previous_indices = indices
        self.previous_limits = limits
        self.previous_changes = changes
        return limits[self.selected_arm]


class Autopilot:
    def __init__(self, cpu=AutopilotCosts(), memory=AutopilotCosts()):
        self.costs = (cpu, memory)

    def predict(self, job: Job, steps: np.ndarray):
        engines = [
            _AutopilotEnsemble(request, costs)
            for request, costs in zip(job.request, self.costs)
        ]
        limits = np.empty((len(steps), 2))
        timing = np.empty(len(steps))
        if not len(steps):
            return limits, timing
        first_step = (job.first_observation // HOUR + 1) * HOUR
        position = 0
        for step in range(first_step, steps[-1] + 1, HOUR):
            started = perf_counter()
            previous_hour = job.peak[step - HOUR:step]
            proposal = [engine.update(previous_hour[:, i]) for i, engine in enumerate(engines)]
            if step == steps[position]:
                limits[position] = proposal
                timing[position] = (perf_counter() - started) * 1000
                position += 1
        return limits, timing


class LightGBM:
    """Two global 0.98 quantile models, one target per resource."""

    def __init__(self, seed, threads):
        self.seed = seed
        self.threads = threads
        self.models = []

    @staticmethod
    def features(job: Job, steps: np.ndarray):
        last = int(steps[-1])
        observations = pd.DataFrame(
            np.column_stack((job.average[:last], job.peak[:last])),
            columns=["cpu_avg", "memory_avg", "cpu_peak", "memory_peak"],
        ).reindex(range(last + 1))
        features = {}
        for i, resource in enumerate(RESOURCES):
            average = observations[f"{resource}_avg"]
            peak = observations[f"{resource}_peak"]
            for lag in (1, 2, 3, HOUR, 6 * HOUR, 24 * HOUR):
                features[f"{resource}_peak_lag_{lag}"] = peak.shift(lag).iloc[steps].to_numpy()
            features[f"{resource}_peak_change_1h"] = (
                peak.shift(1) - peak.shift(1 + HOUR)
            ).iloc[steps].to_numpy()
            features[f"{resource}_request"] = job.request[i]
            for hours in (1, 6, 24, 72):
                avg_window = average.shift(1).rolling(hours * HOUR, min_periods=1)
                peak_window = peak.shift(1).rolling(hours * HOUR, min_periods=1)
                features[f"{resource}_avg_{hours}h"] = avg_window.mean().iloc[steps].to_numpy()
                features[f"{resource}_max_{hours}h"] = peak_window.max().iloc[steps].to_numpy()
                features[f"{resource}_p90_{hours}h"] = peak_window.quantile(0.9).iloc[steps].to_numpy()
        origins = job.index[steps]
        angle = 2 * np.pi * origins.hour / 24
        features["hour_sin"] = np.sin(angle)
        features["hour_cos"] = np.cos(angle)
        features["day_of_week"] = origins.dayofweek
        features["priority"] = job.priority
        features["scheduling_class"] = job.scheduling_class
        return pd.DataFrame(features, index=steps, dtype=np.float32)

    def fit(self, jobs, train_end):
        from lightgbm import LGBMRegressor

        feature_frames, targets = [], []
        for job in jobs:
            steps = job.steps(0, train_end, complete_only=True)
            if not len(steps):
                continue
            feature_frames.append(self.features(job, steps))
            targets.append(job.peak[steps[:, None] + np.arange(HOUR)].max(axis=1))
        if not feature_frames:
            raise ValueError("LightGBM needs complete training hours with prior observations.")
        features = pd.concat(feature_frames, ignore_index=True)
        target = np.concatenate(targets)
        self.models = [
            LGBMRegressor(
                objective="quantile", alpha=0.98, n_estimators=200, learning_rate=0.05,
                num_leaves=31, min_child_samples=20, random_state=self.seed,
                n_jobs=self.threads, verbosity=-1,
            ).fit(features, target[:, i].copy())
            for i in range(len(RESOURCES))
        ]

    def predict(self, job: Job, steps: np.ndarray):
        if not len(steps):
            return np.empty((0, 2)), np.empty(0)
        started = perf_counter()
        features = self.features(job, steps)
        limits = np.maximum(0, np.column_stack([model.predict(features) for model in self.models]))
        return limits, np.full(len(steps), (perf_counter() - started) * 1000 / len(steps))

    def importance(self):
        return pd.concat([
            pd.DataFrame({
                "resource": resource, "feature": model.feature_name_,
                "gain": model.booster_.feature_importance(importance_type="gain"),
            }).sort_values("gain", ascending=False)
            for resource, model in zip(RESOURCES, self.models)
        ], ignore_index=True)


def forecast_limits(quantiles):
    """Convert per-resource, per-step quantiles into one limit for each hour."""
    return np.maximum(0, np.max(quantiles, axis=-1))


class ChronosBolt:
    def __init__(self, device, context_length, batch_size, local_only, pipeline=None):
        self.context_length = context_length
        self.batch_size = batch_size
        if pipeline is None:
            import torch
            from chronos import BaseChronosPipeline

            pipeline = BaseChronosPipeline.from_pretrained(
                CHRONOS_CHECKPOINT, device_map=device, torch_dtype=torch.float32,
                local_files_only=local_only,
            )
        self.pipeline = pipeline

    def predict(self, job: Job, steps: np.ndarray):
        import torch

        limits = np.empty((len(steps), 2))
        timing = np.empty(len(steps))
        for offset in range(0, len(steps), self.batch_size):
            batch = steps[offset:offset + self.batch_size]
            started = perf_counter()
            contexts = [
                torch.from_numpy(values.copy())
                for step in batch
                for values in job.context(step, self.context_length).T
            ]
            with torch.inference_mode():
                quantiles, _ = self.pipeline.predict_quantiles(
                    contexts, prediction_length=HOUR, quantile_levels=[0.9]
                )
                quantiles = quantiles[..., 0].float().cpu().numpy().reshape(len(batch), 2, HOUR)
            limits[offset:offset + len(batch)] = forecast_limits(quantiles)
            timing[offset:offset + len(batch)] = (perf_counter() - started) * 1000 / len(batch)
        return limits, timing


class Toto:
    def __init__(self, device, context_length, batch_size, samples, samples_per_batch,
                 local_only, forecaster=None):
        from toto.data.util.dataset import MaskedTimeseries

        self.device = device
        self.context_length = context_length
        self.batch_size = batch_size
        self.samples = samples
        self.samples_per_batch = samples_per_batch
        self.input_type = MaskedTimeseries
        self._backbone = None
        if forecaster is None:
            from huggingface_hub import snapshot_download
            from toto.inference.forecaster import TotoForecaster
            from toto.model.toto import Toto as TotoModel

            checkpoint = snapshot_download(
                TOTO_CHECKPOINT, allow_patterns=["config.json", "model.safetensors"],
                local_files_only=local_only,
            )
            model = TotoModel.from_pretrained(
                checkpoint, use_memory_efficient_attention=False
            ).to(device)
            self._backbone = model.model
            forecaster = TotoForecaster(self._backbone)
        self.forecaster = forecaster

    def _first_patch_distribution(self, inputs):
        from toto.data.util.dataset import pad_array, pad_id_mask

        backbone = self._backbone
        stride, patch_size = backbone.patch_embed.stride, backbone.patch_embed.patch_size
        series = pad_array(inputs.series, stride)
        embeddings, loc, scale = backbone.backbone(
            series, pad_array(inputs.padding_mask, stride), pad_id_mask(inputs.id_mask, stride),
            scaling_prefix_length=series.shape[-1],
        )
        forecast_positions = embeddings[..., -patch_size:, :][..., :HOUR, :]
        base_distribution = backbone.output_distribution(forecast_positions)
        if loc.shape[-1] > 1:
            loc = loc[..., -patch_size:][..., :HOUR]
            scale = scale[..., -patch_size:][..., :HOUR]
        return self.forecaster.create_affine_transformed(base_distribution, loc, scale)

    def _forecast_quantiles(self, inputs):
        import torch
        from toto.data.util.dataset import replace_extreme_values

        if self._backbone is None or HOUR > self._backbone.patch_embed.patch_size:
            forecast = self.forecaster.forecast(
                inputs, prediction_length=HOUR, num_samples=self.samples,
                samples_per_batch=self.samples_per_batch,
            )
            return torch.quantile(forecast.samples.float(), 0.98, dim=-1)

        # The whole forecast fits in the first patch. Every sample draws from
        # the same conditional distribution, so the context needs one forward pass.
        distribution = self._first_patch_distribution(inputs)
        samples = torch.cat([
            distribution.sample((self.samples_per_batch,))
            for _ in range(self.samples // self.samples_per_batch)
        ])
        return torch.quantile(replace_extreme_values(samples).float(), 0.98, dim=0)

    def predict(self, job: Job, steps: np.ndarray):
        import torch

        limits = np.empty((len(steps), 2))
        timing = np.empty(len(steps))
        for offset in range(0, len(steps), self.batch_size):
            batch = steps[offset:offset + self.batch_size]
            started = perf_counter()
            histories = [job.context(step, self.context_length) for step in batch]
            width = max(map(len, histories))
            series = np.zeros((len(batch), 2, width), dtype=np.float32)
            mask = np.zeros(series.shape, dtype=bool)
            stamps = np.zeros(series.shape, dtype=np.int64)
            for i, (step, history) in enumerate(zip(batch, histories)):
                series[i, :, -len(history):] = history.T
                mask[i, :, -len(history):] = True
                stamps[i, :, -len(history):] = job.index[step - len(history):step].asi8 // 10**9
            inputs = self.input_type(
                series=torch.as_tensor(series, device=self.device),
                padding_mask=torch.as_tensor(mask, device=self.device),
                id_mask=torch.zeros(series.shape, dtype=torch.long, device=self.device),
                timestamp_seconds=torch.as_tensor(stamps, device=self.device),
                time_interval_seconds=torch.full(
                    series.shape[:2], WINDOW_MINUTES * 60, dtype=torch.long, device=self.device
                ),
            )
            with torch.inference_mode():
                quantiles = self._forecast_quantiles(inputs).cpu().numpy()
            limits[offset:offset + len(batch)] = forecast_limits(quantiles)
            timing[offset:offset + len(batch)] = (perf_counter() - started) * 1000 / len(batch)
        return limits, timing
