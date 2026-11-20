import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data import Split, demo_data, load_jobs, prepare_jobs
from evaluate import Stabilizer, choose_setting, replay, summarize
from models import Autopilot, ChronosBolt, FixedLimits, LightGBM, Toto, forecast_limits


class DataTests(unittest.TestCase):
    def test_missing_windows_are_filled_causally_and_remain_unobserved(self):
        split = Split("2019-05-01", 1, 1, 1)
        source = demo_data(split).query("job_id == 'demo-1'").copy()
        source.loc[source.index[0], ["cpu_peak", "memory_peak"]] = [0.1, 0.2]
        source.loc[source.index[2], ["cpu_peak", "memory_peak"]] = [0.3, 0.4]
        source = source.drop(source.index[1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            source.to_csv(path, index=False)
            job = load_jobs(path, split)[0]
        np.testing.assert_allclose(job.context(2, 12), [[0.1, 0.2], [0.1, 0.2]])
        self.assertFalse(job.observed[1])
        self.assertFalse(job.complete_hours[0])

    def test_training_target_ends_before_tuning_begins(self):
        split = Split("2019-05-01", 1, 1, 1)
        job = prepare_jobs(demo_data(split), split)[0]
        self.assertEqual(job.steps(0, split.train_end)[-1] + 12, split.train_end)
        self.assertEqual(job.steps(split.train_end, split.tune_end)[0], split.train_end)
        self.assertEqual(job.steps(split.tune_end, split.end)[0], split.tune_end)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.split = Split("2019-05-01", 1, 1, 1)
        source = demo_data(self.split)
        source[["cpu_avg", "cpu_peak"]] = 0.1
        source[["memory_avg", "memory_peak"]] = 0.2
        self.jobs = prepare_jobs(source, self.split)

    def test_autopilot_covers_a_constant_workload(self):
        steps = self.jobs[0].steps(0, self.split.train_end)
        limits, _ = Autopilot().predict(self.jobs[0], steps)
        self.assertTrue(np.all(limits >= [0.1, 0.2]))
        self.assertTrue(np.all(limits <= [0.13, 0.26]))

    def test_prediction_at_boundary_is_independent_of_current_and_future_usage(self):
        step = self.split.train_end
        source = demo_data(self.split)
        source[["cpu_avg", "cpu_peak"]] = 0.1
        source[["memory_avg", "memory_peak"]] = 0.2
        future = source["timestamp"] >= self.split.timestamp(step)
        source.loc[future, ["cpu_peak", "memory_peak"]] = 0.95
        changed = prepare_jobs(source, self.split)[0]
        limits, _ = Autopilot().predict(changed, np.array([step]))
        self.assertLess(limits[0, 1], 0.3)
        pd.testing.assert_frame_equal(
            LightGBM.features(self.jobs[0], np.array([step])),
            LightGBM.features(changed, np.array([step])),
        )

    def test_lightgbm_learns_separate_quantile_targets(self):
        model = LightGBM(seed=2, threads=1)
        model.fit(self.jobs, self.split.train_end)
        limits, _ = model.predict(self.jobs[0], np.array([self.split.train_end]))
        self.assertAlmostEqual(limits[0, 0], 0.1, places=6)
        self.assertAlmostEqual(limits[0, 1], 0.2, places=6)
        self.assertEqual({estimator.get_params()["alpha"] for estimator in model.models}, {0.98})

    def test_foundation_model_adapters_convert_quantiles_to_hourly_limits(self):
        import torch

        job = self.jobs[0]
        step = self.split.train_end

        def bolt_output(context, prediction_length, quantile_levels):
            self.assertEqual(quantile_levels, [0.9])
            self.assertEqual(prediction_length, 12)
            self.assertEqual(len(context), 2)
            self.assertAlmostEqual(context[0][-1].item(), 0.1)
            output = torch.zeros((2, 12, 1))
            output[0, 1] = 0.3
            output[1, 2] = 0.5
            return output, None

        def toto_output(inputs, prediction_length, num_samples, samples_per_batch):
            self.assertEqual(tuple(inputs.series.shape), (1, 2, 48))
            self.assertTrue(inputs.padding_mask.all())
            self.assertEqual(prediction_length, 12)
            samples = torch.zeros((1, 2, 12, num_samples))
            samples[0, 0, 1] = 0.4
            samples[0, 1, 2] = 0.6
            return SimpleNamespace(samples=samples)

        bolt = ChronosBolt("cpu", 48, 1, False, pipeline=SimpleNamespace(
            predict_quantiles=bolt_output
        ))
        toto = Toto("cpu", 48, 1, 16, 8, False, forecaster=SimpleNamespace(
            forecast=toto_output
        ))
        np.testing.assert_allclose(bolt.predict(job, np.array([step]))[0], [[0.3, 0.5]])
        np.testing.assert_allclose(toto.predict(job, np.array([step]))[0], [[0.4, 0.6]])

    def test_forecast_limits_use_each_resources_highest_horizon_quantile(self):
        quantiles = np.array([[[0.7, 0.2, 0.3], [-1.0, -0.5, -0.1]]])
        np.testing.assert_array_equal(forecast_limits(quantiles), [[0.7, 0.0]])

    def test_toto_short_forecast_uses_the_start_of_the_final_patch_and_its_scaling(self):
        import torch
        from toto.inference.forecaster import TotoForecaster

        patch_size = 16
        values = torch.arange(2 * patch_size, dtype=torch.float32).expand(1, 2, -1)
        model = SimpleNamespace(
            patch_embed=SimpleNamespace(stride=patch_size, patch_size=patch_size),
            backbone=lambda *args, **kwargs: (values[..., None], values * 2, values + 1),
            output_distribution=lambda x: torch.distributions.Normal(x[..., 0], 1.0),
        )
        toto = Toto("cpu", 32, 1, 16, 8, False, forecaster=SimpleNamespace(
            create_affine_transformed=TotoForecaster.create_affine_transformed
        ))
        toto._backbone = model
        inputs = SimpleNamespace(series=values, padding_mask=torch.ones_like(values, dtype=torch.bool),
                                 id_mask=torch.zeros_like(values, dtype=torch.long))
        distribution = toto._first_patch_distribution(inputs)
        self.assertEqual(tuple(distribution.mean.shape), (1, 2, 12))
        self.assertEqual(distribution.mean[0, 0, 0].item(), 304.0)
        self.assertEqual(distribution.mean[0, 0, -1].item(), 810.0)


class EvaluationTests(unittest.TestCase):
    def test_downscale_waits_and_missing_observations_reset_it(self):
        filter_ = Stabilizer(drop_fraction=0.15, wait_hours=2)
        decisions = [
            filter_.apply(0.5, None),
            filter_.apply(0.3, 0.2),
            filter_.apply(0.3, None),
            filter_.apply(0.3, 0.2),
            filter_.apply(0.3, 0.2),
            filter_.apply(0.7, 0.4),
        ]
        self.assertEqual(decisions, [(0.5, False), (0.5, False), (0.5, False),
                                     (0.5, False), (0.3, True), (0.7, True)])

    def test_tuning_prefers_lowest_slack_that_meets_the_overrun_budget(self):
        summaries = [
            {"overrun_rate": 0.04, "relative_slack": 0.05},
            {"overrun_rate": 0.02, "relative_slack": 0.15},
            {"overrun_rate": 0.01, "relative_slack": 0.25},
        ]
        self.assertEqual(choose_setting(summaries, 0.02), 1)
        self.assertEqual(choose_setting(summaries, 0.005), 2)

    def test_fixed_limits_produce_hand_calculated_slack_and_overrun_rate(self):
        split = Split("2019-05-01", 1, 1, 1)
        source = demo_data(split).query("job_id == 'demo-1'").copy()
        source["task_count"] = 2
        source[["cpu_avg", "cpu_peak", "cpu_request"]] = [0.25, 0.25, 0.5]
        source[["memory_avg", "memory_peak", "memory_request"]] = [0.5, 0.5, 0.5]
        overrun_at = source["timestamp"] == split.timestamp(split.tune_end)
        source.loc[overrun_at, "memory_peak"] = 0.75
        job = prepare_jobs(source, split)[0]
        steps = job.steps(split.tune_end, split.tune_end + 12)
        limits, timing = FixedLimits().predict(job, steps)
        rows = replay(job, steps, limits, timing, [0.0, 0.0], 0.15, 2)
        cpu = summarize(rows.query("resource == 'cpu'"))
        memory = summarize(rows.query("resource == 'memory'"))
        self.assertEqual(cpu["absolute_slack_resource_hours"], 0.5)
        self.assertEqual(cpu["relative_slack"], 0.5)
        self.assertEqual(memory["overrun_rate"], 1 / 12)
        self.assertEqual(memory["limit_changes"], 0)


if __name__ == "__main__":
    unittest.main()
