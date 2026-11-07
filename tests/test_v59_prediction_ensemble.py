from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "AIforGreen-参赛包"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from acn_causal.student_api import ArrivalFeatures, Prediction  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FixedRegressor:
    def __init__(self, value: float):
        self.value = value

    def predict(self, x):
        return np.asarray([self.value])


class V59PredictionEnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module(
            ROOT / "A_track_v59_prediction_ensemble" / "submission.py",
            "v59_prediction_ensemble",
        )

    def model(self, with_aux: bool):
        model = self.module.RobustChargingModel.__new__(
            self.module.RobustChargingModel
        )
        model.predictors = {
            "caltech": {
                "demand_kwh": [FixedRegressor(10.0)] * 2,
                "dwell_hours": [FixedRegressor(4.0)] * 2,
            }
        }
        model.aux_predictors = (
            {
                "caltech": {
                    "demand_kwh": [FixedRegressor(20.0)],
                    "dwell_hours": [FixedRegressor(6.0)],
                }
            }
            if with_aux
            else {"caltech": {"demand_kwh": [], "dwell_hours": []}}
        )
        model.request_medians = {"caltech": 12.0}
        model.demand_caps = {"caltech": 150.0}
        model.dwell_caps = {"caltech": 36.0}
        model._prediction_cache = {}
        model._public_prediction_cache = {}
        model._demand_model_cache = {}
        model.fallback_labels = (10.0, 4.0)
        return model

    def features(self):
        return ArrivalFeatures(
            token="vehicle",
            site="caltech",
            hour=9.0,
            weekday=1,
            charger_limit_kw=6.6,
        )

    def test_stable_prediction_remains_base_model(self) -> None:
        model = self.model(with_aux=True)
        prediction = model._stable_predict(self.features())
        self.assertAlmostEqual(prediction.demand_kwh, 10.0)
        self.assertAlmostEqual(prediction.dwell_hours, 4.0)

    def test_public_prediction_blends_aux_models(self) -> None:
        model = self.model(with_aux=True)
        prediction = model.predict(self.features())
        self.assertAlmostEqual(prediction.demand_kwh, 15.0)
        self.assertAlmostEqual(prediction.dwell_hours, 5.0)

    def test_sites_without_aux_keep_stable_prediction(self) -> None:
        model = self.model(with_aux=False)
        prediction = model.predict(self.features())
        self.assertAlmostEqual(prediction.demand_kwh, 10.0)
        self.assertAlmostEqual(prediction.dwell_hours, 4.0)
        self.assertIsInstance(prediction, Prediction)

    def test_explain_documents_stable_planning_path(self) -> None:
        model = self.module.RobustChargingModel.__new__(
            self.module.RobustChargingModel
        )
        method = model.explain()["method"]
        self.assertIn("V59", method)
        self.assertIn("retain the stable", method)

    def test_world_model_and_dispatch_do_not_call_public_blend(self) -> None:
        source = (
            ROOT
            / "A_track_v59_prediction_ensemble"
            / "submission.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("self.predict(", source)
        self.assertGreaterEqual(source.count("self._stable_predict("), 4)


if __name__ == "__main__":
    unittest.main()
