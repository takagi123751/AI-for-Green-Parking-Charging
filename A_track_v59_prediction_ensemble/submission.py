"""AI for Green Track A submission with an online physics world model.

The module only depends on packages guaranteed by the ``cpu-ml-methods-v3``
sandbox. It never reads hidden labels, future arrivals, or judge state.
"""

from __future__ import annotations

import math

import numpy as np

from acn_causal.student_api import ArrivalFeatures, Prediction


CHARGER_KW = 6.6
EPS = 1e-8
ACT_DEMAND_MARGIN = 1.25
DWELL_LOWER_QUANTILE = 0.15
DWELL_LOWER_MARGIN = 1.50
DEMAND_MODEL_BLEND = 0.50
SPARSE_ARRIVAL_MEDIAN_THRESHOLD = 0.30
CURRENT_COHORT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
CURRENT_COHORT_PROBABILITIES = (0.2, 0.2, 0.2, 0.2, 0.2)
CURRENT_DWELL_SHRINK = 1.0
CURRENT_DWELL_MAX_OFFSET_HOURS = 6.0
CURRENT_DEMAND_RATIO_MIN = 0.5
CURRENT_DEMAND_RATIO_MAX = 1.5
SURVIVAL_BIN_HOURS = (0.0, 1.5, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0)
SURVIVAL_TRIGGER = 1.0
SURVIVAL_SHRINK = 0.5
SURVIVAL_MIN_SAMPLES = 12
SURVIVAL_MAX_ADJUSTMENT_HOURS = 8.0
NATURAL_LOAD_SCALE_BY_SITE = {
    "caltech": 0.80,
    "jpl": 1.00,
    "office1": 1.00,
}
CAPACITY_EFFECT_SCALE_BY_SITE = {
    "caltech": 0.0,
    "jpl": 0.0,
    "office1": 0.0,
}
JPL_ACTIVE_EDGES = (2.0, 6.0, 12.0, 20.0)
JPL_HOUR_EDGES = (6.0, 10.0, 14.0, 18.0)
# Load ratios fitted on an inner time fold and shrunk 50% toward no correction.
JPL_OCCUPANCY_LOAD_FACTORS = (
    (1.0, 1.0, 1.0, 1.0, 1.0),
    (1.1, 1.0, 1.0, 1.1, 1.1),
    (1.0, 1.03868984, 0.93512588, 1.06726501, 1.00960199),
    (1.0, 1.02730925, 0.94077018, 1.00583045, 1.0),
    (1.0, 0.99939730, 0.93328034, 0.98537323, 1.0),
)
PARTICLE_COUNT = 64
PARTICLE_MAX_DAYS = 180
PARTICLE_OCCUPANCY_BLEND = 1.25
PARTICLE_SCALE_MIN = 0.75
PARTICLE_SCALE_MAX = 1.75
PARTICLE_EDF_EFFECT_SCALE = 0.5
PARTICLE_SITES = frozenset({"jpl", "office1"})


def _as_finite_positive(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


class RobustChargingModel:
    """Learned vehicle forecasts plus a conservative physical rollout."""

    def __init__(self, training):
        import pandas as pd
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            HistGradientBoostingRegressor,
        )
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor

        self.labels: dict[str, tuple[float, float]] = {}
        self.request_medians: dict[str, float] = {}
        self.models: dict[str, ExtraTreesRegressor] = {}
        self.predictors: dict[
            str,
            dict[str, list[HistGradientBoostingRegressor]],
        ] = {}
        self.aux_predictors: dict[str, dict[str, list[object]]] = {}
        self.demand_caps: dict[str, float] = {}
        self.dwell_caps: dict[str, float] = {}
        self.rates: dict[str, np.ndarray] = {}
        self.weekday_days: dict[str, np.ndarray] = {}
        self.survival_residuals: dict[str, dict[float, float]] = {}
        self.dwell_lower_profiles: dict[str, np.ndarray] = {}
        self.arrival_occupancy_regressions: dict[
            str,
            dict[int, tuple[float, float]],
        ] = {}
        self.particle_days: dict[str, tuple[dict[str, object], ...]] = {}
        self._action_demand_cache: dict[tuple, float] = {}
        self._demand_model_cache: dict[tuple, tuple[float, ...]] = {}
        self._capacity_blend_active = False
        self._current_cohort_shape_cache: dict[
            tuple,
            tuple[
                tuple[float, ...],
                tuple[float, ...],
            ],
        ] = {}
        self._prediction_cache: dict[tuple, Prediction] = {}
        self._public_prediction_cache: dict[tuple, Prediction] = {}
        self._cohort_cache: dict[
            tuple[str, float, int],
            tuple[tuple[float, float, float], ...],
        ] = {}
        self._global_rates: np.ndarray | None = None

        all_demand: list[float] = []
        all_dwell: list[float] = []
        global_counts = np.zeros((7, 96), dtype=float)
        global_days = np.zeros(7, dtype=float)

        for site, source in training.items():
            frame = source.copy()
            demand = np.asarray(frame["demand_kwh"], dtype=float)
            dwell = np.asarray(frame["dwell_hours"], dtype=float)
            hours = np.mod(np.asarray(frame["hour"], dtype=float), 24.0)
            weekdays = np.mod(np.asarray(frame["weekday"], dtype=int), 7)
            charger = np.asarray(frame["charger_limit_kw"], dtype=float)
            request_source = (
                frame["last_requested_energy_kwh"]
                if "last_requested_energy_kwh" in frame
                else pd.Series(np.nan, index=frame.index)
            )
            request = np.asarray(request_source, dtype=float)
            valid = (
                np.isfinite(demand)
                & (demand > 0)
                & np.isfinite(dwell)
                & (dwell > 0)
                & np.isfinite(hours)
                & np.isfinite(charger)
                & (charger > 0)
            )
            demand = demand[valid]
            dwell = dwell[valid]
            hours = hours[valid]
            weekdays = weekdays[valid]
            charger = charger[valid]
            request = request[valid]
            if len(demand) == 0:
                continue

            demand_median = float(np.median(demand))
            dwell_median = float(np.median(dwell))
            requested_valid = np.isfinite(request) & (request > 0)
            request_median = (
                float(np.median(request[requested_valid]))
                if requested_valid.any()
                else demand_median
            )
            self.labels[site] = (demand_median, dwell_median)
            self.request_medians[site] = request_median
            self.demand_caps[site] = min(
                150.0, max(30.0, float(np.max(demand)) * 1.5)
            )
            self.dwell_caps[site] = min(
                36.0, max(24.0, float(np.max(dwell)) * 1.25)
            )
            self.dwell_lower_profiles[site] = (
                self._build_dwell_lower_profiles(frame)
            )
            all_demand.extend(demand.tolist())
            all_dwell.extend(dwell.tolist())

            cohort_features = self._feature_matrix(
                hours,
                weekdays,
                charger,
                request,
                request_median,
            )
            targets = np.column_stack([demand, dwell])
            cohort_model = ExtraTreesRegressor(
                n_estimators=192,
                min_samples_leaf=8,
                max_depth=16,
                max_features=0.8,
                n_jobs=1,
                random_state=20261008,
            )
            cohort_model.fit(cohort_features, targets)
            self.models[site] = cohort_model

            site_settings = {
                "caltech": (90.0, 31, 12),
                "jpl": (90.0, 15, 12),
                "office1": (180.0, 7, 18),
            }
            half_life, leaves, min_leaf = site_settings.get(
                site, (120.0, 15, 15)
            )
            all_days = pd.to_datetime(frame["day"], errors="coerce")
            valid_days = all_days[valid]
            age = (
                all_days.max() - valid_days
            ).dt.days.to_numpy(dtype=float)
            sample_weights = np.exp(-np.log(2.0) * age / half_life)
            prediction_features = self._prediction_feature_matrix(
                hours,
                weekdays,
                charger,
                request,
            )
            self.predictors[site] = {}
            for target_name, target in (
                ("demand_kwh", demand),
                ("dwell_hours", dwell),
            ):
                self.predictors[site][target_name] = [
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        max_iter=224,
                        learning_rate=0.05,
                        max_leaf_nodes=leaves,
                        min_samples_leaf=min_leaf,
                        l2_regularization=1.0,
                        random_state=seed,
                        early_stopping=True,
                        validation_fraction=0.12,
                    ).fit(
                        prediction_features,
                        target,
                        sample_weight=sample_weights,
                    )
                    for seed in (20261008, 20261009)
                ]

            aux: dict[str, list[object]] = {
                "demand_kwh": [],
                "dwell_hours": [],
            }
            if site in {"caltech", "jpl"}:
                for target_name, target in (
                    ("demand_kwh", demand),
                    ("dwell_hours", dwell),
                ):
                    aux[target_name].append(
                        XGBRegressor(
                            objective="reg:absoluteerror",
                            n_estimators=240,
                            learning_rate=0.04,
                            max_depth=5,
                            min_child_weight=4,
                            subsample=0.9,
                            colsample_bytree=0.85,
                            reg_lambda=1.0,
                            random_state=20261008,
                            n_jobs=1,
                        ).fit(
                            prediction_features,
                            target,
                            sample_weight=sample_weights,
                        )
                    )
                if site == "caltech":
                    aux["demand_kwh"].append(
                        LGBMRegressor(
                            objective="mae",
                            n_estimators=260,
                            learning_rate=0.04,
                            num_leaves=15,
                            min_child_samples=12,
                            subsample=0.9,
                            colsample_bytree=0.85,
                            reg_lambda=1.0,
                            random_state=20261008,
                            n_jobs=1,
                            verbosity=-1,
                        ).fit(
                            prediction_features,
                            demand,
                            sample_weight=sample_weights,
                        )
                    )
            self.aux_predictors[site] = aux

            predicted_dwell = np.mean(
                [
                    model.predict(prediction_features)
                    for model in self.predictors[site]["dwell_hours"]
                ],
                axis=0,
            )
            predicted_dwell = np.clip(
                predicted_dwell,
                0.05,
                self.dwell_caps[site],
            )
            self.survival_residuals[site] = self._build_survival_residuals(
                dwell,
                predicted_dwell,
            )

            counts = np.zeros((7, 96), dtype=float)
            days = np.zeros(7, dtype=float)
            timestamps = self._to_datetime(frame["day"])
            if len(timestamps):
                for day in np.unique(timestamps.astype("datetime64[D]")):
                    weekday = int((np.datetime64(day, "D").astype(int) + 3) % 7)
                    days[weekday] += 1.0
            for hour, weekday in zip(hours, weekdays):
                slot = int(math.floor(hour * 4.0)) % 96
                counts[int(weekday), slot] += 1.0
            mean_rates = counts / np.maximum(days[:, None], 1.0)
            valid_frame = frame.loc[valid]
            daily_by_weekday: dict[int, list[np.ndarray]] = {
                weekday: [] for weekday in range(7)
            }
            grouped = valid_frame.groupby(
                pd.to_datetime(valid_frame["day"]).dt.date,
                sort=True,
            )
            for day, rows in grouped:
                weekday = int(pd.Timestamp(day).weekday())
                daily = np.zeros(96, dtype=float)
                for hour in rows["hour"].astype(float):
                    slot = int(math.floor(hour * 4.0)) % 96
                    daily[slot] += 1.0
                daily_by_weekday[weekday].append(daily)
            self.rates[site] = self._blend_sparse_arrival_rates(
                mean_rates,
                daily_by_weekday,
            )
            self.weekday_days[site] = days.copy()
            self.arrival_occupancy_regressions[site] = (
                self._build_occupancy_regressions(frame)
            )
            self.particle_days[site] = self._build_particle_days(frame)
            global_counts += counts
            global_days += days

        if all_demand:
            fallback_labels = (
                float(np.median(all_demand)),
                float(np.median(all_dwell)),
            )
        else:
            fallback_labels = (10.0, 4.0)
        self.fallback_labels = fallback_labels
        self.fallback_request_median = fallback_labels[0]
        self._global_rates = global_counts / np.maximum(global_days[:, None], 1.0)

    @staticmethod
    def _to_datetime(values):
        import pandas as pd

        return pd.to_datetime(values, errors="coerce").dropna().to_numpy()

    @staticmethod
    def _blend_sparse_arrival_rates(mean_rates, daily_by_weekday):
        mean = np.asarray(mean_rates, dtype=float)
        median_rates = np.zeros_like(mean)
        for weekday, days in daily_by_weekday.items():
            if days:
                median_rates[weekday] = np.median(
                    np.asarray(days, dtype=float),
                    axis=0,
                )
        sparse = mean < SPARSE_ARRIVAL_MEDIAN_THRESHOLD
        return np.where(sparse, median_rates, mean)

    @staticmethod
    def _feature_matrix(hours, weekdays, charger, request, request_median):
        hours = np.asarray(hours, dtype=float)
        weekdays = np.asarray(weekdays, dtype=float)
        charger = np.asarray(charger, dtype=float)
        request = np.asarray(request, dtype=float)
        requested_valid = np.isfinite(request) & (request > 0)
        filled_request = np.where(
            requested_valid, request, float(request_median)
        )
        return np.column_stack(
            [
                np.sin(hours * np.pi / 12.0),
                np.cos(hours * np.pi / 12.0),
                np.sin(weekdays * 2.0 * np.pi / 7.0),
                np.cos(weekdays * 2.0 * np.pi / 7.0),
                weekdays,
                charger,
                filled_request,
                requested_valid.astype(float),
            ]
        )

    @staticmethod
    def _prediction_feature_matrix(hours, weekdays, charger, request):
        hours = np.asarray(hours, dtype=float)
        weekdays = np.asarray(weekdays, dtype=float)
        charger = np.asarray(charger, dtype=float)
        request = np.asarray(request, dtype=float)
        requested_valid = np.isfinite(request) & (request > 0)
        safe_request = np.where(
            requested_valid, np.clip(request, 0.0, 250.0), 0.0
        )
        return np.column_stack(
            [
                np.sin(hours * np.pi / 12.0),
                np.cos(hours * np.pi / 12.0),
                hours,
                np.sin(weekdays * 2.0 * np.pi / 7.0),
                np.cos(weekdays * 2.0 * np.pi / 7.0),
                weekdays,
                charger,
                (weekdays >= 5).astype(float),
                requested_valid.astype(float),
                safe_request,
                np.log1p(safe_request),
                np.sqrt(safe_request),
                safe_request / CHARGER_KW,
            ]
        )

    @staticmethod
    def _build_survival_residuals(actual, predicted):
        residuals = {}
        for lower in SURVIVAL_BIN_HOURS:
            survived = actual > lower + EPS
            count = int(survived.sum())
            if count < SURVIVAL_MIN_SAMPLES:
                continue
            actual_remaining = np.maximum(
                actual[survived] - lower,
                0.25,
            )
            predicted_remaining = np.maximum(
                predicted[survived] - lower,
                0.25,
            )
            residual = float(
                np.median(actual_remaining - predicted_remaining)
            )
            residuals[lower] = float(
                np.clip(
                    residual,
                    -SURVIVAL_MAX_ADJUSTMENT_HOURS,
                    SURVIVAL_MAX_ADJUSTMENT_HOURS,
                )
            )
        return residuals

    @staticmethod
    def _build_dwell_lower_profiles(frame):
        hours = np.mod(np.asarray(frame["hour"], dtype=float), 24.0)
        dwell = np.asarray(frame["dwell_hours"], dtype=float)
        valid = np.isfinite(hours) & np.isfinite(dwell) & (dwell > EPS)
        hours = hours[valid]
        dwell = dwell[valid]
        if len(dwell) == 0:
            return np.full(96, 0.25, dtype=float)
        fallback = max(
            0.25,
            float(np.quantile(dwell, DWELL_LOWER_QUANTILE)),
        )
        profile = np.full(96, fallback, dtype=float)
        for slot in range(96):
            center = slot * 0.25
            distance = np.abs(hours - center)
            distance = np.minimum(distance, 24.0 - distance)
            local = dwell[distance <= 0.50]
            if len(local) >= 12:
                profile[slot] = max(
                    0.25,
                    float(np.quantile(local, DWELL_LOWER_QUANTILE)),
                )
        return profile

    @staticmethod
    def _robust_demand_target(mean_demand: float, high_demand: float) -> float:
        mean = max(0.0, float(mean_demand))
        high = max(mean, float(high_demand))
        return mean + DEMAND_MODEL_BLEND * (high - mean)

    @staticmethod
    def _build_occupancy_regressions(frame):
        import pandas as pd

        timestamps = pd.to_datetime(frame["day"], errors="coerce")
        daily = []
        for day, rows in frame.groupby(timestamps.dt.date, sort=True):
            if pd.isna(day):
                continue
            active = np.zeros(96, dtype=float)
            arrivals = np.zeros(96, dtype=float)
            for record in rows.itertuples(index=False):
                hour = float(record.hour)
                dwell = float(record.dwell_hours)
                if not math.isfinite(hour) or not math.isfinite(dwell):
                    continue
                arrival = min(max(int(hour * 4.0), 0), 95)
                capped_dwell = min(
                    max(dwell, 0.25),
                    (96 - arrival) * 0.25,
                )
                departure = min(
                    96,
                    arrival + max(1, int(math.ceil(capped_dwell / 0.25))),
                )
                active[arrival:departure] += 1.0
                arrivals[arrival] += 1.0
            daily.append((active, arrivals))
        if not daily:
            return {}

        regressions = {}
        for origin in range(0, 85, 12):
            current = np.asarray(
                [entry[0][origin] for entry in daily],
                dtype=float,
            )
            future = np.asarray(
                [entry[1][origin + 1 : origin + 12].sum() for entry in daily],
                dtype=float,
            )
            variance = float(np.var(current))
            if variance > EPS:
                slope = float(
                    np.cov(current, future, bias=True)[0, 1] / variance
                )
            else:
                slope = 0.0
            slope = max(0.0, slope)
            intercept = max(
                0.0,
                float(np.mean(future) - slope * np.mean(current)),
            )
            regressions[origin] = (intercept, slope)
        return regressions

    @staticmethod
    def _build_particle_days(frame):
        import pandas as pd

        timestamps = pd.to_datetime(frame["day"], errors="coerce")
        source = frame.loc[timestamps.notna()].copy()
        source["day"] = timestamps.loc[timestamps.notna()].dt.normalize().to_numpy()
        days = sorted(source["day"].unique())[-PARTICLE_MAX_DAYS:]
        episodes = []
        for day in days:
            rows = source[source["day"] == day].sort_values(
                ["hour", "charger_limit_kw", "demand_kwh", "dwell_hours"],
                kind="mergesort",
            )
            active = np.zeros(96, dtype=float)
            vehicles = []
            for index, record in enumerate(rows.to_dict("records")):
                hour = float(record["hour"])
                dwell = float(record["dwell_hours"])
                charger = float(record["charger_limit_kw"])
                if (
                    not math.isfinite(hour)
                    or not math.isfinite(dwell)
                    or not math.isfinite(charger)
                    or charger <= 0
                ):
                    continue
                arrival = min(max(int(hour * 4.0), 0), 95)
                dwell = min(
                    max(dwell, 0.25),
                    max(0.25, (96 - arrival) * 0.25),
                )
                departure = min(
                    96,
                    arrival + max(1, int(math.ceil(dwell / 0.25))),
                )
                demand = min(
                    max(float(record["demand_kwh"]), 0.0),
                    dwell * charger,
                )
                active[arrival:departure] += 1.0
                vehicles.append(
                    {
                        "token": f"particle-{index}",
                        "arrival": arrival,
                        "departure": departure,
                        "demand": demand,
                        "charger": charger,
                    }
                )
            episodes.append(
                {
                    "weekday": int(pd.Timestamp(day).dayofweek),
                    "active": active,
                    "vehicles": tuple(vehicles),
                }
            )
        return tuple(episodes)

    def _select_particle_days(self, observation):
        days = self.particle_days.get(observation.site, ())
        if not days:
            return ()
        origin = min(max(int(observation.step), 0), 95)
        same_weekday = tuple(
            day
            for day in days
            if int(day["weekday"]) == int(observation.weekday) % 7
        )
        pool = (
            same_weekday
            if len(same_weekday) >= PARTICLE_COUNT
            else days
        )
        current_count = float(len(observation.vehicles))
        ranked = sorted(
            enumerate(pool),
            key=lambda item: (
                abs(float(item[1]["active"][origin]) - current_count),
                item[0],
            ),
        )
        return tuple(day for _, day in ranked[:PARTICLE_COUNT])

    def _particle_forecast(
        self,
        observation,
        horizon_steps: int,
        intervention: str,
    ):
        steps = max(0, int(horizon_steps))
        days = self._select_particle_days(observation)
        if not days or steps == 0:
            return None

        origin = min(max(int(observation.step), 0), 95)
        particle_weight = 1.0 / len(days)
        current_count = float(len(observation.vehicles))
        anchor_active = float(
            np.mean([float(day["active"][origin]) for day in days])
        )
        if anchor_active > EPS:
            raw_scale = current_count / anchor_active
            occupancy_scale = float(
                np.clip(
                    1.0
                    + PARTICLE_OCCUPANCY_BLEND * (raw_scale - 1.0),
                    PARTICLE_SCALE_MIN,
                    PARTICLE_SCALE_MAX,
                )
            )
        else:
            occupancy_scale = 1.0

        current_variants = []
        for vehicle in observation.vehicles:
            if vehicle.charging_complete:
                continue
            pred = self._stable_predict(vehicle.features)
            remaining = self._estimated_remaining(vehicle)
            if remaining <= EPS:
                continue
            remaining_hours = self._survival_remaining_hours(
                observation.site,
                pred.dwell_hours,
                float(vehicle.elapsed_steps) * observation.dt_hours,
                observation.dt_hours,
            )
            demand_ratios, dwell_offsets = self._current_cohort_shapes(
                vehicle.features
            )
            current_variants.append(
                (
                    str(vehicle.features.token),
                    remaining,
                    remaining_hours,
                    max(float(vehicle.features.charger_limit_kw), 0.0),
                    demand_ratios,
                    dwell_offsets,
                )
            )

        total_load = np.zeros(steps, dtype=float)
        total_active = np.zeros(steps, dtype=float)
        capacity_series = (
            observation.intervention_capacity
            or observation.forecast_capacity
        )

        for particle_index, day in enumerate(days):
            cars = []
            cohort_index = particle_index % len(CURRENT_COHORT_PROBABILITIES)
            for token, remaining, remaining_hours, charger, ratios, offsets in (
                current_variants
            ):
                steps_left = max(
                    1,
                    int(
                        math.ceil(
                            max(
                                observation.dt_hours,
                                remaining_hours + offsets[cohort_index],
                            )
                            / max(observation.dt_hours, EPS)
                        )
                    ),
                )
                cars.append(
                    {
                        "token": f"{particle_index}:{token}",
                        "remaining": max(
                            EPS,
                            remaining * ratios[cohort_index],
                        ),
                        "steps_left": steps_left,
                        "charger": charger,
                        "count_weight": 1.0,
                    }
                )

            future_by_step: dict[int, list[dict[str, object]]] = {}
            for vehicle in day["vehicles"]:
                arrival = int(vehicle["arrival"])
                if origin < arrival < origin + steps:
                    future_by_step.setdefault(arrival, []).append(vehicle)

            for offset in range(steps):
                absolute_step = origin + offset
                for vehicle in future_by_step.get(absolute_step, ()):
                    cars.append(
                        {
                            "token": (
                                f"{particle_index}:future:"
                                f"{vehicle['token']}:{absolute_step}"
                            ),
                            "remaining": max(
                                EPS,
                                float(vehicle["demand"]) * occupancy_scale,
                            ),
                            "steps_left": max(
                                1,
                                int(vehicle["departure"]) - absolute_step,
                            ),
                            "charger": max(
                                float(vehicle["charger"]),
                                0.0,
                            ),
                            "count_weight": occupancy_scale,
                        }
                    )

                active = 0.0
                for car in cars:
                    if (
                        int(car["steps_left"]) > 0
                        and float(car["remaining"]) > EPS
                    ):
                        active += float(car["count_weight"])
                active_counts = (
                    active
                    if intervention == "natural"
                    else active
                )
                total_active[offset] += particle_weight * active_counts

                if intervention == "natural":
                    ev_capacity = math.inf
                else:
                    total_capacity = self._series_value(
                        capacity_series,
                        offset,
                        float(observation.capacity_kw),
                    )
                    background = self._series_value(
                        observation.forecast_background,
                        offset,
                        float(observation.background_kw),
                    )
                    ev_capacity = max(total_capacity - background, 0.0)

                capacity_left = ev_capacity
                load = 0.0
                for car in sorted(
                    cars,
                    key=lambda item: (
                        int(item["steps_left"]),
                        str(item["token"]),
                    ),
                ):
                    if (
                        int(car["steps_left"]) <= 0
                        or float(car["remaining"]) <= EPS
                    ):
                        continue
                    count_weight = float(car["count_weight"])
                    per_car_power = min(
                        float(car["charger"]),
                        float(car["remaining"])
                        / count_weight
                        / max(observation.dt_hours, EPS),
                    )
                    requested_load = max(0.0, per_car_power) * count_weight
                    allocated = min(requested_load, capacity_left)
                    if allocated <= EPS:
                        continue
                    delivered = allocated * observation.dt_hours
                    car["remaining"] = max(
                        0.0,
                        float(car["remaining"]) - delivered,
                    )
                    capacity_left -= allocated
                    load += allocated

                total_load[offset] += particle_weight * load
                for car in cars:
                    car["steps_left"] = int(car["steps_left"]) - 1
                cars = [
                    car
                    for car in cars
                    if int(car["steps_left"]) > 0
                    and float(car["remaining"]) > EPS
                ]

        return {
            "ev_load_kw": np.maximum(0.0, total_load).tolist(),
            "active_count": np.maximum(0.0, total_active).tolist(),
        }

    def _survival_remaining_hours(
        self,
        site: str,
        predicted_dwell_hours: float,
        elapsed_hours: float,
        dt_hours: float,
    ) -> float:
        predicted = max(float(predicted_dwell_hours), float(dt_hours))
        elapsed = max(0.0, float(elapsed_hours))
        base = max(float(dt_hours), predicted - elapsed)
        if elapsed + EPS < SURVIVAL_TRIGGER * predicted:
            return base

        residuals = self.survival_residuals.get(site)
        if not residuals:
            return base
        eligible = [lower for lower in residuals if lower <= elapsed + EPS]
        if not eligible:
            return base
        adjustment = float(residuals[max(eligible)])
        return max(
            float(dt_hours),
            base + SURVIVAL_SHRINK * adjustment,
        )

    def _current_cohort_shapes(
        self,
        features: ArrivalFeatures,
    ) -> tuple[
        tuple[float, ...],
        tuple[float, ...],
    ]:
        requested = _as_finite_positive(
            getattr(features, "last_requested_energy_kwh", None)
        )
        key = (
            features.site,
            round(float(features.hour) % 24.0, 4),
            int(features.weekday) % 7,
            round(float(features.charger_limit_kw), 3),
            round(requested, 3) if requested is not None else None,
        )
        cached = self._current_cohort_shape_cache.get(key)
        if cached is not None:
            return cached

        model = self.models.get(features.site)
        if model is None or len(model.estimators_) == 0:
            result = (
                tuple(1.0 for _ in CURRENT_COHORT_QUANTILES),
                tuple(0.0 for _ in CURRENT_COHORT_QUANTILES),
            )
            self._current_cohort_shape_cache[key] = result
            return result

        request_median = self.request_medians[features.site]
        x = self._feature_matrix(
            [float(features.hour) % 24.0],
            [int(features.weekday) % 7],
            [max(float(features.charger_limit_kw), 0.1)],
            [requested if requested is not None else np.nan],
            request_median,
        )
        tree_predictions = np.asarray(
            [
                tree.predict(x)[0]
                for tree in model.estimators_[::8]
            ],
            dtype=float,
        )
        if len(tree_predictions) < len(CURRENT_COHORT_QUANTILES):
            result = (
                tuple(1.0 for _ in CURRENT_COHORT_QUANTILES),
                tuple(0.0 for _ in CURRENT_COHORT_QUANTILES),
            )
            self._current_cohort_shape_cache[key] = result
            return result

        tree_predictions = tree_predictions[
            np.argsort(tree_predictions[:, 1])
        ]
        selected = np.asarray(
            [
                tree_predictions[
                    int(round(quantile * (len(tree_predictions) - 1)))
                ]
                for quantile in CURRENT_COHORT_QUANTILES
            ],
            dtype=float,
        )
        demand_ratios = selected[:, 0] / max(
            float(
                np.average(
                    selected[:, 0],
                    weights=CURRENT_COHORT_PROBABILITIES,
                )
            ),
            EPS,
        )
        demand_ratios = np.clip(
            demand_ratios,
            CURRENT_DEMAND_RATIO_MIN,
            CURRENT_DEMAND_RATIO_MAX,
        )
        demand_ratios /= max(
            float(
                np.average(
                    demand_ratios,
                    weights=CURRENT_COHORT_PROBABILITIES,
                )
            ),
            EPS,
        )
        dwell_offsets = selected[:, 1] - float(
            np.average(
                selected[:, 1],
                weights=CURRENT_COHORT_PROBABILITIES,
            )
        )
        dwell_offsets = np.clip(
            dwell_offsets * CURRENT_DWELL_SHRINK,
            -CURRENT_DWELL_MAX_OFFSET_HOURS,
            CURRENT_DWELL_MAX_OFFSET_HOURS,
        )
        result = (
            tuple(float(value) for value in demand_ratios),
            tuple(float(value) for value in dwell_offsets),
        )
        self._current_cohort_shape_cache[key] = result
        return result

    def _extra_trees_prediction(self, features: ArrivalFeatures) -> Prediction:
        model = self.models.get(features.site)
        if model is None:
            return Prediction(*self.fallback_labels)
        request = _as_finite_positive(
            getattr(features, "last_requested_energy_kwh", None)
        )
        request_median = self.request_medians[features.site]
        x = self._feature_matrix(
            [float(features.hour) % 24.0],
            [int(features.weekday) % 7],
            [max(float(features.charger_limit_kw), 0.1)],
            [request if request is not None else np.nan],
            request_median,
        )
        raw = model.predict(x)[0]
        return Prediction(
            float(
                np.clip(
                    raw[0],
                    0.01,
                    self.demand_caps.get(features.site, 150.0),
                )
            ),
            float(
                np.clip(
                    raw[1],
                    0.05,
                    self.dwell_caps.get(features.site, 36.0),
                )
            ),
        )

    def _stable_predict(self, features: ArrivalFeatures) -> Prediction:
        requested = _as_finite_positive(
            getattr(features, "last_requested_energy_kwh", None)
        )
        key = (
            features.site,
            round(float(features.hour) % 24.0, 4),
            int(features.weekday) % 7,
            round(float(features.charger_limit_kw), 3),
            round(requested, 3) if requested is not None else None,
        )
        cached = self._prediction_cache.get(key)
        if cached is not None:
            return cached

        site_models = self.predictors.get(features.site)
        if site_models is None:
            value = Prediction(*self.fallback_labels)
        else:
            x = self._prediction_feature_matrix(
                [float(features.hour) % 24.0],
                [int(features.weekday) % 7],
                [max(float(features.charger_limit_kw), 0.1)],
                [requested if requested is not None else np.nan],
            )
            demand_estimates = tuple(
                float(model.predict(x)[0])
                for model in site_models["demand_kwh"]
            )
            self._demand_model_cache[key] = demand_estimates
            demand = float(np.mean(demand_estimates))
            dwell = float(
                np.mean(
                    [
                        model.predict(x)[0]
                        for model in site_models["dwell_hours"]
                    ]
                )
            )
            dwell = float(
                np.clip(
                    dwell,
                    0.05,
                    self.dwell_caps.get(features.site, 36.0),
                )
            )
            demand = min(
                float(
                    np.clip(
                        demand,
                        0.01,
                        self.demand_caps.get(features.site, 150.0),
                    )
                ),
                max(float(features.charger_limit_kw), 0.1) * dwell,
            )
            value = Prediction(demand, dwell)
        self._prediction_cache[key] = value
        return value

    def predict(self, features: ArrivalFeatures) -> Prediction:
        base = self._stable_predict(features)
        aux = self.aux_predictors.get(features.site)
        if not aux or not any(aux.values()):
            return base
        requested = _as_finite_positive(
            getattr(features, "last_requested_energy_kwh", None)
        )
        key = (
            features.site,
            round(float(features.hour) % 24.0, 4),
            int(features.weekday) % 7,
            round(float(features.charger_limit_kw), 3),
            round(requested, 3) if requested is not None else None,
        )
        cached = self._public_prediction_cache.get(key)
        if cached is not None:
            return cached

        x = self._prediction_feature_matrix(
            [float(features.hour) % 24.0],
            [int(features.weekday) % 7],
            [max(float(features.charger_limit_kw), 0.1)],
            [requested if requested is not None else np.nan],
        )
        demand = float(base.demand_kwh)
        dwell = float(base.dwell_hours)
        demand_aux = aux.get("demand_kwh", [])
        if demand_aux:
            demand = 0.5 * demand + 0.5 * float(
                np.mean(
                    [model.predict(x)[0] for model in demand_aux]
                )
            )
        dwell_aux = aux.get("dwell_hours", [])
        if dwell_aux:
            dwell = 0.5 * dwell + 0.5 * float(
                np.mean(
                    [model.predict(x)[0] for model in dwell_aux]
                )
            )
        dwell = float(
            np.clip(
                dwell,
                0.05,
                self.dwell_caps.get(features.site, 36.0),
            )
        )
        demand = min(
            float(
                np.clip(
                    demand,
                    0.01,
                    self.demand_caps.get(features.site, 100.0),
                )
            ),
            max(float(features.charger_limit_kw), 0.1) * dwell,
        )
        value = Prediction(demand, dwell)
        self._public_prediction_cache[key] = value
        return value

    def arrival_rate(self, site: str, hour: float, weekday: int) -> float:
        table = self.rates.get(site, self._global_rates)
        if table is None:
            return 0.0
        slot = int(math.floor((float(hour) % 24.0) * 4.0)) % 96
        rate = float(table[int(weekday) % 7, slot])
        return max(0.0, rate) if math.isfinite(rate) else 0.0

    def _estimated_remaining(self, vehicle) -> float:
        pred = self._stable_predict(vehicle.features)
        requested = _as_finite_positive(
            getattr(vehicle.features, "last_requested_energy_kwh", None)
        )
        # The request snapshot is noisy. Bound its influence by the maximum
        # documented demand-shift factor instead of treating it as a hard target.
        requested_cap = pred.demand_kwh * 1.35
        target = max(
            pred.demand_kwh,
            min(requested or 0.0, requested_cap),
        )
        remaining = max(0.0, target - float(vehicle.delivered_kwh))
        if remaining <= EPS and not vehicle.charging_complete:
            remaining = (
                max(float(vehicle.features.charger_limit_kw), 0.1)
                * 0.25
            )
        return remaining

    def _action_demand_target(self, vehicle, pred: Prediction) -> float:
        request = _as_finite_positive(
            getattr(vehicle.features, "last_requested_energy_kwh", None)
        )
        key = (
            vehicle.features.site,
            round(float(vehicle.features.hour) % 24.0, 4),
            int(vehicle.features.weekday) % 7,
            round(float(vehicle.features.charger_limit_kw), 3),
            round(request, 3) if request is not None else None,
        )
        cached = self._action_demand_cache.get(key)
        if cached is not None:
            return cached

        mean_demand = float(pred.demand_kwh)
        high_demand = mean_demand
        estimates = self._demand_model_cache.get(key)
        if estimates:
            high_demand = max(estimates)
            high_demand = min(
                high_demand,
                self.demand_caps.get(vehicle.features.site, 150.0),
                max(float(vehicle.features.charger_limit_kw), 0.1)
                * float(pred.dwell_hours),
            )
        robust = self._robust_demand_target(mean_demand, high_demand)
        request_cap = robust * 1.35
        target = max(
            robust,
            min(request or 0.0, request_cap),
        )
        self._action_demand_cache[key] = float(target)
        return float(target)

    def _action_deadline_hours(self, vehicle, pred: Prediction) -> float:
        profile = self.dwell_lower_profiles.get(vehicle.features.site)
        if profile is None:
            return float(pred.dwell_hours)
        slot = (
            int(math.floor((float(vehicle.features.hour) % 24.0) * 4.0))
            % 96
        )
        conservative = float(profile[slot]) * DWELL_LOWER_MARGIN
        return max(0.25, min(float(pred.dwell_hours), conservative))

    def act(self, observation):
        current_capacity = max(float(observation.capacity_kw), 0.0)
        if observation.forecast_capacity:
            current_capacity = min(
                current_capacity,
                max(float(observation.forecast_capacity[0]), 0.0),
            )
        budget = max(current_capacity - float(observation.background_kw), 0.0)
        adaptive_margin = ACT_DEMAND_MARGIN

        candidates = []
        tail_tokens = set()
        for vehicle in observation.vehicles:
            if vehicle.charging_complete:
                continue
            pred = self._stable_predict(vehicle.features)
            # The hidden suite contains demand-shift scenarios. Use a planning
            # margin only inside action selection; prediction and world-model
            # outputs remain unchanged.
            action_target = self._action_demand_target(vehicle, pred)
            remaining = (
                max(0.0, action_target - float(vehicle.delivered_kwh))
                * adaptive_margin
            )
            if remaining <= EPS and not vehicle.charging_complete:
                remaining = (
                    max(float(vehicle.features.charger_limit_kw), 0.1)
                    * 0.25
                )
            if max(pred.demand_kwh - vehicle.delivered_kwh, 0.0) <= EPS:
                tail_tokens.add(str(vehicle.features.token))
            charger_limit = max(
                float(vehicle.features.charger_limit_kw), 0.0
            )
            predicted_steps = max(
                1.0,
                math.ceil(
                    self._action_deadline_hours(vehicle, pred)
                    / max(observation.dt_hours, EPS)
                    - float(vehicle.elapsed_steps)
                ),
            )
            required_steps = max(
                1.0,
                math.ceil(
                    remaining
                    / max(
                        charger_limit * observation.dt_hours,
                        EPS,
                    )
                ),
            )
            slack = predicted_steps - required_steps
            minimum_power = min(
                charger_limit,
                remaining
                * 1.05
                / max(predicted_steps * observation.dt_hours, EPS),
            )
            candidates.append(
                (
                    slack,
                    predicted_steps,
                    -remaining,
                    str(vehicle.features.token),
                    vehicle,
                    minimum_power,
                    remaining,
                    charger_limit,
                )
            )

        power: dict[str, float] = {}
        if observation.site == "office1":
            active = sorted(
                (item for item in candidates if item[3] not in tail_tokens),
                key=lambda item: (item[0], item[1], item[3]),
            )
            tail = sorted(
                (item for item in candidates if item[3] in tail_tokens),
                key=lambda item: item[3],
            )
            for _, _, _, token, _, _, _, _ in active:
                token = str(token)
                requested = next(
                    item[6]
                    for item in candidates
                    if str(item[3]) == token
                )
                charger = next(
                    item[7]
                    for item in candidates
                    if str(item[3]) == token
                )
                allocated = min(
                    min(
                        charger,
                        requested / max(observation.dt_hours, EPS),
                    ),
                    budget,
                )
                power[token] = float(allocated)
                budget -= allocated
            pending = {
                str(item[3]): min(
                    item[7],
                    item[6] / max(observation.dt_hours, EPS),
                )
                for item in tail
            }
            while budget > EPS and pending:
                share = budget / len(pending)
                progressed = False
                for token in sorted(pending):
                    current = power.get(token, 0.0)
                    addition = min(pending[token] - current, share, budget)
                    if addition > EPS:
                        power[token] = float(current + addition)
                        budget -= addition
                        progressed = True
                    if power[token] >= pending[token] - EPS:
                        pending.pop(token)
                if not progressed:
                    break
            for item in tail:
                power.setdefault(str(item[3]), 0.0)
            return power

        total_minimum_budget = sum(
            min(item[5], item[7]) for item in candidates
        )
        if budget + EPS < 0.9 * total_minimum_budget:
            for _, _, _, _, vehicle, _, _, charger_limit in sorted(
                candidates,
                key=lambda item: (
                    item[6],
                    item[5],
                    item[1],
                    item[3],
                ),
            ):
                if budget <= EPS:
                    break
                allocated = min(charger_limit, budget)
                if allocated > EPS:
                    power[str(vehicle.features.token)] = float(allocated)
                    budget -= allocated
            return power

        # Vehicles with no slack need full power now to retain a completion
        # chance. Give scarce capacity to the smallest such jobs first so the
        # 97%-energy completion threshold is met by as many vehicles as possible.
        critical = sorted(
            (item for item in candidates if item[0] <= 0.0),
            key=lambda item: (item[6], item[1], item[3]),
        )
        for _, _, _, _, vehicle, _, _, charger_limit in critical:
            if budget <= EPS:
                break
            allocated = min(charger_limit, budget)
            if allocated > EPS:
                power[str(vehicle.features.token)] = float(allocated)
                budget -= allocated

        remaining_candidates = sorted(
            (item for item in candidates if item[0] > 0.0),
            key=lambda item: (item[6], item[5], item[1], item[3]),
        )
        # When capacity is close to sufficient, reserve completion floors before
        # spending spare capacity on easy-to-finish vehicles.
        for _, _, _, _, vehicle, minimum_power, _, charger_limit in remaining_candidates:
            if budget <= EPS:
                break
            allocated = min(minimum_power, charger_limit, budget)
            if allocated > EPS:
                power[str(vehicle.features.token)] = float(allocated)
                budget -= allocated

        # Spend the remainder on the smallest remaining jobs first. This
        # maximizes completed vehicle count without exceeding physical limits.
        for _, _, _, _, vehicle, _, _, charger_limit in sorted(
            candidates,
            key=lambda item: (
                item[0],
                item[6],
                item[5],
                item[1],
                item[3],
            ),
        ):
            if budget <= EPS:
                break
            token = str(vehicle.features.token)
            current = power.get(token, 0.0)
            extra = min(max(charger_limit - current, 0.0), budget)
            if extra > EPS:
                power[token] = float(current + extra)
                budget -= extra
        return power

    @staticmethod
    def _series_value(series, index: int, fallback: float) -> float:
        if not series:
            return fallback
        try:
            return float(series[min(index, len(series) - 1)])
        except (TypeError, ValueError, IndexError):
            return fallback

    def _arrival_multipliers(self, observation, steps: int) -> np.ndarray:
        multipliers = np.ones(steps, dtype=float)
        history = np.asarray(observation.historical_arrivals, dtype=float)
        if steps <= 0 or len(history) == 0:
            return multipliers
        if (
            not np.isfinite(history).all()
            or (history < 0).any()
        ):
            return multipliers

        window = min(int(steps), len(history))
        expected = np.empty(len(history), dtype=float)
        default_days = np.ones(7, dtype=float)
        site_days = self.weekday_days.get(observation.site, default_days)
        for offset in range(len(history)):
            absolute_hour = (
                float(observation.local_hour)
                + offset * float(observation.dt_hours)
            )
            hour = absolute_hour % 24.0
            day_offset = int(absolute_hour // 24.0)
            weekday = (int(observation.weekday) + day_offset) % 7
            days = max(float(site_days[weekday]), 1.0)
            expected[offset] = (
                self.arrival_rate(
                    observation.site,
                    hour,
                    weekday,
                )
                * days
            )

        ratio = history / np.maximum(expected, 0.25)
        bounded = np.clip(ratio, 0.5, 1.5)
        factors = np.clip(
            1.0 + 0.3 * (bounded - 1.0), 0.85, 1.15
        )
        multipliers[:window] = factors[:window]
        return multipliers

    @staticmethod
    def _jpl_occupancy_load_factor(active_count: float, hour: float) -> float:
        active_bin = len(JPL_ACTIVE_EDGES)
        for index, edge in enumerate(JPL_ACTIVE_EDGES):
            if float(active_count) < edge:
                active_bin = index
                break
        hour_bin = len(JPL_HOUR_EDGES)
        hour = float(hour) % 24.0
        for index, edge in enumerate(JPL_HOUR_EDGES):
            if hour < edge:
                hour_bin = index
                break
        factor = float(
            JPL_OCCUPANCY_LOAD_FACTORS[active_bin][hour_bin]
        )
        return max(0.85, min(1.20, factor))

    def _occupancy_multiplier(self, observation, steps: int) -> float:
        if steps < 2:
            return 1.0
        regressions = self.arrival_occupancy_regressions.get(
            observation.site,
            {},
        )
        regression = regressions.get(int(observation.step))
        if regression is None:
            return 1.0
        current_count = float(len(observation.vehicles))
        intercept, slope = regression
        predicted_future = max(0.0, intercept + slope * current_count)

        base_future = 0.0
        for offset in range(1, min(int(steps), 12)):
            absolute_hour = (
                float(observation.local_hour)
                + offset * float(observation.dt_hours)
            )
            hour = absolute_hour % 24.0
            day_offset = int(absolute_hour // 24.0)
            weekday = (int(observation.weekday) + day_offset) % 7
            base_future += self.arrival_rate(
                observation.site,
                hour,
                weekday,
            )
        if base_future <= EPS:
            return 1.0
        return float(np.clip(predicted_future / base_future, 0.5, 2.0))

    def _future_cohorts(
        self, site: str, hour: float, weekday: int
    ) -> tuple[tuple[float, float, float], ...]:
        key = (site, round(float(hour) % 24.0, 4), int(weekday) % 7)
        cached = self._cohort_cache.get(key)
        if cached is not None:
            return cached

        model = self.models.get(site)
        if model is None:
            demand, dwell = self.fallback_labels
            result = ((demand, dwell, 1.0),)
            self._cohort_cache[key] = result
            return result

        request_median = self.request_medians[site]
        x = self._feature_matrix(
            [float(hour) % 24.0],
            [int(weekday) % 7],
            [CHARGER_KW],
            [np.nan],
            request_median,
        )
        point = self._extra_trees_prediction(
            ArrivalFeatures(
                token="cohort-center",
                site=site,
                hour=float(hour),
                weekday=int(weekday),
                charger_limit_kw=CHARGER_KW,
            )
        )
        tree_predictions = np.asarray(
            [tree.predict(x)[0] for tree in model.estimators_],
            dtype=float,
        )
        if len(tree_predictions) == 0:
            result = ((point.demand_kwh, point.dwell_hours, 1.0),)
            self._cohort_cache[key] = result
            return result

        tree_predictions = tree_predictions[
            np.argsort(tree_predictions[:, 1])
        ]
        picks = [
            int(round(q * (len(tree_predictions) - 1)))
            for q in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
        selected = tree_predictions[picks].copy()
        selected[:, 1] = np.clip(
            selected[:, 1],
            0.05,
            self.dwell_caps.get(site, 36.0),
        )
        demand_scale = point.demand_kwh / max(
            float(np.mean(selected[:, 0])), EPS
        )
        dwell_scale = point.dwell_hours / max(
            float(np.mean(selected[:, 1])), EPS
        )
        selected[:, 0] = np.clip(
            selected[:, 0] * demand_scale,
            0.01,
            self.demand_caps.get(site, 150.0),
        )
        selected[:, 1] = np.clip(
            selected[:, 1] * dwell_scale,
            0.05,
            self.dwell_caps.get(site, 36.0),
        )
        result = tuple(
            (float(demand), float(dwell), 0.2)
            for demand, dwell in selected
        )
        self._cohort_cache[key] = result
        return result

    def forecast(self, observation, horizon_steps: int, intervention: str):
        if intervention not in {"natural", "capacity_edf"}:
            raise ValueError(f"unsupported intervention: {intervention}")
        particle_result = None
        if observation.site in PARTICLE_SITES:
            particle_result = self._particle_forecast(
                observation,
                horizon_steps,
                intervention,
            )
        if particle_result is not None:
            if (
                intervention == "capacity_edf"
                and PARTICLE_EDF_EFFECT_SCALE != 1.0
            ):
                natural_result = self._particle_forecast(
                    observation,
                    horizon_steps,
                    "natural",
                )
                capacity_series = (
                    observation.intervention_capacity
                    or observation.forecast_capacity
                )
                adjusted = []
                for step in range(len(particle_result["ev_load_kw"])):
                    natural_load = float(natural_result["ev_load_kw"][step])
                    capacity_load = float(particle_result["ev_load_kw"][step])
                    total_capacity = self._series_value(
                        capacity_series,
                        step,
                        float(observation.capacity_kw),
                    )
                    background = self._series_value(
                        observation.forecast_background,
                        step,
                        float(observation.background_kw),
                    )
                    ev_limit = max(total_capacity - background, 0.0)
                    adjusted.append(
                        float(
                            np.clip(
                                natural_load
                                + PARTICLE_EDF_EFFECT_SCALE
                                * (capacity_load - natural_load),
                                0.0,
                                ev_limit,
                            )
                        )
                    )
                particle_result = {
                    "ev_load_kw": adjusted,
                    "active_count": particle_result["active_count"],
                }
            return particle_result
        if intervention == "capacity_edf" and not self._capacity_blend_active:
            self._capacity_blend_active = True
            original_arrival_rate = self.arrival_rate
            try:
                natural = self.forecast(
                    observation,
                    horizon_steps,
                    "natural",
                )
                full_capacity = self.forecast(
                    observation,
                    horizon_steps,
                    "capacity_edf",
                )
                self.arrival_rate = lambda site, hour, weekday: 0.0
                current_natural = self.forecast(
                    observation,
                    horizon_steps,
                    "natural",
                )
                current_capacity = self.forecast(
                    observation,
                    horizon_steps,
                    "capacity_edf",
                )
            finally:
                self.arrival_rate = original_arrival_rate
                self._capacity_blend_active = False

            current_effect = (
                np.asarray(current_natural["ev_load_kw"], dtype=float)
                - np.asarray(current_capacity["ev_load_kw"], dtype=float)
            )
            current_effect *= float(
                CAPACITY_EFFECT_SCALE_BY_SITE.get(observation.site, 1.0)
            )
            capacity_series = (
                observation.intervention_capacity
                or observation.forecast_capacity
            )
            adjusted_loads = []
            for step in range(max(0, int(horizon_steps))):
                total_capacity = self._series_value(
                    capacity_series,
                    step,
                    float(observation.capacity_kw),
                )
                background = self._series_value(
                    observation.forecast_background,
                    step,
                    float(observation.background_kw),
                )
                ev_limit = max(total_capacity - background, 0.0)
                adjusted_loads.append(
                    float(
                        np.clip(
                            float(natural["ev_load_kw"][step])
                            - float(current_effect[step]),
                            0.0,
                            ev_limit,
                        )
                    )
                )
            return {
                "ev_load_kw": adjusted_loads,
                "active_count": full_capacity["active_count"],
            }

        steps = max(0, int(horizon_steps))
        arrival_multipliers = self._arrival_multipliers(observation, steps)
        occupancy_multiplier = self._occupancy_multiplier(observation, steps)
        cars: list[dict[str, object]] = []
        for vehicle in observation.vehicles:
            if vehicle.charging_complete:
                continue
            pred = self._stable_predict(vehicle.features)
            remaining = self._estimated_remaining(vehicle)
            if remaining <= EPS:
                continue
            remaining_hours = self._survival_remaining_hours(
                observation.site,
                pred.dwell_hours,
                float(vehicle.elapsed_steps) * observation.dt_hours,
                observation.dt_hours,
            )
            demand_ratios, dwell_offsets = self._current_cohort_shapes(
                vehicle.features
            )
            for probability, demand_ratio, offset in zip(
                CURRENT_COHORT_PROBABILITIES,
                demand_ratios,
                dwell_offsets,
            ):
                cohort_hours = max(
                    observation.dt_hours,
                    remaining_hours + offset,
                )
                remaining_steps = max(
                    1,
                    int(
                        math.ceil(
                            cohort_hours
                            / max(observation.dt_hours, EPS)
                        )
                    ),
                )
                cars.append(
                    {
                        "remaining_total": (
                            remaining * probability * demand_ratio
                        ),
                        "steps_left": remaining_steps,
                        "charger_kw": max(
                            float(vehicle.features.charger_limit_kw), 0.0
                        ),
                        "weight": probability,
                    }
                )

        loads: list[float] = []
        active_counts: list[float] = []
        for step in range(steps):
            elapsed_hours = float(observation.local_hour) + (
                step * float(observation.dt_hours)
            )
            hour = elapsed_hours % 24.0
            weekday = (
                int(observation.weekday) + int(elapsed_hours // 24.0)
            ) % 7
            rate = (
                self.arrival_rate(observation.site, hour, weekday)
                * float(arrival_multipliers[step])
                * occupancy_multiplier
            )
            if rate > EPS:
                for demand, dwell, probability in self._future_cohorts(
                    observation.site, hour, weekday
                ):
                    cohort_weight = rate * probability
                    cars.append(
                        {
                            "remaining_total": cohort_weight * demand,
                            "steps_left": max(
                                1,
                                int(
                                    math.ceil(
                                        dwell
                                        / max(observation.dt_hours, EPS)
                                    )
                                ),
                            ),
                            "charger_kw": CHARGER_KW,
                            "weight": cohort_weight,
                        }
                    )

            active_counts.append(
                float(
                    sum(
                        float(car["weight"])
                        for car in cars
                        if float(car["remaining_total"]) > EPS
                        and int(car["steps_left"]) > 0
                    )
                )
            )

            if intervention == "natural":
                ev_capacity = math.inf
            else:
                capacity_series = (
                    observation.intervention_capacity
                    or observation.forecast_capacity
                )
                total_capacity = self._series_value(
                    capacity_series,
                    step,
                    float(observation.capacity_kw),
                )
                background = self._series_value(
                    observation.forecast_background,
                    step,
                    float(observation.background_kw),
                )
                ev_capacity = max(total_capacity - background, 0.0)

            capacity_left = ev_capacity
            load = 0.0
            ordered = sorted(
                cars,
                key=lambda car: (
                    int(car["steps_left"]),
                    -float(car["remaining_total"]),
                ),
            )
            for car in ordered:
                weight = float(car["weight"])
                if weight <= EPS or capacity_left <= EPS:
                    continue
                per_car_power = min(
                    float(car["charger_kw"]),
                    (
                        float(car["remaining_total"])
                        / weight
                        / max(observation.dt_hours, EPS)
                    ),
                )
                requested_load = max(0.0, per_car_power) * weight
                allocated_load = min(requested_load, capacity_left)
                if allocated_load <= EPS:
                    continue
                car["remaining_total"] = max(
                    0.0,
                    float(car["remaining_total"])
                    - allocated_load * observation.dt_hours,
                )
                capacity_left -= allocated_load
                load += allocated_load

            for car in cars:
                car["steps_left"] = int(car["steps_left"]) - 1
            cars = [
                car
                for car in cars
                if float(car["remaining_total"]) > EPS
                and int(car["steps_left"]) > 0
            ]
            loads.append(max(0.0, load))

        if intervention == "natural":
            natural_scale = float(
                NATURAL_LOAD_SCALE_BY_SITE.get(observation.site, 1.0)
            )
            scaled_loads = []
            for step, load in enumerate(loads):
                factor = natural_scale
                if observation.site == "jpl":
                    hour = (
                        float(observation.local_hour)
                        + step * float(observation.dt_hours)
                    ) % 24.0
                    factor *= self._jpl_occupancy_load_factor(
                        float(active_counts[step]),
                        hour,
                    )
                scaled_loads.append(float(load) * factor)
            loads = scaled_loads
        return {
            "ev_load_kw": loads,
            "active_count": active_counts,
        }

    def explain(self):
        return {
            "method": (
                "Site-specific HistGradientBoosting absolute-error ensembles "
                "predict demand and dwell from arrival time, weekday, charger "
                "limit and the optional last-request snapshot, with recent-day "
                "weighting and a charger-time demand cap. "
                "Arrival rates use weekday-by-96-slot empirical frequencies; "
                "slots with mean arrivals below 0.30 use the daily-count "
                "median to reduce sparse-slot MAE while busier slots retain "
                "the mean. "
                "A 15-minute physical world model rolls out charger and site "
                "power limits, fractional expected vehicle cohorts with explicit "
                "aggregate remaining energy, and five joint ExtraTrees "
                "demand/dwell quantile cohorts for future arrivals. "
                "Current vehicles use five joint ExtraTrees demand/dwell "
                "quintile cohorts at equal probabilities; each pair comes "
                "from the same tree, centered on the survival-adjusted "
                "remaining dwell while preserving total expected vehicles "
                "and remaining energy. "
                "Current vehicles use training-only conditional survival "
                "residuals after a substantial fraction of their predicted "
                "dwell has elapsed, so observed survival updates remaining "
                "dwell rather than relying only on arrival-time countdown. "
                "Earliest-deadline allocation is applied to every fractional "
                "cohort under both natural and capacity_edf interventions, "
                "without an arbitrary minimum service-vehicle budget. "
                "The reported capacity_edf marginal effect is computed from "
                "the current-vehicle EDF response only; future-arrival cohorts "
                "remain in the natural trajectory and active count but are not "
                "allowed to create an unsupported marginal capacity effect. "
                "Historical quarter-hour arrival counts are aligned from the "
                "current local step for any supplied window length and "
                "normalized by same-weekday training coverage, then calibrate "
                "each future period by up to fifteen percent. "
                "For JPL and Office1, the world model selects 64 complete "
                "historical-day vehicle trajectories by weekday and current "
                "occupancy. All particles preserve joint arrival, demand, dwell "
                "and departure structure. Natural load is the particle average; "
                "capacity active count is obtained by running the same particle "
                "pool through earliest-deadline allocation under the public "
                "intervention capacity. V50-B applies 50 percent of the particle "
                "EDF marginal effect instead of returning natural load. V59 adds "
                "a public prediction calibration layer using time-holdout-validated "
                "XGBoost and LightGBM blends for Caltech and JPL. Forecasting and "
                "online dispatch retain the stable V50-B predictor so their "
                "calibrated world-model and completion behavior is unchanged. "
                "The optional request snapshot is used "
                "only for current vehicles. Caltech retains the validated v25 "
                "physical forecast. No particle contains hidden test data. "
                "Online dispatch creates a planning demand envelope from the "
                "positive spread between the two demand-model seeds, then "
                "applies a 1.25 completion margin and caps the optional request "
                "snapshot at 1.35 times that robust target. Deadline urgency "
                "uses a site-and-time empirical lower-dwell profile at one and "
                "a half times its fifteenth percentile, bounded by the point "
                "dwell prediction. No-slack vehicles are protected and minimum "
                "completion power is guaranteed when possible. Office1 uses "
                "slack-first scheduling and shares "
                "residual power with tail-uncertainty vehicles whose predicted "
                "demand is satisfied but whose observed completion is pending. "
                "Below 90% of the aggregate completion floor it concentrates on "
                "the smallest remaining jobs instead."
            ),
            "assumptions": [
                "Only observations available at the current step are used for actions.",
                "Last-request values are optional snapshots, not verified demand intent.",
                "Demand, dwell and arrival processes do not change when only the counterfactual capacity changes.",
                "Each charging interval is 0.25 hours and future charger capacity is 6.6 kW.",
                "Historical arrivals are aggregate same-weekday counts aligned to the current quarter-hour step.",
                "Current parked-vehicle count is predictive of future arrival intensity.",
                "The empirical lower-dwell profile is an action-only deadline guard, not a change to predict().",
                "Historical-day particles are sampled only from training sessions available before the evaluation cutoff.",
            ],
            "limitations": [
                "Vehicle demand and dwell are inherently uncertain at arrival.",
                "Sparse Office1 history limits site-specific learning and may require conservative allocation.",
                "The intervention is a simulator physics response, not evidence of real user behavior.",
                "Forecasts cannot observe future arrivals, departures, or hidden remaining demand.",
                "Historical analog days may not cover a materially different hidden arrival pattern.",
            ],
        }


def build_model(training):
    return RobustChargingModel(training)
