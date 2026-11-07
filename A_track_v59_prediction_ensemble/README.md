# AI for Green Track A Submission

This directory is a runnable submission source for the official Track A
`停车场充电世界模型` task.

## Model

- `submission.py` exports `build_model(training)`.
- Vehicle demand and dwell are predicted by site-specific
  HistGradientBoosting absolute-error ensembles. Demand and dwell use separate
  targets, two seeds are averaged, and recent sessions receive more weight.
- V59 adds a public-prediction calibration layer. XGBoost blends improve the
  time-ordered holdout MAE for Caltech demand/dwell and JPL demand/dwell;
  LightGBM is additionally blended for Caltech demand. Office1 keeps the stable
  predictor. Forecasting and online dispatch call `_stable_predict`, so their
  V50-B world-model and completion behavior is unchanged.
- On the official public 80:20 time holdout (8,063 validation records), V59
  reduced demand MAE from `4.41349` to `4.39924` kWh and dwell MAE from
  `1.68334` to `1.67520` hours. A full build plus public evaluation completed
  locally in about 77 seconds, inside the 120-second development CPU limit.
- The existing ExtraTrees ensemble is retained only for future-arrival
  demand/dwell cohort modeling, so the world-model distribution is not replaced
  by a single gradient-boosted point forecast.
- The optional `last_requested_energy_kwh` snapshot is used with an explicit
  missing-value indicator and transformed request features, and is not treated
  as ground truth.
- Predicted demand is capped by charger power multiplied by predicted dwell.
- Arrival rates use site-by-weekday-by-96-slot empirical frequencies, matching
  the official 15-minute interface instead of discarding within-hour timing.
- Sparse slots whose historical mean arrival rate is below `0.30` use the
  daily-count median instead of the mean. Busier slots retain the mean. This
  improves the MAE-oriented arrival prediction without degrading downstream
  world-model counts.
- The world model advances in 15-minute physical steps.
- Expected future arrivals keep the same fractional semantics as V9. V10 stores
  each cohort's total remaining energy explicitly, which is behaviorally
  equivalent when no historical calibration factor is applied.
- Future arrivals now use five joint demand/dwell quantile cohorts sampled from
  the ExtraTrees ensemble instead of assigning every arriving vehicle the same
  point dwell time. The cohort probabilities preserve the expected vehicle
  count.
- Current vehicles use five joint demand/dwell cohorts at the 10th, 30th,
  50th, 70th and 90th ExtraTrees quantiles with equal probabilities. Each pair
  comes from the same tree, is centered on the survival-adjusted remaining
  dwell, and preserves expected vehicle count and remaining energy.
- Historical quarter-hour arrival counts are aligned from the current local
  step for any truncated window length, normalized by same-weekday training
  coverage, and calibrate each future period by a bounded factor. An all-zero
  valid history is treated as genuine low demand.
- Future-arrival intensity is conditioned on the current number of parked
  vehicles using site-and-origin regressions fitted only from complete
  historical training days. This captures high-occupancy burst conditions and
  low-occupancy days without reading future scenario vehicles.
- Current vehicles whose predicted dwell has already expired use a
  training-only conditional survival residual instead of being collapsed to a
  one-step tail.
- Caltech `natural` load receives a `0.80` factor selected from time-ordered
  folds inside the public training data. JPL applies a `5 × 5` occupancy-by-time
  correction table fitted on an inner training fold and shrunk 50% toward no
  correction. It uses only the predicted active-vehicle count and local hour,
  caps factors to `0.85–1.20`, and does not change demand prediction, arrival
  rate, active count or online actions. Office1 retains a factor of `1.0`.
- `natural` and `capacity_edf` use the same initial state and arrival assumptions;
  only capacity and allocation priority change.
- `capacity_edf` rolls out every fractional current and future cohort in
  earliest-predicted-departure order under the intervention capacity, without
  an arbitrary minimum service-vehicle budget.
- `natural` remains the full current-plus-future trajectory and active count.
  The reported `capacity_edf` load is the calibrated natural load clipped by
  intervention headroom. The unstable current-only marginal effect is disabled:
  cross-fold checks showed that adding it increased effect error and created
  false reductions when capacity did not bind.
- Online remaining demand caps the optional last-request snapshot at
  `1.35 × predicted demand`; the unverified request is never an unbounded hard
  target.
- Office1 uses a sparse-site slack-first scheduler. Vehicles whose predicted
  demand is satisfied but which are not yet observed complete remain in a
  shared tail bucket, so residual power is not discarded while completion
  uncertainty remains.
- Online actions use current capacity and forecast capacity, never
  `intervention_capacity`.
- Completion-aware dispatch calculates each vehicle's latest feasible start
  and minimum required power. No-slack vehicles are protected first, remaining
  vehicles receive a minimum completion allocation, and spare power then
  follows smallest-remaining-job order to maximize completed vehicle count.
- Action planning uses the positive disagreement between the two demand-model
  seeds as an uncertainty envelope. This does not change `predict()` output.
- Deadline urgency uses a site-and-time empirical lower-dwell profile equal to
  `1.5 × q15`, bounded by the point dwell prediction. The lower profile is
  fitted only from training sessions and is used only inside `act()`.
- If current capacity is below 90% of the aggregate completion floor, the
  scheduler stops spreading power thinly and concentrates on the smallest
  feasible jobs.
- Online dispatch plans with a 1.25 demand margin to retain completion safety
  in the hidden 1.35 demand-shift condition. This margin affects action choice
  only; prediction and world-model outputs are unchanged.
- For JPL and Office1, future arrivals are represented by 64 complete
  historical-day trajectories selected by weekday and current parked-vehicle
  count. Each particle preserves the joint correlation between arrival time,
  demand, dwell and departure, and occupancy scaling is applied to the future
  session pool. Natural load is the particle average. Capacity active count is
  produced by running the same particle pool through earliest-deadline
  allocation under the public intervention capacity. V50-B reports 50 percent
  of the particle EDF marginal effect instead of returning natural load.
  Caltech retains the validated v25 physical forecast.

## Local check

Use the official fixed environment. From the project root:

```powershell
$env:PYTHONPATH = (Resolve-Path "AIforGreen-参赛包").Path
python -m unittest discover -s tests -v
```

The official packaging command can be run from this directory:

```powershell
python ..\AIforGreen-参赛包\make_submission.py `
  --source . `
  --route ml `
  --environment-id cpu-ml-methods-v3 `
  --out ..\submissions\AIforGreen_track_a_v59_prediction_ensemble.zip
```

## Limitations

The model does not read hidden labels, future arrivals, future departures, or
future observations. The capacity intervention is a simulator physics
comparison and is not evidence of real driver behavior. Office1 has the least
training history and therefore the greatest sampling uncertainty. The Caltech
and JPL corrections are calibrated on public training folds; a different hidden
station distribution could make the factors too aggressive. Some sparse JPL
occupancy/time cells have limited samples, so all factors are heavily shrunk.

## AI use disclosure

OpenAI Codex was used to inspect the official interface, implement the model,
write tests, and prepare documentation. The submitted model runs offline and
does not call any online model API.
