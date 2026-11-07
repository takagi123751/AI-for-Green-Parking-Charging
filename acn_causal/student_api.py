"""Public AI for Green Track A data structures.

This compatibility copy lets the submitted model be imported locally. The
official evaluation environment provides the same public interface.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ArrivalFeatures:
    token: str
    site: str
    hour: float
    weekday: int
    charger_limit_kw: float
    last_requested_energy_kwh: Optional[float] = None


@dataclass(frozen=True)
class VehicleObservation:
    features: ArrivalFeatures
    connected_step: int
    elapsed_steps: int
    delivered_kwh: float
    charging_complete: bool


@dataclass(frozen=True)
class Observation:
    step: int
    dt_hours: float
    horizon_steps: int
    site: str
    local_hour: float
    weekday: int
    vehicles: Tuple[VehicleObservation, ...]
    capacity_kw: float
    background_kw: float
    prices: Tuple[float, ...]
    forecast_capacity: Tuple[float, ...]
    forecast_background: Tuple[float, ...]
    historical_arrivals: Tuple[int, ...]
    intervention_capacity: Tuple[float, ...] = ()


@dataclass(frozen=True)
class Prediction:
    demand_kwh: float
    dwell_hours: float
