import csv
import os
import time
from typing import Dict, List, Optional

import numpy as np

from utils.config import Config, load_config
from utils.rng import RNG
from sensors.temp_sensor import TempSensor
from sensors.filters import hold_last, MovingAverageFilter
from controllers.onoff import OnOffThermostat
from controllers.predictive_onoff import PredictiveOnOff
from simulations.room_model import step_room
from simulations.environment import Environment
from plotting.plots import plot_timeseries, plot_error, plot_duty, plot_predictive, plot_heater

def run_scenario(scenario_path: str):
    scenario = load_config(scenario_path)
    log, use_predictive = _simulate_scenario(scenario, scenario.sim.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.splitext(os.path.basename(scenario_path))[0]
    log_dir = os.path.join("outputs", "logs")
    fig_dir = os.path.join("outputs", "figures")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{base}-{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(log.keys())
        writer.writerows(zip(*(log[key] for key in log)))

    plot_timeseries(log, os.path.join(fig_dir, f"{base}-temps-{timestamp}.png"))
    plot_heater(log, os.path.join(fig_dir, f"{base}-heater-{timestamp}.png"))
    plot_error(log, os.path.join(fig_dir, f"{base}-error-{timestamp}.png"))
    plot_duty(log, os.path.join(fig_dir, f"{base}-duty-{timestamp}.png"))
    if use_predictive:
        plot_predictive(log, os.path.join(fig_dir, f"{base}-predictive-{timestamp}.png"))

    print(f"Wrote log to {csv_path}")
    print(f"Figures saved to {fig_dir}")
    
def _simulate_scenario(scenario: Config, seed: int) -> tuple[Dict[str, List[float]], bool]:
    rng = RNG(seed)

    env = Environment(
        base=scenario.env.base,
        amplitude=scenario.env.amplitude,
        period_s=scenario.env.period_s,
        door_drop_C=scenario.env.door_drop_C,
        door_start_s=scenario.env.door_start_s,
        door_duration_s=scenario.env.door_duration_s,
    )

    sensor = TempSensor(
        sigma=scenario.sensor.sigma,
        bias=scenario.sensor.bias,
        dropout_prob=scenario.sensor.dropout_prob,
        rng=rng,
    )

    if getattr(scenario.controller, "type", "predictive_onoff") == "onoff":
        ctrl = OnOffThermostat(
            setpoint=scenario.controller.setpoint,
            deadband=scenario.controller.deadband,
            safety_high=scenario.controller.safety_high,
            state=0,
        )
        use_predictive = False
    else:
        ctrl = PredictiveOnOff(
            setpoint=scenario.controller.setpoint,
            deadband=scenario.controller.deadband,
            tau=scenario.controller.tau,
            safety_high=scenario.controller.safety_high,
            state=0,
        )
        use_predictive = True

    dt = scenario.sim.dt
    steps = int(scenario.sim.duration_s / dt)
    temperature = scenario.sim.init_T
    last_valid: Optional[float] = temperature

    keys = ["t", "T_true", "T_meas", "T_out", "setpoint", "heater", "error", "T_pred", "lower", "upper"]
    log: Dict[str, List[float]] = {key: [] for key in keys}
    moving_average = MovingAverageFilter(window=5)

    for step in range(steps):
        current_time = step * dt
        outside_temperature = env.T_out(current_time)
        measured_temperature = sensor.read(temperature)
        measured_temperature = hold_last(measured_temperature, last_valid)
        if measured_temperature is None:
            measured_temperature = temperature
        last_valid = measured_temperature
        filtered_temperature = moving_average.update(measured_temperature)

        if use_predictive:
            heater = ctrl.update(filtered_temperature, dt)
            predicted_temperature = ctrl.last_pred if ctrl.last_pred is not None else filtered_temperature
            lower = ctrl.lower_threshold if ctrl.lower_threshold is not None else (
                ctrl.setpoint - ctrl.deadband / 2
            )
            upper = ctrl.upper_threshold if ctrl.upper_threshold is not None else (
                ctrl.setpoint + ctrl.deadband / 2
            )
        else:
            heater = ctrl.update(filtered_temperature)
            predicted_temperature = filtered_temperature
            lower = ctrl.setpoint - ctrl.deadband / 2
            upper = ctrl.setpoint + ctrl.deadband / 2

        log["t"].append(current_time)
        log["T_true"].append(temperature)
        log["T_meas"].append(filtered_temperature)
        log["T_out"].append(outside_temperature)
        log["setpoint"].append(ctrl.setpoint)
        log["heater"].append(heater)
        log["error"].append(ctrl.setpoint - filtered_temperature)
        log["T_pred"].append(predicted_temperature)
        log["lower"].append(lower)
        log["upper"].append(upper)

        temperature = step_room(
            temperature,
            heater,
            outside_temperature,
            scenario.model.R,
            scenario.model.C,
            scenario.model.P,
            dt,
            scenario.model.process_sigma,
            rng,
        )

    return log, use_predictive


def run_monte_carlo(scenario_path: str, runs: int = 100) -> Dict[str, Dict[str, float]]:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    scenario = load_config(scenario_path)
    metric_names = [
        "min_temp_C",
        "below_setpoint_fraction",
        "heater_duty_fraction",
        "max_overshoot_C",
    ]
    trial_rows = []

    for trial_index in range(runs):
        seed = scenario.sim.seed + trial_index
        log, _ = _simulate_scenario(scenario, seed)
        temperatures = log["T_true"]
        heater_states = log["heater"]
        trial_rows.append({
            "trial": trial_index + 1,
            "seed": seed,
            "min_temp_C": min(temperatures),
            "below_setpoint_fraction": sum(
                temperature < scenario.controller.setpoint for temperature in temperatures
            ) / len(temperatures),
            "heater_duty_fraction": sum(heater_states) / len(heater_states),
            "max_overshoot_C": max(0.0, max(temperatures) - scenario.controller.setpoint),
        })

    summary: Dict[str, Dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.asarray([row[metric_name] for row in trial_rows], dtype=float)
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.splitext(os.path.basename(scenario_path))[0]
    log_dir = os.path.join("outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{base}-monte-carlo-{timestamp}.csv")
    fieldnames = ["trial", "seed", *metric_names]
    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)

    print(f"Monte Carlo results for {runs} runs written to {csv_path}")
    for metric_name, statistics in summary.items():
        print(
            f"{metric_name}: mean={statistics['mean']:.3f}, "
            f"p05={statistics['p05']:.3f}, p95={statistics['p95']:.3f}"
        )
    return summary
