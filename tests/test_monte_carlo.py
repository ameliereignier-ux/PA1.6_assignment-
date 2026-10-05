import math
import csv

from simulations.environment import Environment
from simulations.room_model import step_room
from utils.rng import RNG


def test_rng_is_seeded_and_samples_distribution_methods():
    first = RNG(seed=12)
    second = RNG(seed=12)

    first_values = [first.gauss(0.0, 1.0), first.uniform(-1.0, 1.0), first.bernoulli(0.5)]
    second_values = [second.gauss(0.0, 1.0), second.uniform(-1.0, 1.0), second.bernoulli(0.5)]

    assert first_values == second_values
    assert RNG(seed=1).bernoulli(0.0) is False
    assert RNG(seed=1).bernoulli(1.0) is True


def test_environment_applies_door_drop_only_during_event():
    environment = Environment(
        base=5.0,
        amplitude=2.0,
        period_s=3600.0,
        door_drop_C=7.0,
        door_start_s=10.0,
        door_duration_s=5.0,
    )
    baseline_at_start = 5.0 + 2.0 * math.sin(2.0 * math.pi * 10.0 / 3600.0)

    assert math.isclose(environment.T_out(10.0), baseline_at_start - 7.0)
    assert math.isclose(environment.T_out(15.0), 5.0 + 2.0 * math.sin(2.0 * math.pi * 15.0 / 3600.0))


def test_room_model_uses_euler_step():
    updated = step_room(T=20.0, heater_on=1, T_out=10.0, R=2.0, C=100.0, P=50.0, dt=10.0)

    assert math.isclose(updated, 24.5)


def test_room_process_noise_is_seeded():
    first = step_room(20.0, 0, 10.0, 2.0, 100.0, 50.0, 10.0, 0.2, RNG(9))
    second = step_room(20.0, 0, 10.0, 2.0, 100.0, 50.0, 10.0, 0.2, RNG(9))

    assert first == second


def test_monte_carlo_writes_reproducible_trial_seeds_and_summary(tmp_path, monkeypatch):
        scenario_path = tmp_path / "short_scenario.yaml"
        scenario_path.write_text(
                """env:
    base: 5.0
    amplitude: 2.0
    period_s: 100.0
    door_drop_C: 8.0
    door_start_s: 5.0
    door_duration_s: 10.0
sensor:
    sigma: 0.2
    bias: 0.0
    dropout_prob: 0.05
controller:
    type: onoff
    setpoint: 21.0
    deadband: 1.0
    safety_high: 26.0
model:
    R: 0.5
    C: 10000.0
    P: 200.0
    process_sigma: 0.1
sim:
    dt: 1.0
    duration_s: 30.0
    seed: 17
    init_T: 18.0
""",
                encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        summary = run_monte_carlo(str(scenario_path), runs=3)

        assert set(summary) == {
                "min_temp_C",
                "below_setpoint_fraction",
                "heater_duty_fraction",
                "max_overshoot_C",
        }
        assert all(set(values) == {"mean", "p05", "p95"} for values in summary.values())
        result_files = list((tmp_path / "outputs" / "logs").glob("*-monte-carlo-*.csv"))
        assert len(result_files) == 1
        with result_files[0].open(newline="", encoding="utf-8") as result_file:
                rows = list(csv.DictReader(result_file))
        assert [row["seed"] for row in rows] == ["17", "18", "19"]