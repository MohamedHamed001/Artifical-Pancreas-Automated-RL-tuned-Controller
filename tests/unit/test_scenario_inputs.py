from dataclasses import replace

import numpy as np
import pytest

from ap_rl.controllers.pid_controller import PIDController
from ap_rl.core.types import PatientConfig, Scenario
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
from ap_rl.envs.diabetes_pid_env import DiabetesPIDEnv
from ap_rl.simulation.simulator import SimulationConfig, SimulationRunner
from ap_rl.utils.insulin_calculator import InsulinCalculator
from ap_rl.utils.scenarios import ScenarioLoader


def _make_runner(duration_min: int = 1, dt_min: int = 1) -> SimulationRunner:
    return SimulationRunner(
        SimulationConfig(
            patient_config=PatientConfig(
                name="scenario-input-test",
                params=DEFAULT_PATIENT_PARAMS,
                body_weight_kg=75.0,
            ),
            controller=PIDController(Kp=0.1, Ki=0.01, Kd=0.001),
            duration_min=duration_min,
            dt_min=dt_min,
        )
    )


def _record_runner_activity(runner):
    safety_calls = []
    patient_inputs = []
    evaluate = runner.safety.evaluate
    patient_step = runner.patient.step

    def record_safety(command, state, basal_rate_uh):
        decision = evaluate(command, state, basal_rate_uh)
        safety_calls.append((command, decision))
        return decision

    def record_patient_input(t, insulin_rate_u_h, meal_carbs_g, exercise_active):
        patient_inputs.append((t, insulin_rate_u_h, meal_carbs_g))
        return patient_step(t, insulin_rate_u_h, meal_carbs_g, exercise_active)

    runner.safety.evaluate = record_safety
    runner.patient.step = record_patient_input
    return safety_calls, patient_inputs


def _run_known_bolus(dt_min):
    runner = _make_runner(duration_min=5, dt_min=dt_min)
    runner.insulin_calc = InsulinCalculator(
        patient_weight_kg=75.0,
        carb_ratio=10.0,
        split_ratio=1.0,
        correction_dead_band=1000.0,
    )
    safety_calls, patient_inputs = _record_runner_activity(runner)
    controller_calls = []
    get_action = runner.controller.get_action

    def record_controller_call(*args, **kwargs):
        controller_calls.append(None)
        return get_action(*args, **kwargs)

    runner.controller.get_action = record_controller_call
    episode = runner.run(meal_data=[{"time": 2, "carbs": 20.0}])
    return episode, safety_calls, patient_inputs, controller_calls


def test_runner_applies_case10_structured_meal_grams_once():
    runner = _make_runner(duration_min=175, dt_min=5)
    scenario = ScenarioLoader.load_case(10)
    safety_calls, patient_inputs = _record_runner_activity(runner)
    bolus_calls = []
    deliver_bolus = runner.insulin_calc.deliver_bolus

    def record_bolus(carbs_grams, current_glucose_mgdl, target_glucose_mgdl):
        bolus_calls.append((runner.insulin_calc.current_time, carbs_grams))
        return deliver_bolus(
            carbs_grams,
            current_glucose_mgdl,
            target_glucose_mgdl,
        )

    runner.insulin_calc.deliver_bolus = record_bolus
    runner.run(scenario=scenario)

    applied_meals = [(t, meal) for t, _, meal in patient_inputs if meal]
    event_commands = [
        command for command, _ in safety_calls if command.time_min == 172
    ]
    assert applied_meals == [(172, 78.06)]
    assert bolus_calls == [(172, 78.06)]
    assert len(event_commands) == 1
    assert event_commands[0].total_u_h > event_commands[0].basal_u_h


def test_runner_delivers_immediate_dose_only_in_event_minute():
    episode, safety_calls, patient_inputs, controller_calls = _run_known_bolus(5)
    requested_non_basal = [
        command.total_u_h - command.basal_u_h for command, _ in safety_calls
    ]

    assert requested_non_basal == pytest.approx([0.0, 0.0, 120.0, 0.0, 0.0])
    assert [t for t, _, _ in patient_inputs] == [0, 1, 2, 3, 4]
    assert [insulin for _, insulin, _ in patient_inputs] == pytest.approx(
        [decision.delivered.total_u_h for _, decision in safety_calls]
    )
    assert patient_inputs[2][1] == pytest.approx(
        safety_calls[2][1].delivered.total_u_h
    )
    assert safety_calls[2][1].delivered.total_u_h < safety_calls[2][0].total_u_h
    assert len(controller_calls) == 1
    assert episode.steps[0].bolus == pytest.approx(24.0)
    assert episode.steps[0].requested_insulin == pytest.approx(
        np.mean([command.total_u_h for command, _ in safety_calls])
    )
    assert episode.steps[0].delivered_insulin == pytest.approx(
        np.mean([decision.delivered.total_u_h for _, decision in safety_calls])
    )


def test_requested_bolus_units_are_invariant_between_step_sizes():
    integrated_units = []
    for dt_min in (1, 5):
        _, safety_calls, _, _ = _run_known_bolus(dt_min)
        integrated_units.append(
            sum(
                (command.total_u_h - command.basal_u_h) / 60.0
                for command, _ in safety_calls
            )
        )

    assert integrated_units == pytest.approx([2.0, 2.0])


def test_runner_reset_clears_queued_tail():
    runner = _make_runner()
    runner.insulin_calc = InsulinCalculator(
        patient_weight_kg=75.0,
        carb_ratio=10.0,
        split_ratio=0.5,
        tail_duration_min=10.0,
        correction_dead_band=1000.0,
    )
    runner.step(meal_data=[{"time": 0, "carbs": 20.0}])
    assert runner.insulin_calc.pending_tail_dose > 0.0

    runner.reset()
    safety_calls, _ = _record_runner_activity(runner)
    runner.step()

    command = safety_calls[0][0]
    assert runner.insulin_calc.pending_tail_dose == 0.0
    assert command.total_u_h - command.basal_u_h == pytest.approx(0.0)


def test_safety_blocked_tail_is_consumed_without_later_retry():
    runner = _make_runner()
    runner.safety.min_glucose_mgdl = 1000.0
    runner.insulin_calc = InsulinCalculator(
        patient_weight_kg=75.0,
        carb_ratio=10.0,
        split_ratio=0.0,
        tail_duration_min=2.0,
        correction_dead_band=1000.0,
    )
    _, patient_inputs = _record_runner_activity(runner)

    records = [runner.step(meal_data=[{"time": 0, "carbs": 20.0}])]
    records.extend(runner.step() for _ in range(2))

    assert [record.bolus for record in records] == pytest.approx([60.0, 60.0, 0.0])
    assert [insulin for _, insulin, _ in patient_inputs] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert runner.insulin_calc.pending_tail_dose == 0.0


def test_runner_deduplicates_minute_safety_events_in_first_seen_order():
    runner = _make_runner(duration_min=3, dt_min=3)
    evaluate = runner.safety.evaluate

    def repeat_safety_events(command, state, basal_rate_uh):
        decision = evaluate(command, state, basal_rate_uh)
        events = ["REPEATED"]
        if command.time_min == 2:
            events.insert(0, "SECOND")
        return replace(decision, active_constraints=events)

    runner.safety.evaluate = repeat_safety_events

    record = runner.step()

    assert record.safety_events == ["REPEATED", "SECOND"]


def test_runner_retains_generic_raw_scenario_contract():
    scenario = Scenario(id="raw", meal_data=np.array([[0.0, 12.0]]))

    episode = _make_runner().run(scenario=scenario)

    assert episode.steps[0].cob > 0


def test_environment_loads_structured_case_inputs_from_configured_root(tmp_path):
    (tmp_path / "TestCases.txt").write_text(
        "Test Case [99]\n"
        "Body Weight: 75 kg\n"
        "Meal 1 Time: 12 minutes, Carb Amount: 34.5 grams\n"
        "Exercise 1: Start = 20 minutes, Duration = 4 minutes\n"
    )
    env = DiabetesPIDEnv(
        patient_params=DEFAULT_PATIENT_PARAMS,
        data_root=tmp_path,
        test_case_id=99,
    )

    assert env.meal_data == [{"time": 12, "carbs": 34.5}]
    assert env.exercise_data == [{"start": 20, "duration": 4}]


def test_loader_represents_declared_empty_case_without_raw_fallback():
    scenario = ScenarioLoader.load_case(1)

    assert scenario.meals == []
    assert scenario.meal_data is None


@pytest.mark.parametrize(
    "exercise_data",
    [
        [{"start": 10, "duration": 3}],
        [{"time": 10, "active": 1}, {"time": 13, "active": 0}],
    ],
)
def test_exercise_schedules_use_half_open_intervals(exercise_data):
    runner = _make_runner()

    assert runner._get_scenario_exercise(9, exercise_data) is False
    assert runner._get_scenario_exercise(10, exercise_data) is True
    assert runner._get_scenario_exercise(12, exercise_data) is True
    assert runner._get_scenario_exercise(13, exercise_data) is False
    assert runner._get_scenario_exercise(20, exercise_data) is False


def test_active_exercise_transition_persists_after_last_change():
    runner = _make_runner()

    assert runner._get_scenario_exercise(20, [{"time": 10, "active": 1}]) is True


def test_runner_sums_coincident_meal_events():
    runner = _make_runner()
    applied_meals = []
    patient_step = runner.patient.step

    def record_patient_input(t, insulin_rate_u_h, meal_carbs_g, exercise_active):
        applied_meals.append(meal_carbs_g)
        return patient_step(t, insulin_rate_u_h, meal_carbs_g, exercise_active)

    runner.patient.step = record_patient_input
    runner.step(
        meal_data=[
            {"time": 0, "carbs": 10.0},
            {"time": 0, "carbs": 15.0},
        ]
    )

    assert applied_meals == [25.0]


def test_loader_rejects_active_raw_meals_without_structured_metadata(tmp_path):
    (tmp_path / "MealData_case99.data").write_text("0 0\n10 2.5\n")

    with pytest.raises(ValueError, match="structured meal metadata"):
        ScenarioLoader.load_case(99, tmp_path)
