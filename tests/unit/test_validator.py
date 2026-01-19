from oellm_autoexp.sweep.planner import JobPlan
from oellm_autoexp.sweep.validator import validate_execution_plan


def test_validate_execution_plan_valid():
    jobs = [
        JobPlan(
            name="j1",
            parameters=[],
            output_dir="/o",
            log_path="/l",
            log_path_current="/c",
            config={},
        )
    ]
    assert validate_execution_plan(jobs).is_valid


def test_validate_execution_plan_duplicate_names():
    jobs = [
        JobPlan(
            name="j1",
            parameters=[],
            output_dir="/o",
            log_path="/l",
            log_path_current="/c",
            config={},
        ),
        JobPlan(
            name="j1",
            parameters=[],
            output_dir="/o2",
            log_path="/l2",
            log_path_current="/c2",
            config={},
        ),
    ]
    assert not validate_execution_plan(jobs).is_valid
