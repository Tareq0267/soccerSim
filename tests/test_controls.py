import math

import pytest

from soccer_sim.config import SHOT_SLIDERS, FieldDims
from soccer_sim.joystick import FirmwareRemote, ManualLimits, manual_velocity
from soccer_sim.referee import Referee, parse_schedule
from soccer_sim.shots import Shot, ShotPanel, ShotTracker, shot_from_dict, shot_velocity

FD = FieldDims()
GOAL_LINE_X = -FD.length / 2
PANEL = {name: default for name, _lo, _hi, default in SHOT_SLIDERS}


# ---- shots ---------------------------------------------------------------
def test_shot_velocity_ground_and_lofted():
    assert shot_velocity(4.0, 180.0, 0.0) == pytest.approx((-4.0, 0.0, 0.0), abs=1e-9)
    assert shot_velocity(5.0, 90.0, 30.0) == pytest.approx((0.0, 5 * math.cos(math.radians(30)), 2.5), abs=1e-9)


def test_shot_from_dict_accepts_valid_input():
    shot = shot_from_dict({"x": -6, "y": 1, "speed": 5, "direction_deg": 170, "elevation_deg": 10})
    assert shot == Shot(-6.0, 1.0, 5.0, 170.0, 10.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"x": -6, "y": 1, "speed": 5, "direction_deg": 170},  # missing field
        {"x": float("nan"), "y": 1, "speed": 5, "direction_deg": 170, "elevation_deg": 0},
        {"x": -6, "y": 1, "speed": 99, "direction_deg": 170, "elevation_deg": 0},
        {"x": -6, "y": 1, "speed": 5, "direction_deg": 170, "elevation_deg": -5},
        {"x": -6, "y": 1, "speed": 5, "direction_deg": 170, "elevation_deg": 0, "spin": 3},
    ],
)
def test_shot_from_dict_rejects_bad_input(bad):
    with pytest.raises((KeyError, ValueError)):
        shot_from_dict(bad)


def test_panel_fires_once_per_drag_and_rearms():
    panel = ShotPanel()
    assert panel.read(PANEL) is None
    assert panel.read({**PANEL, "shot_fire": 0.9}) == Shot(-6.0, 0.0, 4.0, 180.0, 0.0)
    assert panel.read({**PANEL, "shot_fire": 1.0}) is None  # still held: no repeat
    assert panel.read({**PANEL, "shot_fire": 0.1}) is None  # released: re-armed
    assert panel.read({**PANEL, "shot_fire": 0.6}) is not None


def test_tracker_reports_goal_between_posts():
    t = ShotTracker(FD, goal_height=1.2)
    t.start(0.0)
    assert t.update(0.2, (-9.0, 0.0, 0.1), (-5.0, 0.0, 0.0)) is None
    assert t.update(0.6, (GOAL_LINE_X - 0.15, 0.5, 0.1), (-5.0, 0.0, 0.0)) == "goal"
    assert t.tally == {"goal": 1, "no_goal": 0}
    assert t.update(0.7, (GOAL_LINE_X - 0.2, 0.5, 0.1), (-5.0, 0.0, 0.0)) is None  # one result per shot


def test_tracker_wide_shot_and_stopped_ball_are_no_goal():
    t = ShotTracker(FD, goal_height=1.2)
    t.start(0.0)
    assert t.update(0.6, (GOAL_LINE_X - 0.15, 3.0, 0.1), (-5.0, 0.0, 0.0)) == "no_goal"  # wide of the post
    t.start(1.0)
    assert t.update(1.2, (-8.0, 0.0, 0.1), (0.0, 0.0, 0.0)) is None  # grace period right after launch
    assert t.update(1.6, (-8.0, 0.0, 0.1), (0.0, 0.0, 0.0)) == "no_goal"  # stopped: saved or died
    assert t.tally == {"goal": 0, "no_goal": 2}


def test_tracker_times_out():
    t = ShotTracker(FD, goal_height=1.2, timeout_s=8.0)
    t.start(0.0)
    assert t.update(8.1, (-8.0, 0.0, 0.1), (0.3, 0.0, 0.0)) == "no_goal"


def test_over_the_bar_is_no_goal():
    t = ShotTracker(FD, goal_height=1.2)
    t.start(0.0)
    assert t.update(0.6, (GOAL_LINE_X - 0.15, 0.0, 1.5), (-5.0, 0.0, 1.0)) == "no_goal"


def test_goal_needs_the_whole_ball_over_the_line():
    t = ShotTracker(FD, goal_height=1.2, ball_radius=0.105)
    t.start(0.0)
    assert t.update(0.1, (GOAL_LINE_X - 0.05, 0.0, 0.105), (-3.0, 0.0, 0.0)) is None  # centre over, ball on line
    assert t.update(0.2, (GOAL_LINE_X - 0.11, 0.0, 0.105), (-3.0, 0.0, 0.0)) == "goal"


# ---- gamepad -------------------------------------------------------------
def test_centered_sticks_leave_brain_in_control():
    assert manual_velocity(0.05, -0.08, 0.02, 0.0) is None


def test_stick_mapping_matches_booster_remote_axes():
    lim = ManualLimits()
    # RemoteControllerState: ly = -1 is "push front", lx/rx = -1 is "push left".
    assert manual_velocity(0.0, -1.0, 0.0, 0.0) == pytest.approx((lim.vx, 0.0, 0.0))
    assert manual_velocity(-1.0, 0.0, 0.0, 0.0) == pytest.approx((0.0, lim.vy, 0.0))
    assert manual_velocity(0.0, 0.0, -1.0, 0.0) == pytest.approx((0.0, 0.0, lim.vyaw))


def test_right_stick_vertical_alone_holds_the_robot_still():
    assert manual_velocity(0.0, 0.0, 0.0, 0.8) == (0.0, 0.0, 0.0)


def test_lt_up_fires_once_per_press():
    remote = FirmwareRemote()
    assert remote.update(lt=True, hat_u=True) is True
    assert remote.update(lt=True, hat_u=True) is False  # still held
    assert remote.update(lt=True, hat_u=False) is False  # released
    assert remote.update(lt=True, hat_u=True) is True


def test_up_without_lt_is_not_the_stand_up_combo():
    assert FirmwareRemote().update(lt=False, hat_u=True) is False


# ---- referee -------------------------------------------------------------
def test_schedule_runs_states_in_order_then_holds_last():
    ref = Referee(parse_schedule("0:3,1:20,2:3,3:0"), 131, 1, t0=100.0)
    states = [ref.snapshot(100.0 + t).state for t in (0, 2.9, 3.1, 22.9, 23.1, 26.1, 999)]
    assert states == [0, 0, 1, 1, 2, 3, 3]


def test_override_holds_state_over_the_schedule():
    ref = Referee(parse_schedule("0:3,1:20,2:3,3:0"), 131, 1, t0=0.0)
    ref.override(1.0, state="PLAYING")
    assert ref.snapshot(1.5).state == 3
    assert ref.snapshot(50.0).state == 3


def test_override_back_to_schedule_resumes_the_timeline():
    ref = Referee(parse_schedule("0:3,1:20,2:3,3:0"), 131, 1, t0=0.0)
    ref.override(1.0, state="FINISHED")
    ref.override(2.0, state="SCHEDULE")
    assert ref.snapshot(10.0).state == 1


def test_opponent_corner_kick_counts_down_then_clears():
    ref = Referee(parse_schedule("3:0"), 131, 1, t0=0.0)
    ref.override(10.0, set_play="CORNER_KICK", kicking="opponent", stopped=True)
    snap = ref.snapshot(10.0)
    assert (snap.set_play, snap.kicking_team, snap.stopped, snap.secondary_time) == (6, 1, True, 30)
    assert ref.snapshot(25.2).secondary_time == 15
    cleared = ref.snapshot(40.5)
    assert (cleared.set_play, cleared.stopped, cleared.secondary_time) == (0, False, 0)


def test_our_free_kick_uses_our_team_number():
    ref = Referee(parse_schedule("3:0"), 131, 1, t0=0.0)
    ref.override(0.0, set_play="direct_free_kick", kicking="us")
    assert ref.snapshot(1.0).kicking_team == 131


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state": "HALFTIME"},
        {"set_play": "OFFSIDE"},
        {"kicking": "them"},
        {"stopped": "yes"},
        {"state": "PLAYING", "set_play": "NOPE"},
    ],
)
def test_invalid_override_changes_nothing(kwargs):
    ref = Referee(parse_schedule("0:3,1:20"), 131, 1, t0=0.0)
    before = ref.snapshot(1.0)
    with pytest.raises(ValueError):
        ref.override(1.0, **kwargs)
    assert ref.snapshot(1.0) == before
