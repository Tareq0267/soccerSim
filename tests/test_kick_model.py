import math

import pytest

from soccer_sim.kick_model import KickParams, KickRef, VisualKickModel, ball_speed_for_power
from soccer_sim.loco_rpc import KICK_V1, KICK_V2

P = KickParams()


def ref(x, y, direction=0.0, power=3.0, stamp=0.0):
    return KickRef(ball_x=x, ball_y=y, dir=direction, power=power, stamp=stamp)


def test_strike_when_aligned_and_in_reach():
    out = VisualKickModel().step(ref(P.strike_ball_x, 0.0), 0.1, (P.strike_ball_x, 0.0), KICK_V2)
    assert out.strike_speed == pytest.approx(ball_speed_for_power(3.0, KICK_V2, P))
    assert (out.vx, out.vy, out.vyaw) == (0.0, 0.0, 0.0)


def test_approach_moves_toward_ball_and_turns_toward_kick_direction():
    out = VisualKickModel().step(ref(1.0, 0.3, direction=0.4), 0.1, (1.0, 0.3), KICK_V2)
    assert out.strike_speed == 0.0
    assert out.vx > 0.0 and out.vy > 0.0 and out.vyaw > 0.0


def test_approach_velocity_is_clamped():
    out = VisualKickModel().step(ref(10.0, -10.0, direction=-3.0), 0.1, (10.0, -10.0), KICK_V2)
    assert out.vx == pytest.approx(P.max_vx)
    assert out.vy == pytest.approx(-P.max_vy)
    assert out.vyaw == pytest.approx(-P.max_vyaw)


def test_no_strike_when_real_ball_out_of_reach():
    # brain believes the ball is at the foot, but the true ball is 2 m away
    out = VisualKickModel().step(ref(P.strike_ball_x, 0.0), 0.1, (2.0, 0.0), KICK_V2)
    assert out.strike_speed == 0.0


@pytest.mark.parametrize("true_ball", [(-0.4, 0.0), (0.1, 0.4), (0.0, -0.45)])
def test_no_strike_when_real_ball_is_beside_or_behind(true_ball):
    # brain's (wrong) reference says aligned, but the real ball is within 0.55 m behind/beside the robot
    out = VisualKickModel().step(ref(P.strike_ball_x, 0.0), 0.1, true_ball, KICK_V2)
    assert out.strike_speed == 0.0


def test_stale_or_missing_reference_stops_the_robot():
    m = VisualKickModel()
    assert m.step(None, 1.0, (1.0, 0.0), KICK_V2) == m.step(ref(1.0, 0.0, stamp=0.0), 1.0, (1.0, 0.0), KICK_V2)
    out = m.step(ref(1.0, 0.0, stamp=0.0), 1.0, (1.0, 0.0), KICK_V2)
    assert (out.vx, out.vy, out.vyaw, out.strike_speed) == (0.0, 0.0, 0.0, 0.0)


def test_cooldown_after_strike_then_ready_again():
    m = VisualKickModel()
    aligned = (P.strike_ball_x, 0.0)
    assert m.step(ref(*aligned, stamp=0.0), 0.0, aligned, KICK_V2).strike_speed > 0.0
    assert m.step(ref(*aligned, stamp=0.1), 0.1, aligned, KICK_V2).strike_speed == 0.0
    later = P.cooldown + 0.01
    assert m.step(ref(*aligned, stamp=later), later, aligned, KICK_V2).strike_speed > 0.0


def test_power_maps_to_rollout_distance():
    v = ball_speed_for_power(4.0, KICK_V1, P)
    assert v == pytest.approx(math.sqrt(2.0 * P.ball_decel * 4.0))
    assert ball_speed_for_power(-1.0, KICK_V1, P) == 0.0
    assert ball_speed_for_power(1e6, KICK_V2, P) == pytest.approx(P.max_ball_speed)


def test_v2_kicks_harder_than_v1():
    assert ball_speed_for_power(3.0, KICK_V2, P) > ball_speed_for_power(3.0, KICK_V1, P)


def test_min_approach_speed_keeps_a_dead_zone_policy_walking_but_not_a_proportional_one():
    ref = KickRef(ball_x=0.50, ball_y=0.0, dir=0.0, power=3.0, stamp=0.0)
    slow = VisualKickModel().step(ref, 0.0, (0.50, 0.0), KICK_V2)
    assert slow.vx == pytest.approx(0.15)  # 1.5 * 0.10 m
    boosted = VisualKickModel(KickParams(min_approach_speed=0.35)).step(ref, 0.0, (0.50, 0.0), KICK_V2)
    assert boosted.vx == pytest.approx(0.35) and boosted.vy == pytest.approx(0.0)
