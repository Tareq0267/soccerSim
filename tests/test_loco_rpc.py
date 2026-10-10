import json

import pytest

from soccer_sim.loco_rpc import (
    API_CHANGE_MODE, API_GET_MODE, API_GET_UP, API_MOVE, API_ROTATE_HEAD, API_VISUAL_KICK, KICK_V1, KICK_V2,
    MODE_PREPARE, MODE_SOCCER, MODE_WALKING, STATUS_BAD_REQUEST, STATUS_OK, LocoState, handle_request,
)


def req(api_id, **params):
    return json.dumps({"api_id": api_id}), json.dumps(params)


def status(reply):
    return json.loads(reply.header)["status"]


def test_move_sets_velocity():
    s = LocoState()
    r = handle_request(s, *req(API_MOVE, vx=0.5, vy=-0.2, vyaw=0.3))
    assert status(r) == STATUS_OK
    assert s.effective_velocity() == pytest.approx((0.5, -0.2, 0.3))


def test_move_is_clamped_to_policy_limits():
    s = LocoState()
    handle_request(s, *req(API_MOVE, vx=5.0, vy=1.2, vyaw=-3.0))
    assert s.effective_velocity() == pytest.approx((3.0, 0.8, -2.0))
    handle_request(s, *req(API_MOVE, vx=-2.0, vy=-1.2, vyaw=3.0))
    assert s.effective_velocity() == pytest.approx((-0.7, -0.8, 2.0))


def test_non_walking_modes_zero_the_velocity():
    s = LocoState()
    handle_request(s, *req(API_MOVE, vx=0.5, vy=0.0, vyaw=0.0))
    handle_request(s, *req(API_CHANGE_MODE, mode=MODE_PREPARE))
    assert s.effective_velocity() == (0.0, 0.0, 0.0)
    handle_request(s, *req(API_CHANGE_MODE, mode=MODE_WALKING))
    assert s.effective_velocity() == (0.0, 0.0, 0.0)  # old command does not resume by itself


def test_get_mode_reports_current_mode():
    s = LocoState()
    r = handle_request(s, *req(API_GET_MODE))
    assert status(r) == STATUS_OK
    assert json.loads(r.body) == {"mode": MODE_SOCCER}


def test_rotate_head_sets_targets():
    s = LocoState()
    handle_request(s, *req(API_ROTATE_HEAD, pitch=0.3, yaw=-0.5))
    assert (s.head_pitch, s.head_yaw) == (0.3, -0.5)


def test_get_up_sets_request_flag():
    s = LocoState()
    handle_request(s, *req(API_GET_UP))
    assert s.getup_requested


def test_visual_kick_start_and_stop():
    s = LocoState()
    assert status(handle_request(s, *req(API_VISUAL_KICK, start=True, version=KICK_V1))) == STATUS_OK
    assert s.kicking() and s.visual_kick_version == KICK_V1
    handle_request(s, *req(API_VISUAL_KICK, start=False, version=KICK_V2))
    assert not s.kicking()


def test_change_mode_ends_visual_kick_and_prepare_blocks_it():
    s = LocoState()
    handle_request(s, *req(API_VISUAL_KICK, start=True, version=KICK_V2))
    handle_request(s, *req(API_CHANGE_MODE, mode=MODE_SOCCER))
    assert not s.kicking()
    handle_request(s, *req(API_CHANGE_MODE, mode=MODE_PREPARE))
    handle_request(s, *req(API_VISUAL_KICK, start=True, version=KICK_V2))
    assert s.visual_kick_active and not s.kicking()  # flag set, but no kicking outside walking/soccer


def test_unknown_ids_are_acknowledged_no_ops():
    s = LocoState()
    assert status(handle_request(s, *req(9999, anything=1))) == STATUS_OK
    assert s.effective_velocity() == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "header,body",
    [
        ("not json", "{}"),
        (json.dumps({"no_api_id": 1}), "{}"),
        (json.dumps({"api_id": API_MOVE}), json.dumps({"vx": 0.1})),
        (json.dumps({"api_id": API_MOVE}), '{"vx": NaN, "vy": 0, "vyaw": 0}'),
        (json.dumps({"api_id": API_CHANGE_MODE}), json.dumps({"mode": 42})),
        (json.dumps({"api_id": API_ROTATE_HEAD}), '{"pitch": Infinity, "yaw": 0}'),
        (json.dumps({"api_id": API_VISUAL_KICK}), json.dumps({"start": "yes", "version": 1})),
        ('{"api_id": Infinity}', "{}"),  # int(inf) raises OverflowError
        (json.dumps({"api_id": API_CHANGE_MODE}), '{"mode": 1e999}'),
        (json.dumps({"api_id": API_VISUAL_KICK}), '{"start": true, "version": -Infinity}'),
    ],
)
def test_malformed_requests_are_rejected_without_state_change(header, body):
    s = LocoState()
    before = (s.mode, s.vx, s.vy, s.vyaw, s.head_pitch, s.head_yaw, s.visual_kick_active)
    r = handle_request(s, header, body)
    assert status(r) == STATUS_BAD_REQUEST
    assert (s.mode, s.vx, s.vy, s.vyaw, s.head_pitch, s.head_yaw, s.visual_kick_active) == before


def test_get_up_drops_the_stale_walk_command():
    s = LocoState()
    handle_request(s, *req(API_MOVE, vx=0.8, vy=0.0, vyaw=0.0))
    handle_request(s, *req(API_GET_UP))
    assert s.effective_velocity() == (0.0, 0.0, 0.0)


def test_visual_kick_stop_drops_the_pre_kick_walk_command():
    s = LocoState()
    handle_request(s, *req(API_MOVE, vx=0.8, vy=0.0, vyaw=0.0))
    handle_request(s, *req(API_VISUAL_KICK, start=True, version=KICK_V2))
    handle_request(s, *req(API_VISUAL_KICK, start=False, version=KICK_V2))
    assert s.effective_velocity() == (0.0, 0.0, 0.0)


def test_rotate_head_is_clamped_to_t2_joint_ranges():
    s = LocoState()
    handle_request(s, *req(API_ROTATE_HEAD, pitch=1.2, yaw=-2.0))
    assert (s.head_pitch, s.head_yaw) == pytest.approx((0.401426, -1.09956))
    handle_request(s, *req(API_ROTATE_HEAD, pitch=-1.0, yaw=2.0))
    assert (s.head_pitch, s.head_yaw) == pytest.approx((-0.314159, 1.09956))


def test_unknown_visual_kick_version_is_rejected():
    s = LocoState()
    r = handle_request(s, *req(API_VISUAL_KICK, start=True, version=7))
    assert status(r) == STATUS_BAD_REQUEST and not s.visual_kick_active
