import json
import urllib.error
import urllib.request

import pytest

from soccer_sim_bridge.web import ControlServer

pytestmark = pytest.mark.ros


@pytest.fixture
def server():
    calls = {"joystick": [], "referee": [], "shoot": []}

    def on_referee(payload):
        if payload.get("state") == "BOGUS":
            raise ValueError("unknown state 'BOGUS'")
        calls["referee"].append(payload)

    srv = ControlServer(0, calls["joystick"].append, on_referee, calls["shoot"].append, lambda: {"state": "PLAYING"})
    srv.start()
    yield f"http://127.0.0.1:{srv.port}", calls
    srv.stop()


def post(url, body: bytes, ctype="application/json"):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_serves_the_controller_page(server):
    base, _ = server
    with urllib.request.urlopen(base + "/", timeout=2) as r:
        assert r.status == 200 and b"Soccer Sim Controller" in r.read()


def test_status_endpoint(server):
    base, _ = server
    with urllib.request.urlopen(base + "/api/status", timeout=2) as r:
        assert json.loads(r.read()) == {"state": "PLAYING"}


def test_posts_reach_their_callbacks(server):
    base, calls = server
    assert post(base + "/api/joystick", b'{"lx": 0.5}')[0] == 200
    assert post(base + "/api/referee", b'{"state": "READY"}')[0] == 200
    assert post(base + "/api/shoot", b"{}")[0] == 200
    assert calls == {"joystick": [{"lx": 0.5}], "referee": [{"state": "READY"}], "shoot": [{}]}


def test_rejects_bad_json_wrong_type_and_oversized_bodies(server):
    base, calls = server
    assert post(base + "/api/joystick", b"{not json")[0] == 400
    assert post(base + "/api/joystick", b"[1, 2]")[0] == 400
    assert post(base + "/api/joystick", b'{"lx": 0.5}', ctype="text/plain")[0] == 415
    assert post(base + "/api/joystick", b"{" + b" " * 5000 + b"}")[0] == 413
    assert calls["joystick"] == []


def test_callback_validation_errors_become_400(server):
    base, _ = server
    status, body = post(base + "/api/referee", b'{"state": "BOGUS"}')
    assert status == 400 and "unknown state" in body["error"]


def test_overflowing_numbers_become_400_not_a_dropped_connection():
    def on_joystick(payload):
        float(payload["lx"])  # a 400-digit JSON integer raises OverflowError here

    srv = ControlServer(0, on_joystick, lambda p: None, lambda p: None, lambda: {})
    srv.start()
    try:
        status, body = post(f"http://127.0.0.1:{srv.port}/api/joystick", b'{"lx": 1' + b"0" * 400 + b"}")
        assert status == 400
    finally:
        srv.stop()


def test_unknown_paths_are_404(server):
    base, _ = server
    assert post(base + "/api/nope", b"{}")[0] == 404


def test_controller_page_recentres_sticks_and_releases_buttons_on_pointer_cancel(server):
    # A cancelled touch (scroll, palm, tab switch) never sends pointerup; a stuck stick keeps the robot walking.
    base, _ = server
    with urllib.request.urlopen(base + "/", timeout=2) as r:
        page = r.read()
    assert b'"lostpointercapture"' in page and b"onpointercancel" in page


def test_negative_content_length_is_400_not_a_hung_read(server):
    import http.client
    from urllib.parse import urlparse
    base, calls = server
    u = urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=2)
    conn.putrequest("POST", "/api/joystick")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", "-1")
    conn.endheaders()
    assert conn.getresponse().status == 400
    assert calls["joystick"] == []
