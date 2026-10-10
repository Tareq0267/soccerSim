"""GameController stand-in: a timed schedule that the controller tab can override, incl. set plays."""

import math
from dataclasses import dataclass

STATES = {"INITIAL": 0, "READY": 1, "SET": 2, "PLAYING": 3, "FINISHED": 4}
SET_PLAYS = {
    "NONE": 0,
    "DIRECT_FREE_KICK": 1,
    "INDIRECT_FREE_KICK": 2,
    "PENALTY_KICK": 3,
    "THROW_IN": 4,
    "GOAL_KICK": 5,
    "CORNER_KICK": 6,
}
SET_PLAY_SECS = 30.0


def name_of(table: dict[str, int], value: int) -> str:
    return next(name for name, v in table.items() if v == value)


def parse_schedule(text: str) -> list[tuple[int, float]]:
    """"0:3,1:20,2:3,3:0" -> [(0, 3.0), (1, 20.0), (2, 3.0), (3, 0.0)]; the last entry is held."""
    out = []
    for item in text.split(","):
        state_text, secs_text = item.split(":")
        state, secs = int(state_text), float(secs_text)
        if state not in STATES.values() or secs < 0:
            raise ValueError(f"bad schedule entry {item!r}")
        out.append((state, secs))
    if not out:
        raise ValueError("empty schedule")
    return out


def state_at(schedule: list[tuple[int, float]], elapsed: float) -> int:
    t = 0.0
    for state, secs in schedule[:-1]:
        t += secs
        if elapsed < t:
            return state
    return schedule[-1][0]


@dataclass(frozen=True)
class RefereeSnapshot:
    state: int
    set_play: int
    kicking_team: int
    stopped: bool
    secondary_time: int


def _lookup(table: dict[str, int], name, what: str) -> int:
    if not isinstance(name, str) or name.upper() not in table:
        raise ValueError(f"unknown {what} {name!r}; expected one of {list(table)}")
    return table[name.upper()]


class Referee:
    def __init__(self, schedule: list[tuple[int, float]], team_id: int, opp_team_id: int, t0: float):
        self.schedule = schedule
        self.team_id = team_id
        self.opp_team_id = opp_team_id
        self.t0 = t0
        self._manual_state: int | None = None
        self.set_play = 0
        self.kicking_team = team_id
        self.stopped = False
        self._set_play_started: float | None = None

    def override(self, now: float, state=None, set_play=None, kicking=None, stopped=None) -> None:
        # Validate everything first so a bad request never half-applies.
        resume = isinstance(state, str) and state.upper() == "SCHEDULE"
        new_state = None if state is None or resume else _lookup(STATES, state, "state")
        new_set_play = None if set_play is None else _lookup(SET_PLAYS, set_play, "set_play")
        if kicking is not None and kicking not in ("us", "opponent"):
            raise ValueError(f"kicking must be 'us' or 'opponent', got {kicking!r}")
        if stopped is not None and not isinstance(stopped, bool):
            raise ValueError("stopped must be a bool")

        if resume:
            self._manual_state = None
        elif new_state is not None:
            self._manual_state = new_state
        if new_set_play is not None:
            self.set_play = new_set_play
            self._set_play_started = now if new_set_play != 0 else None
        if kicking is not None:
            self.kicking_team = self.team_id if kicking == "us" else self.opp_team_id
        if stopped is not None:
            self.stopped = stopped

    def snapshot(self, now: float) -> RefereeSnapshot:
        state = self._manual_state if self._manual_state is not None else state_at(self.schedule, now - self.t0)
        secondary = 0
        if self.set_play != 0 and self._set_play_started is not None:
            remaining = SET_PLAY_SECS - (now - self._set_play_started)
            if remaining <= 0.0:
                self.set_play, self._set_play_started, self.stopped = 0, None, False
            else:
                secondary = int(math.ceil(remaining))
        return RefereeSnapshot(state, self.set_play, self.kicking_team, self.stopped, secondary)
