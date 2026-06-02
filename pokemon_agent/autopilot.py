"""Standalone driver that lets **Hermes Agent** play Pokemon through a session.

This is NOT a raw-LLM loop. The brain is a real Hermes Agent session — with
the `pokemon-player` skill, vision, memory, and the terminal tool — driven one
turn at a time. The driver is intentionally thin:

  loop while /control == "running":
      hermes chat --resume <session> --yolo -s pokemon-player \\
        --image <grid screenshot> -q "<turn nudge + compact state + ascii map>"

Hermes itself does the work each turn: it reads the state/map we hand it (and
can curl the server for more), looks at the grid screenshot with its own
vision, decides, then calls the game server's HTTP API with its terminal tool
to POST /action and POST /event (narration) and POST /objectives. Because we
pass --resume with a single persistent session id, Hermes keeps memory and
context across the whole playthrough — it is "running through a session."

The loop is gated by the server's /control state (Start/Pause/Stop buttons).

Config (env, optional):
  POKEMON_HERMES_MODEL     model override passed to `hermes chat -m`
  POKEMON_HERMES_PROVIDER  provider override passed to `hermes chat --provider`
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import requests

# What Hermes is told once at the start of the session, then nudged each turn.
TURN_NUDGE = """You are playing Pokémon Red live on the Hermes Plays Pokémon dashboard.

The game server is at {server}. Take ONE short turn now, then stop and reply.

This turn:
1. Look at the attached grid screenshot (A1..J9 cells, you are the player at
   E5; the labelled grid + green/red walkability tint is drawn on it).
2. Use the game state and the ASCII walkability map below to decide a move.
   `.` = walkable, `#` = blocked, `@` = you (E5). Count cells from E5:
   up=row-1, down=row+1, left=col-1, right=col+1. NEVER route through `#`.
3. Narrate to the stream, then act, using the terminal tool with curl:
   - POST {server}/event  body {{"type":"reasoning","text":"..."}}  (what you see)
   - POST {server}/event  body {{"type":"decision","text":"..."}}   (your plan)
   - POST {server}/action body {{"actions":["walk_down","walk_down"]}} (2-4 moves)
   - On a real beat (new town/badge/item/catch): POST {server}/event
     body {{"type":"key_moment","description":"...","category":"milestone|badge|catch"}}
   - If your goals change: POST {server}/objectives body
     {{"objectives":[{{"tier":"primary","text":"...","done":false}}, ...]}}
   All POSTs need  -H 'Content-Type: application/json'.
4. Keep it to 2-4 game actions this turn — you'll get another turn next.

Reserve vision (the screenshot) for identifying WHAT things are (doors, signs,
NPCs, the Mart's blue roof). Use the ASCII map for WHERE you can walk.

CURRENT STATE:
{state}

WALKABILITY MAP (you are @ at E5):
{ascii_map}

Take your turn now."""

FIRST_TURN_PREFIX = """This is the start of your Pokémon Red run. First, set your objectives by
POSTing to {server}/objectives (primary/secondary/tertiary tiers), then take
your first turn as described below.

"""


def _compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    p = state.get("player", {}) or {}
    party = []
    for m in state.get("party", []) or []:
        party.append({
            "nickname": m.get("nickname"), "species": m.get("species"),
            "level": m.get("level"), "hp": m.get("hp"), "max_hp": m.get("max_hp"),
            "status": m.get("status"), "types": m.get("types"),
            "moves": [mv.get("name") if isinstance(mv, dict) else mv for mv in m.get("moves", [])],
        })
    battle = state.get("battle") or {}
    enemy = battle.get("enemy") or {}
    return {
        "map": (state.get("map") or {}).get("map_name"),
        "position": p.get("position"), "facing": p.get("facing"),
        "cell": (state.get("collision") or {}).get("player_cell", "E5"),
        "money": p.get("money"), "badges": p.get("badges"),
        "party": party,
        "dialog_active": (state.get("dialog") or {}).get("active"),
        "in_battle": battle.get("in_battle"),
        "enemy": ({"species": enemy.get("species"), "level": enemy.get("level"),
                   "hp": enemy.get("hp"), "max_hp": enemy.get("max_hp")}
                  if battle.get("in_battle") else None),
    }


class HermesDriver:
    def __init__(self, server: str, model: Optional[str], provider: Optional[str],
                 turn_delay: float = 1.5, save_every: int = 20,
                 turn_timeout: int = 240):
        self.server = server.rstrip("/")
        self.model = model
        self.provider = provider
        self.turn_delay = turn_delay
        self.save_every = save_every
        self.turn_timeout = turn_timeout
        self.session_id: Optional[str] = None
        self.turn = 0

    # --- server helpers ---
    def _get(self, path: str):
        return requests.get(self.server + path, timeout=15)

    def control_state(self) -> str:
        try:
            return self._get("/control").json().get("state", "stopped")
        except Exception:
            return "stopped"

    def event(self, **kw):
        try:
            requests.post(self.server + "/event", json=kw, timeout=15)
        except Exception:
            pass

    # --- one turn = one Hermes invocation ---
    def step(self) -> None:
        try:
            state = self._get("/state").json()
        except Exception as e:
            print(f"[driver] state read failed: {e}", file=sys.stderr)
            time.sleep(2)
            return
        ascii_map = (state.get("collision") or {}).get("ascii")
        if not ascii_map:
            ascii_map = ("(in battle — no overworld map this turn)"
                         if (state.get("battle") or {}).get("in_battle")
                         else "(no map available)")

        # Grab the grid screenshot to a temp file for --image.
        img_path = "/tmp/pokemon_turn_grid.png"
        try:
            shot = self._get("/screenshot/grid?scale=3").content
            with open(img_path, "wb") as f:
                f.write(shot)
            have_img = True
        except Exception:
            have_img = False

        prompt = TURN_NUDGE.format(
            server=self.server,
            state=json.dumps(_compact_state(state), indent=2),
            ascii_map=ascii_map,
        )
        if self.session_id is None:
            prompt = FIRST_TURN_PREFIX.format(server=self.server) + prompt

        cmd = ["hermes", "chat", "-Q", "--yolo", "--pass-session-id",
               "-s", "pokemon-player"]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        if self.model:
            cmd += ["-m", self.model]
        if self.provider:
            cmd += ["--provider", self.provider]
        if have_img:
            cmd += ["--image", img_path]
        cmd += ["-q", prompt]

        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=self.turn_timeout)
            stdout = out.stdout or ""
        except subprocess.TimeoutExpired:
            print("[driver] hermes turn timed out", file=sys.stderr)
            self.event(type="alert", text="Turn timed out — retrying.")
            return
        except Exception as e:
            print(f"[driver] hermes invocation failed: {e}", file=sys.stderr)
            self.event(type="alert", text=f"Driver error: {e}")
            time.sleep(3)
            return

        # Capture the session id from the first run so later turns resume it.
        if self.session_id is None:
            m = re.search(r"hermes --resume (\S+)", stdout) or \
                re.search(r"Session:\s*(\S+)", stdout)
            if m:
                self.session_id = m.group(1)
                print(f"[driver] Hermes session: {self.session_id}")
                self.event(type="key_moment",
                           description="Hermes session started",
                           category="milestone")

        self.turn += 1

    def run(self):
        model_note = self.model or "config default"
        print(f"[driver] Hermes-driven autopilot. server={self.server} model={model_note}")
        print("[driver] waiting for control=running (dashboard Start button)…")
        self.event(type="alert", text="Hermes online — press START to play.")
        idle_logged = False
        while True:
            st = self.control_state()
            if st == "stopped":
                if not idle_logged:
                    print("[driver] stopped — idling.")
                    idle_logged = True
                time.sleep(2)
                continue
            if st == "paused":
                time.sleep(1.5)
                continue
            idle_logged = False
            self.step()
            time.sleep(self.turn_delay)


def run_autopilot(server: str = "http://localhost:8765", model: Optional[str] = None,
                  turn_delay: float = 1.5):
    model = model or os.environ.get("POKEMON_HERMES_MODEL")
    provider = os.environ.get("POKEMON_HERMES_PROVIDER")
    HermesDriver(server, model, provider, turn_delay=turn_delay).run()
