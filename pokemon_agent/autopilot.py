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
  POKEMON_HERMES_MODEL            model override passed to `hermes chat -m`
  POKEMON_HERMES_PROVIDER         provider override passed to `hermes chat --provider`
  POKEMON_HERMES_DISABLE_IMAGE    set to "1" to never pass --image to hermes
                                  (auto-enabled when provider=ollama unless the model
                                  is in the vision allowlist)
  POKEMON_HERMES_VISION_ALLOWLIST comma-separated model name prefixes to add to the
                                  built-in Ollama vision allowlist (e.g. "llava,moondream")
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .sessions import GameSessionManager

# Model name prefixes (lowercase) known to correctly handle image input via Ollama's
# OpenAI-compatible API. Verified by direct API test: the model receives and describes
# the image without producing garbled special-token output.
# mistral3 is explicitly excluded — it produces broken output with image input via Ollama.
OLLAMA_VISION_ALLOWLIST: frozenset[str] = frozenset({"gemma4"})


def _ollama_model_supports_vision(model: str) -> bool:
    """Return True if model is in the Ollama vision allowlist (built-in + env override)."""
    extra = {p.strip().lower() for p in
             os.environ.get("POKEMON_HERMES_VISION_ALLOWLIST", "").split(",") if p.strip()}
    allowlist = OLLAMA_VISION_ALLOWLIST | extra
    base = model.split(":")[0].lower()
    return any(base.startswith(prefix) for prefix in allowlist)


# What Hermes is told once at the start of the session, then nudged each turn.
TURN_NUDGE = """You are playing Pokémon Red live on the Hermes Plays Pokémon dashboard.

The game server is at {server}. Take ONE short turn now, then stop and reply.

This turn:
1. {vision_instruction}
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

TEXT WINDOW / DIALOG HANDLING (do this FIRST every turn):
- If a text window / dialog box is on screen (check `dialog_active` in CURRENT STATE),
  the game is waiting for you — you CANNOT walk. Advance it with `press_a`.
- Send `press_a` repeatedly (e.g. {{"actions":["press_a","press_a","press_a"]}})
  until the text window is fully closed (`dialog_active` is false). Only then
  resume walking.
- This applies to NPC/Oak speeches, signs, item pickups, battle prompts, and
  scripted events. Do NOT try to walk while a text window is open — it does
  nothing and wastes the turn.

Use the ASCII map for WHERE you can walk — it is the authoritative navigation source.

CURRENT STATE:
{state}

WALKABILITY MAP (you are @ at E5):
{ascii_map}

Take your turn now."""

FIRST_TURN_PREFIX = """This is the start of your Pokémon Red run. First, set your objectives by
POSTing to {server}/objectives (primary/secondary/tertiary tiers), then take
your first turn as described below.

"""


def _is_stuck(state: Dict[str, Any]) -> bool:
    """Return True when no walkable cell exists outside the player's own cell (E5)."""
    walkable = (state.get("collision") or {}).get("walkable")
    if not walkable:
        return False
    for r, row in enumerate(walkable):
        for c, cell in enumerate(row):
            if (r, c) != (4, 4) and cell:
                return False
    return True


def _extract_state_after(events: list) -> Optional[Dict[str, Any]]:
    """Return state_after from the last action event in the turn's event list."""
    for ev in reversed(events):
        if ev.get("type") == "action" and "state_after" in ev:
            return ev["state_after"]
    return None


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
                 turn_timeout: int = 480, data_dir: str = "~/.pokemon-agent",
                 disable_image: bool = False):
        self.server = server.rstrip("/")
        self.model = model
        self.provider = provider
        self.turn_delay = turn_delay
        self.save_every = save_every
        self.turn_timeout = turn_timeout
        self.data_dir = data_dir
        # Disable --image when the provider is ollama AND the model is not in the vision
        # allowlist, or when explicitly forced via env/arg.
        # Ollama's OpenAI-compatible API forwards images incorrectly for some GGUF
        # multimodal models (e.g. mistral3 produces garbled special-token output), but
        # works correctly for others (e.g. gemma4).
        ollama_no_vision = (
            provider == "ollama" and
            not _ollama_model_supports_vision(model or "")
        )
        self.disable_image = disable_image or ollama_no_vision
        self._session_mgr = GameSessionManager(data_dir)
        self.game_id: Optional[str] = None        # active game session id
        self.session_id: Optional[str] = None     # bound Hermes session id
        self.turn = 0

    # --- session id helpers ---
    def _capture_session_from_output(self, text: str) -> bool:
        """Try to extract session id from hermes stdout/stderr text. Returns True on success."""
        m = re.search(r"hermes --resume (\S+)", text) or \
            re.search(r"Session:\s*(\S+)", text) or \
            re.search(r"session_id:\s*(\S+)", text)
        if m:
            self.session_id = m.group(1)
            print(f"[driver] Hermes session: {self.session_id}")
            self.bind_hermes()
            return True
        return False

    def _capture_session_from_list(self) -> None:
        """Fallback: query `hermes sessions list` to find the most recently created session."""
        print("[driver] _capture_session_from_list: querying hermes sessions list",
              file=sys.stderr)
        sys.stderr.flush()
        try:
            r = subprocess.run(["hermes", "sessions", "list", "--limit", "1"],
                               capture_output=True, text=True, timeout=10)
            print(f"[driver] sessions list rc={r.returncode} "
                  f"stdout={r.stdout[:300]!r}", file=sys.stderr)
            sys.stderr.flush()
            for line in r.stdout.splitlines():
                m = re.search(r"\b(\d{8}_\d{6}_\w+)\b", line)
                if m:
                    self.session_id = m.group(1)
                    print(f"[driver] Hermes session (from sessions list): {self.session_id}",
                          file=sys.stderr)
                    sys.stderr.flush()
                    self.bind_hermes()
                    return
            print("[driver] sessions list: no session id found", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"[driver] sessions list fallback failed: {e}", file=sys.stderr)
            sys.stderr.flush()

    # --- server helpers ---
    def _get(self, path: str):
        return requests.get(self.server + path, timeout=15)

    def control_state(self) -> str:
        try:
            return self._get("/control").json().get("state", "stopped")
        except Exception:
            return "stopped"

    def sync_active_game(self) -> None:
        """Read the active game session and adopt its id + Hermes brain id.

        This is how 'load game' on the dashboard takes effect: the driver
        resumes the SAME Hermes session that game was played with, and scopes
        its work to that game.
        """
        try:
            cur = self._get("/games/current").json().get("active")
        except Exception:
            cur = None
        if not cur:
            self.game_id = None
            return
        if cur.get("id") != self.game_id:
            # switched to a different game (new or loaded) — adopt its brain
            self.game_id = cur.get("id")
            self.session_id = cur.get("hermes_session_id")  # may be None for a new game
            print(f"[driver] active game: {self.game_id} (hermes={self.session_id})")

    def event(self, **kw):
        try:
            requests.post(self.server + "/event", json=kw, timeout=15)
        except Exception:
            pass

    def _fetch_turn_events(self) -> list:
        """Drain the server's per-turn event buffer."""
        try:
            return self._get("/turn/events").json().get("events", [])
        except Exception as e:
            print(f"[driver] failed to fetch turn events: {e}", file=sys.stderr)
            return []

    def _save_frame(self, state_before: Dict[str, Any], img_bytes: Optional[bytes],
                    hermes_output: Optional[str] = None,
                    hermes_input: Optional[str] = None,
                    quality: str = "ok",
                    events: Optional[list] = None,
                    state_after: Optional[Dict[str, Any]] = None) -> None:
        if not self.game_id:
            return
        try:
            frames = self._session_mgr.frames_dir(self.game_id)
            turn_num = self.turn + 1
            stem = f"turn_{turn_num:04d}"
            if img_bytes is not None:
                (frames / f"{stem}.png").write_bytes(img_bytes)
            record = {
                "turn": turn_num,
                "session_id": self.game_id,
                "hermes_session_id": self.session_id,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "quality": quality,
                "hermes_input": hermes_input,
                "hermes_input_image": f"{stem}.png" if img_bytes is not None else None,
                "hermes_output": hermes_output,
                "events": events or [],
                "state_before": state_before,
                "state_after": state_after,
            }
            (frames / f"{stem}.json").write_text(json.dumps(record, indent=2))
        except Exception as e:
            print(f"[driver] frame save failed: {e}", file=sys.stderr)

    def bind_hermes(self):
        if self.game_id and self.session_id:
            try:
                requests.post(f"{self.server}/games/{self.game_id}/hermes",
                              json={"hermes_session_id": self.session_id}, timeout=15)
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

        # Flush any leftover events from the previous turn before this one starts.
        self._fetch_turn_events()

        # Grab the grid screenshot (always save for frame records; only pass to hermes
        # when image input is not disabled).
        img_path = "/tmp/pokemon_turn_grid.png"
        shot: Optional[bytes] = None
        try:
            shot = self._get("/screenshot/grid?scale=3").content
            with open(img_path, "wb") as f:
                f.write(shot)
            have_img = True
        except Exception:
            have_img = False

        use_img = have_img and not self.disable_image
        if use_img:
            vision_instruction = (
                "Look at the attached grid screenshot (A1..J9 cells, you are the player at"
                " E5; the labelled grid + green/red walkability tint is drawn on it)."
                " Use it to identify WHAT things are (doors, signs, NPCs, buildings)."
            )
        else:
            vision_instruction = (
                "No screenshot is available this turn — rely entirely on the ASCII"
                " walkability map and CURRENT STATE below to decide your move."
            )

        prompt = TURN_NUDGE.format(
            server=self.server,
            state=json.dumps(_compact_state(state), indent=2),
            ascii_map=ascii_map,
            vision_instruction=vision_instruction,
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
        if use_img:
            cmd += ["--image", img_path]
        cmd += ["-q", prompt]

        print(f"[driver] turn {self.turn + 1}: calling hermes (timeout={self.turn_timeout}s)",
              file=sys.stderr)
        sys.stderr.flush()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=self.turn_timeout)
            stdout = out.stdout or ""
            stderr = out.stderr or ""
            print(f"[driver] turn {self.turn + 1}: hermes rc={out.returncode} "
                  f"stdout={len(stdout)}B stderr={len(stderr)}B", file=sys.stderr)
            sys.stderr.flush()
        except subprocess.TimeoutExpired as te:
            print(f"[driver] turn {self.turn + 1}: timed out after {self.turn_timeout}s",
                  file=sys.stderr)
            sys.stderr.flush()
            # subprocess.run() kills the child and populates te.stdout/te.stderr before raising.
            extra_stdout = te.stdout or ""
            extra_stderr = te.stderr or ""
            if isinstance(extra_stdout, bytes):
                extra_stdout = extra_stdout.decode("utf-8", errors="replace")
            if isinstance(extra_stderr, bytes):
                extra_stderr = extra_stderr.decode("utf-8", errors="replace")
            print(f"[driver] post-timeout output: stdout={len(extra_stdout)}B "
                  f"stderr={len(extra_stderr)}B", file=sys.stderr)
            sys.stderr.flush()
            self.event(type="alert", text="Turn timed out — retrying.")
            if self.session_id is None:
                combined = extra_stdout + "\n" + extra_stderr
                if not self._capture_session_from_output(combined):
                    self._capture_session_from_list()
            turn_events = self._fetch_turn_events()
            self._save_frame(state, shot, hermes_output=None, hermes_input=prompt,
                             quality="timeout", events=turn_events,
                             state_after=_extract_state_after(turn_events))
            self.turn += 1
            return
        except Exception as e:
            print(f"[driver] hermes invocation failed: {e}", file=sys.stderr)
            self.event(type="alert", text=f"Driver error: {e}")
            turn_events = self._fetch_turn_events()
            self._save_frame(state, shot, hermes_output=None, hermes_input=prompt,
                             quality="error", events=turn_events,
                             state_after=_extract_state_after(turn_events))
            time.sleep(3)
            return

        turn_events = self._fetch_turn_events()
        quality = "error" if not stdout else ("stuck" if _is_stuck(state) else "ok")
        self._save_frame(state, shot, hermes_output=stdout, hermes_input=prompt,
                         quality=quality, events=turn_events,
                         state_after=_extract_state_after(turn_events))

        # Capture the session id from the first run so later turns resume it.
        # session_id appears in stderr (hermes banner) after process exits.
        if self.session_id is None:
            if not self._capture_session_from_output(stdout + "\n" + stderr):
                self._capture_session_from_list()
            if self.session_id:
                self.event(type="key_moment",
                           description="Hermes session started",
                           category="milestone")

        self.turn += 1

    def run(self):
        model_note = self.model or "config default"
        print(f"[driver] Hermes-driven autopilot. server={self.server} model={model_note}")
        print("[driver] waiting for control=running + an active game…")
        self.event(type="alert", text="Hermes online — start or load a game, then press START.")
        idle_logged = False
        no_game_logged = False
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
            self.sync_active_game()
            if not self.game_id:
                if not no_game_logged:
                    print("[driver] running but no active game — start/load one on the dashboard.")
                    self.event(type="alert", text="No active game — click New Game or load one.")
                    no_game_logged = True
                time.sleep(2)
                continue
            no_game_logged = False
            self.step()
            time.sleep(self.turn_delay)


def run_autopilot(server: str = "http://localhost:8765", model: Optional[str] = None,
                  turn_delay: float = 1.5, turn_timeout: Optional[int] = None,
                  data_dir: str = "~/.pokemon-agent"):
    model = model or os.environ.get("POKEMON_HERMES_MODEL")
    provider = os.environ.get("POKEMON_HERMES_PROVIDER")
    if turn_timeout is None:
        env_val = os.environ.get("POKEMON_HERMES_TIMEOUT")
        turn_timeout = int(env_val) if env_val else 480
    disable_image = os.environ.get("POKEMON_HERMES_DISABLE_IMAGE", "").strip() == "1"
    HermesDriver(server, model, provider, turn_delay=turn_delay,
                 turn_timeout=turn_timeout, data_dir=data_dir,
                 disable_image=disable_image).run()
