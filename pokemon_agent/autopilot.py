"""Standalone autopilot — lets pokemon-agent play itself via an LLM.

`pokemon-agent play` runs an OBSERVE -> THINK -> ACT loop against a running
game server:

  1. OBSERVE  GET /state (+ embedded collision grid) and GET /screenshot/grid
  2. THINK    send the compact state, the ASCII walkability map, and the
              grid screenshot to a vision LLM; it returns JSON:
              {reasoning, decision, actions[], milestone?, objectives?}
  3. ACT      POST narration to /event, then POST the actions to /action
  4. periodically POST /save

The loop is gated by the server's /control state: it only acts while
"running", idles while "paused", and exits on "stopped". This is what the
dashboard's Start / Pause / Stop buttons drive.

LLM config (env, all optional — defaults to OpenRouter):
  POKEMON_LLM_BASE_URL   default https://openrouter.ai/api/v1
  POKEMON_LLM_API_KEY    default $OPENROUTER_API_KEY
  POKEMON_LLM_MODEL      default anthropic/claude-sonnet-4.5  (must be vision-capable)
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

SYSTEM_PROMPT = """You are Hermes, an AI playing Pokémon Red autonomously on a live stream.

Each turn you receive: the structured game state (party, position, dialog,
battle, badges), a ground-truth ASCII walkability map, and a screenshot with a
labelled A1..J9 grid (you are ALWAYS at cell E5; green = walkable, red = blocked).

NAVIGATION RULES (critical):
- Use the ASCII map to decide WHERE to move — it is exact. `.` = walkable,
  `#` = blocked, `@` = you (E5). Count cells from E5: up=row-1, down=row+1,
  left=col-1, right=col+1. NEVER plan a move through a `#` cell.
- Use the screenshot to identify WHAT things are (NPCs, signs, doors, the
  Mart's blue roof, the Center's red roof). Doors read as walkable on the map.
- Move 2-4 steps at a time, then you'll re-observe. Don't send long blind paths.
- After walking through a door/stairs the screen fades — add wait_60 once or twice.

GAMEPLAY:
- If dialog is active: advance it (a_until_dialog_end, or press_a).
- If in battle: pick a good move (super-effective if possible), or run from
  trash wild battles. Squirtle's Water beats Rock/Ground/Fire.
- Priority: dialog > battle > heal if hurt > story objective > explore.

Respond with ONLY a JSON object, no prose around it:
{
  "reasoning": "1-2 sentences: what the map/screen shows and your read",
  "decision": "the concrete move you're about to make",
  "actions": ["walk_down","walk_down"],     // 1-4 action strings
  "milestone": "short text" or null,         // set ONLY on a real beat (new town/badge/item/catch)
  "objectives": [                            // OPTIONAL: only when goals change
    {"tier":"primary","text":"...","done":false}
  ]
}

Valid actions: press_a, press_b, press_start, press_select,
walk_up, walk_down, walk_left, walk_right, wait_60, a_until_dialog_end, hold_b_120."""


def _compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Trim /state to what the model needs (drop the raw collision grid)."""
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
        "enemy": {"species": enemy.get("species"), "level": enemy.get("level"),
                  "hp": enemy.get("hp"), "max_hp": enemy.get("max_hp")} if battle.get("in_battle") else None,
    }


class Autopilot:
    def __init__(self, server: str, model: str, base_url: str, api_key: str,
                 turn_delay: float = 1.5, save_every: int = 20):
        self.server = server.rstrip("/")
        self.turn_delay = turn_delay
        self.save_every = save_every
        self.turn = 0
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    # --- server helpers ---
    def _get(self, path: str):
        return requests.get(self.server + path, timeout=15)

    def _post(self, path: str, body: dict):
        return requests.post(self.server + path, json=body, timeout=30)

    def control_state(self) -> str:
        try:
            return self._get("/control").json().get("state", "stopped")
        except Exception:
            return "stopped"

    def event(self, **kw):
        try:
            self._post("/event", kw)
        except Exception:
            pass

    # --- one turn ---
    def step(self) -> bool:
        """Run one OBSERVE->THINK->ACT cycle. Returns False to stop."""
        try:
            state = self._get("/state").json()
        except Exception as e:
            print(f"[autopilot] state read failed: {e}", file=sys.stderr)
            return True
        ascii_map = (state.get("collision") or {}).get("ascii", "(no map)")
        try:
            shot = self._get("/screenshot/grid?scale=3").content
            b64 = base64.b64encode(shot).decode("ascii")
        except Exception:
            b64 = None

        user_text = (
            "GAME STATE:\n" + json.dumps(_compact_state(state), indent=2) +
            "\n\nWALKABILITY MAP (you are @ at E5):\n" + ascii_map +
            "\n\nDecide your next move. Respond with the JSON object only."
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if b64:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + b64}})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": content}],
                temperature=0.4, max_tokens=600,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            plan = json.loads(raw)
        except Exception as e:
            print(f"[autopilot] LLM call failed: {e}", file=sys.stderr)
            self.event(type="alert", text=f"LLM error: {e}")
            time.sleep(3)
            return True

        reasoning = (plan.get("reasoning") or "").strip()
        decision = (plan.get("decision") or "").strip()
        actions = plan.get("actions") or []
        if isinstance(actions, str):
            actions = [actions]
        actions = [str(a).strip() for a in actions if a][:4]

        if reasoning:
            self.event(type="reasoning", text=reasoning)
        if decision:
            self.event(type="decision", text=decision)
        if plan.get("milestone"):
            self.event(type="key_moment", description=str(plan["milestone"]), category="milestone")
        if isinstance(plan.get("objectives"), list) and plan["objectives"]:
            try:
                self._post("/objectives", {"objectives": plan["objectives"]})
            except Exception:
                pass

        if actions:
            try:
                self._post("/action", {"actions": actions})
            except Exception as e:
                print(f"[autopilot] action failed: {e}", file=sys.stderr)

        self.turn += 1
        if self.save_every and self.turn % self.save_every == 0:
            try:
                self._post("/save", {"name": "autopilot_latest"})
                self.event(type="key_moment", description=f"Autosaved (turn {self.turn})", category="milestone")
            except Exception:
                pass
        return True

    # --- main loop ---
    def run(self):
        print(f"[autopilot] connected to {self.server}, model={self.model}")
        print("[autopilot] waiting for control=running (use the dashboard Start button)…")
        self.event(type="alert", text="Autopilot online — press START to play.")
        idle_logged = False
        while True:
            st = self.control_state()
            if st == "stopped":
                # idle until started; exit only on explicit Ctrl-C
                if not idle_logged:
                    print("[autopilot] stopped — idling.")
                    idle_logged = True
                time.sleep(2)
                continue
            if st == "paused":
                time.sleep(1.5)
                continue
            idle_logged = False
            if not self.step():
                break
            time.sleep(self.turn_delay)


def run_autopilot(server: str = "http://localhost:8765", model: Optional[str] = None,
                  turn_delay: float = 1.5):
    base_url = os.environ.get("POKEMON_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = (os.environ.get("POKEMON_LLM_API_KEY")
               or os.environ.get("OPENROUTER_API_KEY") or "")
    model = (model or os.environ.get("POKEMON_LLM_MODEL")
             or "anthropic/claude-sonnet-4.5")
    if not api_key:
        print("ERROR: no LLM API key. Set POKEMON_LLM_API_KEY or OPENROUTER_API_KEY.",
              file=sys.stderr)
        sys.exit(1)
    Autopilot(server, model, base_url, api_key, turn_delay=turn_delay).run()
