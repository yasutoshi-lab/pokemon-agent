"""Game sessions — the unit that binds a playthrough together.

A *game session* is a named run that bundles:
  - a Hermes Agent session id (the brain's memory/continuity across turns)
  - a folder of emulator save-states (the game's progress)
  - objectives, a milestone timeline, and run stats
  - metadata (game, created/updated timestamps)

This is what lets you "start a new game", "load a previous game and its saved
states", and have the autopilot resume the *same Hermes brain* it played with
before. Everything lives on disk under:

    <data_dir>/games/<session_id>/
        manifest.json          # GameSession.to_dict()
        saves/<name>.state      # emulator save-states for THIS run

The legacy flat <data_dir>/saves/ directory still works for ad-hoc saves, but
new play goes through a GameSession so saves and brain-memory stay scoped to a
single run.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_OBJECTIVES = [
    {"tier": "primary", "text": "Become Pokémon League Champion — earn all 8 badges", "done": False},
    {"tier": "secondary", "text": "Deliver Oak's Parcel · get the Pokédex", "done": False},
    {"tier": "tertiary", "text": "Build a balanced team", "done": False},
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GameSession:
    id: str
    name: str
    game: str = "red"
    hermes_session_id: Optional[str] = None
    objectives: List[Dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_OBJECTIVES))
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "turns": 0, "actions": 0, "blackouts": 0, "saves": 0,
    })
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GameSession":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class GameSessionManager:
    """Disk-backed CRUD for game sessions under <data_dir>/games/."""

    def __init__(self, data_dir: str):
        self.root = Path(data_dir).expanduser().resolve() / "games"
        self.root.mkdir(parents=True, exist_ok=True)

    # --- paths ---
    def _dir(self, sid: str) -> Path:
        return self.root / sid

    def _manifest(self, sid: str) -> Path:
        return self._dir(sid) / "manifest.json"

    def saves_dir(self, sid: str) -> Path:
        d = self._dir(sid) / "saves"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def frames_dir(self, sid: str) -> Path:
        d = self._dir(sid) / "frames"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- persistence ---
    def save(self, gs: GameSession) -> GameSession:
        gs.updated_at = _now_iso()
        self._dir(gs.id).mkdir(parents=True, exist_ok=True)
        tmp = self._manifest(gs.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(gs.to_dict(), indent=2))
        tmp.replace(self._manifest(gs.id))
        return gs

    def load(self, sid: str) -> Optional[GameSession]:
        mf = self._manifest(sid)
        if not mf.exists():
            return None
        try:
            return GameSession.from_dict(json.loads(mf.read_text()))
        except Exception:
            return None

    def exists(self, sid: str) -> bool:
        return self._manifest(sid).exists()

    def create(self, name: Optional[str] = None, game: str = "red") -> GameSession:
        sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        gs = GameSession(id=sid, name=name or f"Run {sid}", game=game)
        self.saves_dir(sid)  # make the saves folder
        return self.save(gs)

    def delete(self, sid: str) -> bool:
        d = self._dir(sid)
        if d.exists():
            shutil.rmtree(d)
            return True
        return False

    def list(self) -> List[Dict[str, Any]]:
        """Summaries of all sessions, newest first."""
        out: List[Dict[str, Any]] = []
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            gs = self.load(d.name)
            if not gs:
                continue
            saves = sorted(self.saves_dir(gs.id).glob("*.state"))
            out.append({
                "id": gs.id, "name": gs.name, "game": gs.game,
                "hermes_session_id": gs.hermes_session_id,
                "badges": _latest_badges(gs),
                "save_count": len(saves),
                "latest_save": saves[-1].stem if saves else None,
                "turns": gs.stats.get("turns", 0),
                "milestones": len(gs.milestones),
                "created_at": gs.created_at, "updated_at": gs.updated_at,
            })
        out.sort(key=lambda x: x["updated_at"], reverse=True)
        return out

    # --- per-session save-state listing ---
    def list_saves(self, sid: str) -> List[Dict[str, Any]]:
        d = self.saves_dir(sid)
        out = []
        for f in sorted(d.glob("*.state")):
            st = f.stat()
            out.append({"name": f.stem, "size_bytes": st.st_size, "modified": st.st_mtime})
        out.sort(key=lambda x: x["modified"], reverse=True)
        return out

    def latest_save_path(self, sid: str) -> Optional[Path]:
        saves = sorted(self.saves_dir(sid).glob("*.state"), key=lambda f: f.stat().st_mtime)
        return saves[-1] if saves else None

    # --- milestone helper ---
    def add_milestone(self, gs: GameSession, description: str, category: str = "milestone"):
        gs.milestones.insert(0, {
            "description": description, "category": category,
            "turn": gs.stats.get("turns", 0), "at": _now_iso(),
        })
        gs.milestones = gs.milestones[:100]
        self.save(gs)


def _latest_badges(gs: GameSession) -> int:
    for m in gs.milestones:
        if m.get("category") == "badge":
            return 1  # rough; real count comes from live state
    return 0
