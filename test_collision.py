"""Regression tests for the collision/grid geometry (no emulator needed).

Run: python -m pytest test_collision.py -q   (or: python test_collision.py)
"""

from pokemon_agent.collision import (
    BLOCK_COLS,
    BLOCK_ROWS,
    PLAYER_COL,
    PLAYER_ROW,
    TILESET_WALKABLE,
    cell_label,
    render_ascii_map,
)


def test_grid_dimensions():
    assert BLOCK_COLS == 10
    assert BLOCK_ROWS == 9
    assert (PLAYER_COL, PLAYER_ROW) == (4, 4)  # cell E5


def test_player_cell_label():
    assert cell_label(PLAYER_COL, PLAYER_ROW) == "E5"
    assert cell_label(0, 0) == "A1"
    assert cell_label(9, 8) == "J9"


def test_all_24_tilesets_have_collision_sets():
    assert len(TILESET_WALKABLE) == 24
    for tid in range(24):
        assert tid in TILESET_WALKABLE
        assert isinstance(TILESET_WALKABLE[tid], frozenset)
        assert len(TILESET_WALKABLE[tid]) > 0


def test_overworld_walkable_set_matches_pokered():
    # Spot-check the overworld (tileset 0) against the pokered data:
    # path tile 0x2C walkable, tree/rock 0x14 NOT walkable.
    ov = TILESET_WALKABLE[0]
    assert 0x2C in ov
    assert 0x14 not in ov
    assert 0x39 in ov  # decorative-but-passable


def test_render_ascii_map_shape():
    # A fully-walkable 9x10 grid renders the player at E5 and a legend.
    walkable = [[True] * BLOCK_COLS for _ in range(BLOCK_ROWS)]
    out = render_ascii_map({"walkable": walkable}, legend=True)
    lines = out.splitlines()
    # header + 9 rows + blank + 2 legend lines
    assert lines[0].split() == list("ABCDEFGHIJ")
    body = lines[1:10]
    assert len(body) == 9
    # player marker present in the grid body exactly once
    grid_text = "\n".join(body)
    assert grid_text.count("@") == 1
    # row 5 (index 4 in body) should contain the @ at column E
    assert body[PLAYER_ROW].split()[1:][PLAYER_COL] == "@"


if __name__ == "__main__":
    test_grid_dimensions()
    test_player_cell_label()
    test_all_24_tilesets_have_collision_sets()
    test_overworld_walkable_set_matches_pokered()
    test_render_ascii_map_shape()
    print("All collision tests passed.")
