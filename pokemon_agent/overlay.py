"""Grid + annotation overlay for agent vision.

Pokemon Gen 1 (Red/Blue/Yellow) renders the overworld on a 160x144 px Game
Boy screen. The walkable world is a grid of 16x16 px blocks (each block is a
2x2 arrangement of 8x8 hardware tiles). That yields a 10x9 block grid:

    columns: 160 / 16 = 10   (labelled A..J left -> right)
    rows:    144 / 16 = 9    (labelled 1..9 top -> bottom)

The player sprite is locked to a fixed on-screen block while the map scrolls
underneath. In Gen 1 that block is column index 4 (E) and row index 4 (5) for
the standard overworld camera, i.e. cell "E5" is *always* the player.

This module draws that grid over a screenshot and labels each cell (A1 top
left, J9 bottom right) so a vision model can reason about movement in
discrete, nameable steps: "the door is at C3, I'm at E5, so I walk up 2 and
left 2."

The overlay is scaled up (default 4x -> 640x576) so the labels are legible
to a vision model; the underlying pixels stay crisp via nearest-neighbour.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# --- Geometry -------------------------------------------------------------

GB_W, GB_H = 160, 144
BLOCK = 16                      # world block size in GB pixels
COLS = GB_W // BLOCK            # 10
ROWS = GB_H // BLOCK            # 9
PLAYER_COL = 4                  # on-screen block the player is locked to (E)
PLAYER_ROW = 4                  # (row 5, 1-indexed)

COL_LABELS = "ABCDEFGHIJ"       # 10 columns

# DMG-flavoured overlay colours (RGBA)
GRID_LINE = (139, 172, 15, 150)        # #8BAC0F semi-transparent
GRID_LINE_MAJOR = (15, 56, 15, 200)    # darker every cell edge
LABEL_BG = (15, 19, 15, 170)
LABEL_FG = (232, 228, 214, 255)
PLAYER_BOX = (217, 72, 47, 235)        # vermilion signal colour
WALK_WASH = (139, 172, 15, 38)         # faint DMG-green over walkable cells
BLOCK_WASH = (217, 72, 47, 70)         # translucent red over blocked cells


def cell_label(col: int, row: int) -> str:
    """Return e.g. 'E5' for 0-indexed (col=4, row=4)."""
    return f"{COL_LABELS[col]}{row + 1}"


def _load_font(size: int):
    """Try a few common monospace/bitmap fonts, fall back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_grid_overlay(
    screen: Image.Image,
    scale: int = 4,
    show_labels: bool = True,
    mark_player: bool = True,
    walkable: Optional[list] = None,
) -> Image.Image:
    """Draw a labelled 10x9 movement grid over a GB screenshot.

    Parameters
    ----------
    screen : PIL.Image
        The raw 160x144 emulator frame.
    scale : int
        Integer upscale factor (nearest-neighbour) so labels are legible.
    show_labels : bool
        Draw the A1..J9 cell labels.
    mark_player : bool
        Highlight the fixed player cell (E5) with a vermilion box.
    walkable : list, optional
        A 9x10 grid of bool (from collision.build_collision_grid). When
        given, blocked cells get a translucent red wash and walkable cells a
        faint green wash, so spatial reasoning is unambiguous.

    Returns
    -------
    PIL.Image
        The annotated, upscaled image.
    """
    if screen.mode != "RGBA":
        screen = screen.convert("RGBA")
    if screen.size != (GB_W, GB_H):
        screen = screen.resize((GB_W, GB_H), Image.NEAREST)

    big = screen.resize((GB_W * scale, GB_H * scale), Image.NEAREST)
    overlay = Image.new("RGBA", big.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cell = BLOCK * scale
    font = _load_font(max(10, cell // 3))

    # Walkability wash (under the grid lines / labels)
    if walkable is not None:
        for r in range(min(ROWS, len(walkable))):
            for c in range(min(COLS, len(walkable[r]))):
                if r == PLAYER_ROW and c == PLAYER_COL:
                    continue
                wash = WALK_WASH if walkable[r][c] else BLOCK_WASH
                x0, y0 = c * cell, r * cell
                draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=wash)

    # Player box (over the wash, under nothing important)
    if mark_player:
        x0 = PLAYER_COL * cell
        y0 = PLAYER_ROW * cell
        draw.rectangle(
            [x0, y0, x0 + cell - 1, y0 + cell - 1],
            outline=PLAYER_BOX,
            width=max(2, scale),
        )

    # Grid lines
    for c in range(COLS + 1):
        x = c * cell
        draw.line([(x, 0), (x, big.height)], fill=GRID_LINE, width=1)
    for r in range(ROWS + 1):
        y = r * cell
        draw.line([(0, y), (big.width, y)], fill=GRID_LINE, width=1)

    # Labels in each cell's top-left corner
    if show_labels:
        for r in range(ROWS):
            for c in range(COLS):
                label = cell_label(c, r)
                lx = c * cell + 2
                ly = r * cell + 1
                # tiny dark plate behind the text for contrast over busy art
                tb = draw.textbbox((lx, ly), label, font=font)
                draw.rectangle(
                    [tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1],
                    fill=LABEL_BG,
                )
                draw.text((lx, ly), label, fill=LABEL_FG, font=font)

    return Image.alpha_composite(big, overlay)


def render_grid_overlay_bytes(screen: Image.Image, **kwargs) -> bytes:
    """Same as render_grid_overlay but returns PNG bytes."""
    img = render_grid_overlay(screen, **kwargs)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def player_cell() -> str:
    """The grid cell the player always occupies (E5)."""
    return cell_label(PLAYER_COL, PLAYER_ROW)
