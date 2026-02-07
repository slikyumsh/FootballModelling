"""Artifact generation (GIF and heatmaps) and run directory management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .domain import Pitch
from .exceptions import AssetNotFoundError

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # noqa: N816

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactPaths:
    """Locations for artifacts produced by a run."""

    run_dir: Path
    gif_path: Path
    heatmap_path: Path


def prepare_run_dir(base_dir: Path, run_id: str) -> ArtifactPaths:
    """Create a run directory and return artifact paths."""
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        run_dir=run_dir,
        gif_path=run_dir / "player_positions.gif",
        heatmap_path=run_dir / "ball_focus_heatmap.png",
    )


def _load_font(
    font_path: Path | None,
    size: int = 15,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TTF font if available, otherwise fall back to a default font."""
    if font_path and font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)

    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def create_gif(
    background_image: Path,
    snapshots: Iterable[Pitch],
    ball_holders: list[int],
    output_path: Path,
    frame_duration_ms: int = 120,
    font_path: Path | None = None,
) -> None:
    """Render a GIF of player positions over time.

    Args:
        background_image: Path to pitch background image.
        snapshots: Pitch states over simulation steps.
        ball_holders: Player IDs holding the ball at each step (aligned with snapshots).
        output_path: Path to save the GIF.
        frame_duration_ms: Frame duration in milliseconds.
        font_path: Optional font path for jersey numbers.

    Raises:
        AssetNotFoundError: If background image does not exist.
    """
    background_image = Path(background_image)
    if not background_image.exists():
        raise AssetNotFoundError(f"Background image not found: {background_image}")

    bg = Image.open(background_image).convert("RGBA")
    font = _load_font(font_path, size=15)

    frames: list[Image.Image] = []
    snapshots_list = list(snapshots)

    for i, pitch in enumerate(snapshots_list):
        frame = bg.copy()
        draw = ImageDraw.Draw(frame)

        ball_holder_id = ball_holders[i] if i < len(ball_holders) else None

        for player in pitch.players:
            x = player.position.x
            y = player.position.y
            team_id = player.player.team_id
            pid = player.player.player_id

            if team_id == 2:
                color = "red"
            elif pid == ball_holder_id:
                color = "black"
            else:
                color = "blue"

            r = 17
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color, outline="black")

            text = str(pid % 100)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw / 2, y - th / 2), text, fill="white", font=font)

        frames.append(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    logger.info("GIF saved to %s", output_path)


def create_ball_focus_heatmap(
    background_image: Path,
    snapshots: Iterable[Pitch],
    ball_holders: list[int],
    output_path: Path,
    radius: int = 20,
) -> None:
    """Create a heatmap showing where the ball carrier appeared over time.

    Raises:
        AssetNotFoundError: If background image does not exist.
        RuntimeError: If OpenCV is not installed.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for heatmaps. Install opencv-python.")

    background_image = Path(background_image)
    if not background_image.exists():
        raise AssetNotFoundError(f"Background image not found: {background_image}")

    bg = Image.open(background_image).convert("RGBA")
    heatmap = np.zeros((bg.height, bg.width), dtype=np.float32)

    snapshots_list = list(snapshots)

    for idx, pitch in enumerate(snapshots_list):
        pid = ball_holders[idx] if idx < len(ball_holders) else None
        if pid is None:
            continue

        carrier = next((p for p in pitch.players if p.player.player_id == pid), None)
        if carrier is None:
            continue

        x = int(carrier.position.x)
        y = int(carrier.position.y)
        cv2.circle(heatmap, (x, y), radius, 1.0, thickness=-1)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=radius, sigmaY=radius)
    maxv = float(heatmap.max())
    if maxv > 0:
        heatmap = np.clip(heatmap / maxv, 0.0, 1.0)

    rgba = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
    nonzero = heatmap > 0
    rgba[nonzero, 0] = 255
    rgba[nonzero, 1] = (255 * (1 - heatmap[nonzero])).astype(np.uint8)
    rgba[nonzero, 2] = 0
    rgba[nonzero, 3] = (255 * heatmap[nonzero]).astype(np.uint8)

    hm_img = Image.fromarray(rgba, "RGBA")
    combined = Image.alpha_composite(bg, hm_img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)
    logger.info("Heatmap saved to %s", output_path)
