"""Record a frozen Doom perception suite from ViZDoom: frames at 640x480 plus typed questions
whose labels come from the engine (the labels buffer and game variables), never from a
human. The frames stay out of git (suites/*/frames/ is ignored); the manifest pins their
hashes and this script regenerates them deterministically from the recorded seeds.

Run: uv run python tools/record_doom_suite.py [--seeds 3] [--steps 120] [--every 3]
Needs the `games` extra (vizdoom).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import zlib
from pathlib import Path

import numpy as np
import vizdoom as vzd

ROOT = Path(__file__).resolve().parents[1] / "suites"
SCENARIO = "deadly_corridor"
TICS_PER_ACTION = 4
ENEMIES = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "Imp", "Demon", "MarineChainsaw"}
BUTTONS = [
    "move left",
    "move right",
    "attack",
    "move forward",
    "move backward",
    "turn left",
    "turn right",
]


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as PNG with the standard library only (no Pillow)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def make_game(
    seed: int, scenario: str = SCENARIO, skill: int | None = None, timeout_tics: int | None = None
) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / f"{scenario}.cfg"))
    if skill is not None:
        game.set_doom_skill(skill)
    if timeout_tics is not None:
        game.set_episode_timeout(timeout_tics)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_available_buttons(
        [
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.ATTACK,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
        ]
    )
    game.set_available_game_variables(
        [vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2, vzd.GameVariable.KILLCOUNT]
    )
    game.set_seed(seed)
    game.init()
    return game


def enemies_on_screen(state, width: int) -> list[dict]:
    out = []
    for label in state.labels:
        if label.object_name in ENEMIES and label.width > 0:
            centre = (label.x + label.width / 2) / width  # 0 = left edge, 1 = right edge
            out.append(
                {
                    "name": label.object_name,
                    "centre": centre,
                    "width": label.width,
                    "height": label.height,
                }
            )
    return sorted(out, key=lambda e: -e["height"])  # the tallest on screen is the nearest


def expert_action(enemies: list[dict], attack_band: float = 0.12) -> str:
    """jevlike's labels-buffer expert: attack when the nearest enemy is centred, else turn
    towards it, else move forward."""
    if not enemies:
        return "move forward"
    offset = enemies[0]["centre"] - 0.5
    if abs(offset) <= attack_band:
        return "attack"
    return "turn left" if offset < 0 else "turn right"


def side(enemies: list[dict]) -> str:
    if not enemies:
        return "none"
    offset = enemies[0]["centre"] - 0.5
    if abs(offset) <= 0.12:
        return "centre"
    return "left" if offset < 0 else "right"


def record(
    seeds: list[int], steps: int, every: int, out_dir: Path, rng: random.Random
) -> list[dict]:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n = 0
    for seed in seeds:
        game = make_game(seed)
        game.new_episode()
        for step in range(steps):
            state = game.get_state()
            if state is None or game.is_episode_finished():
                break
            width = state.screen_buffer.shape[1]
            enemies = enemies_on_screen(state, width)
            action = expert_action(enemies)
            if step % every == 0:
                name = f"frames/{seed:02d}_{step:04d}.png"
                (out_dir / name).write_bytes(png_bytes(np.ascontiguousarray(state.screen_buffer)))
                health, ammo, kills = (float(v) for v in state.game_variables)
                context = f"A first-person view from a shooter game. Health {health:.0f}, ammo {ammo:.0f}."
                base = {"context": context, "images": [name], "seed": seed, "step": step}
                rows.append(
                    {
                        **base,
                        "case_id": f"visible_{n}",
                        "group": "enemy_visible",
                        "question_type": "noul",
                        "instruction": "An enemy is visible on screen.",
                        "options": ["false", "true"],
                        "label": 1 if enemies else 0,
                    }
                )
                rows.append(
                    {
                        **base,
                        "case_id": f"side_{n}",
                        "group": "enemy_side",
                        "instruction": "Where is the nearest enemy?",
                        "options": ["left", "centre", "right", "none"],
                        "label": ["left", "centre", "right", "none"].index(side(enemies)),
                    }
                )
                rows.append(
                    {
                        **base,
                        "case_id": f"count_{n}",
                        "group": "enemy_count",
                        "question_type": "score",
                        "instruction": "How many enemies are on screen?",
                        "options": ["none", "one", "several"],
                        "label": min(len(enemies), 2),
                    }
                )
                rows.append(
                    {
                        **base,
                        "case_id": f"action_{n}",
                        "group": "expert_action",
                        "instruction": "Which button should the player press now?",
                        "options": BUTTONS,
                        "label": BUTTONS.index(action),
                    }
                )
                n += 1
            # the expert plays, with a little noise so the frames are not all the same corridor
            chosen = action if rng.random() > 0.15 else rng.choice(BUTTONS)
            game.make_action([1 if b == chosen else 0 for b in BUTTONS], TICS_PER_ACTION)
        game.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    seeds = list(range(7, 7 + args.seeds))
    rng = random.Random(20260920)
    tmp_name = args.name or "doom-frames"
    out_dir = ROOT / tmp_name
    rows = record(seeds, args.steps, args.every, out_dir, rng)
    frames = sorted({r["images"][0] for r in rows})
    name = args.name or f"doom-frames-{len(frames)}"
    if name != tmp_name:
        final = ROOT / name
        if final.exists():
            raise SystemExit(f"{final} exists; a suite is frozen, pick another name")
        out_dir.rename(final)
        out_dir = final
    suite = out_dir / f"{name}.jsonl"
    suite.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    groups: dict[str, int] = {}
    for r in rows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    manifest = {
        "name": name,
        "file": suite.name,
        "sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
        "cases": len(rows),
        "groups": groups,
        "question_type": "mixed",
        "images_sha256": {
            f: hashlib.sha256((out_dir / f).read_bytes()).hexdigest() for f in frames
        },
        "source": (
            f"tools/record_doom_suite.py on ViZDoom {vzd.__version__}, scenario {SCENARIO}, seeds {seeds}, "
            f"{args.steps} steps per seed, one frame every {args.every} steps, 640x480 RGB"
        ),
        "construction": (
            "frames recorded while jevlike's labels-buffer expert plays (with 15% random actions); every "
            "label derives from the engine: the labels buffer for enemy presence, side and count, the "
            "expert rule for the button. Frames are not committed; regenerate with the same command and "
            "the manifest checks their hashes"
        ),
        "expected_answers_by": "the game engine (labels buffer) and a fixed expert rule",
        "measures": "perception of a game frame: presence, side and count of enemies; agreement with a scripted policy",
        "known_results": "none at authoring time",
    }
    (out_dir / f"{name}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(name, len(rows), "cases", len(frames), "frames", manifest["sha256"][:16])


if __name__ == "__main__":
    main()
