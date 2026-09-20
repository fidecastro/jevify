"""Let a recipe play ViZDoom's deadly_corridor from screenshots and record a video.

Each step: one 640x480 frame becomes the state (warmed once), one choice question picks a
button, the game advances four tics. The video plays at game speed (8.75 decisions per
second) with the decision, its probability and the wall-clock latency burned in. Frames,
trace and video land in runs/ (git-ignored). Needs the `games` extra and ffmpeg.

Run: uv run python tools/play_doom.py recipes/<name>.yaml --seconds 60 --seed 7
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from record_doom_suite import BUTTONS, TICS_PER_ACTION, enemies_on_screen, make_game, png_bytes

from jevify.compose import build_engine
from jevify.domain.questions import ChoiceQuestion, ImagePart, Option, State, TextPart
from jevify.recipes import load_recipe

INSTRUCTIONS = {
    "deadly_corridor": (
        "You are playing Doom. Reach the green armor at the far end of the corridor without dying; "
        "enemies step out of the side alcoves. Which button should the player press now?"
    ),
    "defend_the_center": (
        "You are playing Doom. You stand in the middle of a circular room; monsters walk in from "
        "the walls. Turn to face a monster, then shoot it. Which button should the player press now?"
    ),
}
OPTIONS = tuple(Option(b) for b in BUTTONS)


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


async def play(recipe_path: Path, seconds: int, seed: int, out_dir: Path, scenario: str) -> dict:
    recipe = load_recipe(recipe_path)
    engine = build_engine(recipe)
    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    game = make_game(seed, scenario)
    game.new_episode()
    steps = int(seconds * 35 / TICS_PER_ACTION)
    trace: list[dict] = []
    started = time.perf_counter()
    for step in range(steps):
        state = game.get_state()
        if state is None or game.is_episode_finished():
            break
        frame = np.ascontiguousarray(state.screen_buffer)
        png = png_bytes(frame)
        (frames_dir / f"{step:05d}.png").write_bytes(png)
        health, ammo, kills = (float(v) for v in state.game_variables)
        context = f"A first-person view from a shooter game. Health {health:.0f}, ammo {ammo:.0f}."
        jev_state = State(
            (
                TextPart(context),
                ImagePart("data:image/png;base64," + base64.b64encode(png).decode()),
            )
        )
        t0 = time.perf_counter()
        evaluation = await engine.ask(
            jev_state,
            [
                ChoiceQuestion(
                    id="button",
                    instructions=INSTRUCTIONS.get(scenario, INSTRUCTIONS["deadly_corridor"]),
                    options=OPTIONS,
                )
            ],
        )
        latency = (time.perf_counter() - t0) * 1000
        [answer] = evaluation.answers
        chosen = answer.selected
        probs = answer.distribution.as_mapping()
        enemies = enemies_on_screen(state, frame.shape[1])
        trace.append(
            {
                "step": step,
                "tic": state.tic,
                "button": chosen,
                "p": probs[chosen],
                "latency_ms": latency,
                "health": health,
                "ammo": ammo,
                "kills": kills,
                "enemies": len(enemies),
                "rung": answer.readout.rung,
            }
        )
        game.make_action([1 if b == chosen else 0 for b in BUTTONS], TICS_PER_ACTION)
    total = game.get_total_reward()
    finished = game.is_episode_finished()
    dead = game.is_player_dead()
    game.close()
    return {
        "recipe": str(recipe_path),
        "scenario": scenario,
        "seed": seed,
        "steps": len(trace),
        "game_seconds": len(trace) * TICS_PER_ACTION / 35,
        "wall_seconds": time.perf_counter() - started,
        "reward": total,
        "finished": finished,
        "dead": dead,
        "kills": trace[-1]["kills"] if trace else 0,
        "final_health": trace[-1]["health"] if trace else None,
        "median_latency_ms": sorted(t["latency_ms"] for t in trace)[len(trace) // 2]
        if trace
        else None,
        "buttons": {b: sum(1 for t in trace if t["button"] == b) for b in BUTTONS},
        "trace": trace,
    }


def render_video(out_dir: Path, summary: dict, stem: str) -> Path:
    fps = 35 / TICS_PER_ACTION
    srt = out_dir / "trace.srt"
    lines = []
    for i, t in enumerate(summary["trace"]):
        lines += [
            str(i + 1),
            f"{srt_time(i / fps)} --> {srt_time((i + 1) / fps)}",
            f"{stem}  step {t['step']}  {t['button']} (p={t['p']:.2f}, {t['latency_ms']:.0f} ms)  "
            f"health {t['health']:.0f}  kills {t['kills']:.0f}",
            "",
        ]
    srt.write_text("\n".join(lines), encoding="utf-8")
    video = out_dir.parent / f"doom-{stem}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps}",
        "-i",
        str(out_dir / "frames" / "%05d.png"),
        "-vf",
        f"subtitles={srt}:force_style='FontSize=16,Outline=1'",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video),
    ]
    if subprocess.run(cmd).returncode != 0:  # no libass: burn nothing, keep the frames
        cmd = [c for c in cmd if not c.startswith("subtitles=") and c != "-vf"]
        subprocess.run(cmd, check=True)
    return video


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe", type=Path)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--scenario", default="deadly_corridor")
    args = ap.parse_args()
    stem = f"{args.recipe.stem}--{args.scenario}"
    out_dir = args.runs_dir / f"doom-{stem}"
    summary = asyncio.run(play(args.recipe, args.seconds, args.seed, out_dir, args.scenario))
    (out_dir / "trace.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    video = render_video(out_dir, summary, stem)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "trace"} | {"video": str(video)}, indent=1
        )
    )


if __name__ == "__main__":
    main()
