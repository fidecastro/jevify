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
from jevify.domain.questions import (
    ChoiceQuestion,
    ImagePart,
    NoulQuestion,
    Option,
    State,
    TextPart,
)
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
SIDE_QUESTION = ChoiceQuestion(
    id="side",
    instructions="Where is the nearest monster relative to the centre of the view?",
    options=(Option("left"), Option("centre"), Option("right"), Option("none")),
)
AHEAD_QUESTION = NoulQuestion(
    id="ahead", instructions="A monster is directly ahead, in the middle of the view."
)


VISIBLE_QUESTION = NoulQuestion(id="visible", instructions="A monster is visible in this image.")
CLOSE_QUESTION = NoulQuestion(
    id="close", instructions="The nearest monster is close, filling a large part of the view."
)
CLOSE_HEIGHT = 80  # label height in a 480-pixel frame at which the pistol reliably hits


def crops(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Three overlapping vertical thirds of the view: localisation becomes detection."""
    w = frame.shape[1]
    third = w // 3
    return {
        "left": frame[:, : third + third // 2],
        "centre": frame[:, third - third // 4 : 2 * third + third // 4],
        "right": frame[:, 2 * third - third // 2 :],
    }


class Corridor:
    """The model perceives, this rule acts, with a target lock: once a side is chosen the
    player keeps turning that way until a monster is ahead, shoots until nothing is ahead,
    and only then looks for the next target; with nothing in view it walks forward. The
    lock is what keeps two monsters on opposite sides from turning the player in circles."""

    def __init__(self, patience: int = 3) -> None:
        self.lock: str | None = None  # "left" or "right"
        self.turned = 0
        self.patience = patience
        self.opposite = 0
        self.steps_ahead = 0

    def act(self, side: str, ahead: bool, close: bool = True) -> str:
        if ahead or side == "centre":
            self.lock, self.turned, self.opposite = None, 0, 0
            # a monster ahead but far: advance and fire in alternation so the pistol keeps
            # hitting while the distance closes
            self.steps_ahead += 1
            # run and gun: keep firing while closing the distance
            return "attack" if close else "attack+move forward"
        self.steps_ahead = 0
        if self.lock is None:
            if side in ("left", "right"):
                self.lock, self.turned = side, 0
            else:
                return "move forward"
        if side == self.lock or side == "none":
            self.opposite = 0
        elif side in ("left", "right"):
            self.opposite += 1
            if self.opposite >= self.patience:  # the answers insist: switch target
                self.lock, self.turned, self.opposite = side, 0, 0
        self.turned += 1
        if self.turned > 12:  # a full sweep found nothing ahead: give up the lock
            self.lock, self.turned = None, 0
            return "move forward"
        return f"turn {self.lock}+attack"  # spray while turning: the alcoves are close


class _CropAnswer:
    """What the trace needs from a decision that was not one engine answer."""

    def __init__(self, selected: str, probs: dict[str, float]) -> None:
        self.selected = selected
        self._probs = probs

    class _Dist:
        def __init__(self, probs: dict[str, float]) -> None:
            self._probs = probs

        def as_mapping(self) -> dict[str, float]:
            return self._probs

    @property
    def distribution(self) -> _CropAnswer._Dist:
        return self._Dist(self._probs)

    @property
    def readout(self):
        class _R:
            rung = "rule"

        return _R()


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


async def play(
    recipe_path: Path,
    seconds: int,
    seed: int,
    out_dir: Path,
    scenario: str,
    policy: str,
    skill: int | None,
    engage: str = "far",
) -> dict:
    recipe = load_recipe(recipe_path)
    engine = build_engine(recipe)
    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    game = make_game(seed, scenario, skill, timeout_tics=seconds * 35 + 1)
    game.new_episode()
    steps = int(seconds * 35 / TICS_PER_ACTION)
    trace: list[dict] = []
    memory: list[str] = []
    rule = Corridor()
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
        if policy == "perception":
            # the model perceives, a fixed rule acts: face the nearest monster, then shoot
            [answer] = (await engine.ask(jev_state, [SIDE_QUESTION])).answers
            chosen = {
                "left": "turn left",
                "right": "turn right",
                "centre": "attack",
                "none": "turn right",
            }[answer.selected]
        elif policy == "corridor":
            # two questions over one warmed frame, then the corridor rule with a short memory
            side_answer, ahead_answer, close_answer = (
                await engine.ask(jev_state, [SIDE_QUESTION, AHEAD_QUESTION, CLOSE_QUESTION])
            ).answers
            answer = side_answer
            chosen = rule.act(
                side_answer.selected,
                ahead_answer.probability_true >= 0.5,
                engage == "far" or close_answer.probability_true >= 0.5,
            )
            memory.append(side_answer.selected)
        elif policy == "crops":
            # detection on three crops: one warm per crop, one yes/no each
            seen: dict[str, float] = {}
            for name, crop in crops(frame).items():
                crop_state = State(
                    (
                        TextPart(context),
                        ImagePart(
                            "data:image/png;base64," + base64.b64encode(png_bytes(crop)).decode()
                        ),
                    )
                )
                [a] = (await engine.ask(crop_state, [VISIBLE_QUESTION])).answers
                seen[name] = a.probability_true
                answer = a
            best = max(seen, key=seen.get)
            side = best if seen[best] >= 0.5 else "none"
            chosen = rule.act(side, best == "centre" and seen["centre"] >= 0.5)
            memory.append(side)
            answer = _CropAnswer(side, {**seen, "none": 1.0 - seen[best]})
        elif policy == "expert":
            # the labels buffer instead of a model: the rule's ceiling with perfect perception
            enemies = enemies_on_screen(state, frame.shape[1])
            offset = (enemies[0]["centre"] - 0.5) if enemies else None
            side = (
                "none"
                if offset is None
                else "centre"
                if abs(offset) <= 0.12
                else ("left" if offset < 0 else "right")
            )
            close = engage == "far" or (bool(enemies) and enemies[0]["height"] >= CLOSE_HEIGHT)
            chosen = rule.act(side, side == "centre", close)
            if chosen == "move forward":
                # nothing to fight: steer towards the armour at the end of the corridor
                armour = [lb for lb in state.labels if lb.object_name == "GreenArmor"]
                if armour:
                    offset = (armour[0].x + armour[0].width / 2) / frame.shape[1] - 0.5
                    if abs(offset) > 0.08:
                        chosen = "turn left" if offset < 0 else "turn right"
                else:
                    chosen = "turn right"  # lost the corridor: sweep until the armour is in view
            memory.append(side)
            answer = _CropAnswer(side, {side: 1.0})
        else:
            question = ChoiceQuestion(
                id="button",
                instructions=INSTRUCTIONS.get(scenario, INSTRUCTIONS["deadly_corridor"]),
                options=OPTIONS,
            )
            [answer] = (await engine.ask(jev_state, [question])).answers
            chosen = answer.selected
        latency = (time.perf_counter() - t0) * 1000
        probs = answer.distribution.as_mapping()
        enemies = enemies_on_screen(state, frame.shape[1])
        trace.append(
            {
                "step": step,
                "tic": state.tic,
                "button": chosen,
                "answer": answer.selected,
                "p": probs[answer.selected],
                "latency_ms": latency,
                "health": health,
                "ammo": ammo,
                "kills": kills,
                "enemies": len(enemies),
                "rung": answer.readout.rung,
            }
        )
        pressed = set(chosen.split("+"))
        game.make_action([1 if b in pressed else 0 for b in BUTTONS], TICS_PER_ACTION)
    total = game.get_total_reward()
    finished = game.is_episode_finished()
    dead = game.is_player_dead()
    game.close()
    return {
        "recipe": str(recipe_path),
        "scenario": scenario,
        "policy": policy,
        "skill": skill,
        "engage": engage,
        "seed": seed,
        "steps": len(trace),
        "game_seconds": len(trace) * TICS_PER_ACTION / 35,
        "wall_seconds": time.perf_counter() - started,
        "reward": total,
        "finished": finished,
        "dead": dead,
        "armour_reached": bool(finished and not dead and len(trace) < steps),
        "timed_out": bool(len(trace) >= steps and not dead),
        "kills": trace[-1]["kills"] if trace else 0,
        "final_health": trace[-1]["health"] if trace else None,
        "median_latency_ms": sorted(t["latency_ms"] for t in trace)[len(trace) // 2]
        if trace
        else None,
        "buttons": {b: sum(1 for t in trace if b in t["button"].split("+")) for b in BUTTONS},
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
            f"{stem}  step {t['step']}  {t['answer']} -> {t['button']} (p={t['p']:.2f}, {t['latency_ms']:.0f} ms)  "
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
    ap.add_argument(
        "--policy",
        choices=["button", "perception", "corridor", "crops", "expert"],
        default="button",
        help="button: the model picks the button; perception: it says where the monster is, a rule acts",
    )
    ap.add_argument(
        "--skill", type=int, default=None, help="Doom skill 1-5 (the scenario's default otherwise)"
    )
    ap.add_argument(
        "--engage",
        choices=["far", "close"],
        default="far",
        help="shoot anything ahead, or only when close",
    )
    args = ap.parse_args()
    stem = f"{args.recipe.stem}--{args.scenario}--{args.policy}" + (
        f"--skill{args.skill}" if args.skill else ""
    )
    out_dir = args.runs_dir / f"doom-{stem}"
    summary = asyncio.run(
        play(
            args.recipe,
            args.seconds,
            args.seed,
            out_dir,
            args.scenario,
            args.policy,
            args.skill,
            args.engage,
        )
    )
    (out_dir / "trace.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    video = render_video(out_dir, summary, stem)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "trace"} | {"video": str(video)}, indent=1
        )
    )


if __name__ == "__main__":
    main()
