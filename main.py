"""
Play two matches between two bots (each gets first move once).

Usage:
    python main.py <bot1_folder> <bot2_folder>

Example:
    python main.py ruleBot myBot

Bot folders are looked up inside sample_submission/.
"""

import sys
import importlib.util
import inspect
from pathlib import Path

import numpy as np
from PIL import Image

from connect4plus.game import env as make_env

SUBMISSIONS_DIR = Path(__file__).resolve().parent / "sample_submission"
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


# ─── Model Loading ───────────────────────────────────────────────────────────


def load_bot(name):
    """Load the first class with an `act` method from sample_submission/<name>/model.py."""
    model_path = SUBMISSIONS_DIR / name / "model.py"
    if not model_path.exists():
        print(f"Error: submission '{name}' not found at {model_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location(name, model_path)
    if spec is None or spec.loader is None:
        print(f"Error: could not load module spec for '{name}'")
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if hasattr(obj, "act") and callable(getattr(obj, "act")):
            return obj()

    print(f"Error: no class with an act() method found in {model_path}")
    sys.exit(1)


# ─── Single Game ─────────────────────────────────────────────────────────────


def save_recording(frames, path):
    """Save a list of RGB numpy arrays as an animated GIF."""
    if not frames:
        return
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=500, loop=0)
    print(f"  Recording saved to {path}")


def play_game(bot_first, bot_second, name_first, name_second):
    """
    Play one game.  bot_first acts as player_0 (first move).
    Returns (winner_name | "draw", move_count, frames).
    """
    game = make_env(render_mode="rgb_array")
    game.reset()
    moves = 0
    frames = []

    # Capture initial board state
    frame = game.render()
    if isinstance(frame, np.ndarray):
        frames.append(frame.copy())

    bots = {"player_0": bot_first, "player_1": bot_second}
    names = {"player_0": name_first, "player_1": name_second}

    for agent in game.agent_iter():
        obs, reward, termination, truncation, info = game.last()

        if termination or truncation:
            break

        action = bots[agent].act(obs)
        game.step(action)
        moves += 1

        frame = game.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame.copy())

    rewards = game.rewards
    game.close()

    if rewards["player_0"] > rewards["player_1"]:
        return names["player_0"], moves, frames
    elif rewards["player_1"] > rewards["player_0"]:
        return names["player_1"], moves, frames
    else:
        return "draw", moves, frames


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <bot1> <bot2>")
        sys.exit(1)

    name1, name2 = sys.argv[1], sys.argv[2]

    print(f"Loading {name1}...")
    bot1 = load_bot(name1)
    print(f"Loading {name2}...")
    bot2 = load_bot(name2)

    RECORDINGS_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*45}")
    print(f"  Match: {name1} vs {name2}  (2 games)")
    print(f"{'='*45}\n")

    # Game 1: bot1 goes first
    print(f"Game 1 — {name1} (first) vs {name2}")
    winner1, moves1, frames1 = play_game(bot1, bot2, name1, name2)
    if winner1 == "draw":
        print(f"  Result: Draw after {moves1} moves")
    else:
        print(f"  Winner: {winner1} in {moves1} moves")
    save_recording(frames1, RECORDINGS_DIR / f"{name1}_vs_{name2}_game1.gif")
    print()

    # Game 2: bot2 goes first
    print(f"Game 2 — {name2} (first) vs {name1}")
    winner2, moves2, frames2 = play_game(bot2, bot1, name2, name1)
    if winner2 == "draw":
        print(f"  Result: Draw after {moves2} moves")
    else:
        print(f"  Winner: {winner2} in {moves2} moves")
    save_recording(frames2, RECORDINGS_DIR / f"{name2}_vs_{name1}_game2.gif")
    print()

    # Summary
    scores = {name1: 0, name2: 0}
    for w in [winner1, winner2]:
        if w in scores:
            scores[w] += 1

    print(f"{'='*45}")
    print(f"  Final: {name1} {scores[name1]} - {scores[name2]} {name2}")
    if scores[name1] == scores[name2]:
        print("  Overall: Tied")
    else:
        overall = name1 if scores[name1] > scores[name2] else name2
        print(f"  Overall winner: {overall}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
