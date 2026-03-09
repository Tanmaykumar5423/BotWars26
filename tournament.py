"""
BotWars26 — 100-Game League Tournament Runner
=============================================
Runs a full round-robin tournament between specified bots.
Each pair plays 100 games (50 going first, 50 going second).
Win = 3 pts | Draw = 1 pt | Loss = 0 pts
"""

import sys
import time
import itertools
import importlib.util
import inspect
from pathlib import Path
from collections import defaultdict

from connect4plus.game import env as make_env

# The names of the folders containing your bots
BOT_FOLDERS = ["bot1", "rulebot", "myBotcopilot", "botgemin"]
GAMES_PER_PAIR = 100  # Must be an even number so they get equal turns going first

def load_bot(name):
    """Dynamically load the bot class from its folder."""
    # Check both root and sample_submission directories
    base_paths = [Path(__file__).resolve().parent, Path(__file__).resolve().parent / "sample_submission"]
    
    for base in base_paths:
        model_path = base / name / "model.py"
        if model_path.exists():
            spec = importlib.util.spec_from_file_location(name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "act") and callable(getattr(obj, "act")):
                    return obj()
                    
    print(f"❌ Error: Could not find model.py for '{name}' in root or sample_submission folders.")
    sys.exit(1)

def play_match(bot_0, bot_1):
    """
    Plays a single game where bot_0 goes first. 
    Returns: 0 if bot_0 wins, 1 if bot_1 wins, -1 for a draw.
    """
    # render_mode=None makes the simulation run at maximum CPU speed
    game = make_env(render_mode=None)
    game.reset()
    
    bots = {"player_0": bot_0, "player_1": bot_1}
    
    for agent in game.agent_iter():
        obs, reward, termination, truncation, info = game.last()

        if termination or truncation:
            break

        try:
            action = bots[agent].act(obs)
            game.step(action)
        except Exception as e:
            # If a bot crashes or returns an invalid move, it instantly loses
            print(f"\n[!] Crash detected for {agent}: {e}")
            game.close()
            return 1 if agent == "player_0" else 0

    rewards = game.rewards
    game.close()

    if rewards["player_0"] > rewards["player_1"]:
        return 0
    elif rewards["player_1"] > rewards["player_0"]:
        return 1
    else:
        return -1

def main():
    print(f"🚀 Initializing BotWars26 Local Tournament...")
    print(f"Loading {len(BOT_FOLDERS)} bots: {', '.join(BOT_FOLDERS)}")
    
    loaded_bots = {}
    for name in BOT_FOLDERS:
        loaded_bots[name] = load_bot(name)
        
    print("All bots loaded successfully! Starting matches...\n")

    # Tracking metrics
    points = defaultdict(int)
    stats = defaultdict(lambda: {"W": 0, "L": 0, "D": 0})
    
    # Generate all unique pairs
    pairs = list(itertools.combinations(BOT_FOLDERS, 2))
    total_matches = len(pairs) * GAMES_PER_PAIR
    matches_played = 0
    start_time = time.time()

    for bot_a, bot_b in pairs:
        print(f"⚔️  {bot_a} vs {bot_b} ({GAMES_PER_PAIR} games) ", end="", flush=True)
        
        # Split who goes first 50/50
        for game_num in range(GAMES_PER_PAIR):
            if game_num % 2 == 0:
                p0_name, p1_name = bot_a, bot_b
                p0_bot, p1_bot = loaded_bots[bot_a], loaded_bots[bot_b]
            else:
                p0_name, p1_name = bot_b, bot_a
                p0_bot, p1_bot = loaded_bots[bot_b], loaded_bots[bot_a]

            winner_idx = play_match(p0_bot, p1_bot)
            
            if winner_idx == 0:
                points[p0_name] += 3
                stats[p0_name]["W"] += 1
                stats[p1_name]["L"] += 1
            elif winner_idx == 1:
                points[p1_name] += 3
                stats[p1_name]["W"] += 1
                stats[p0_name]["L"] += 1
            else:
                points[p0_name] += 1
                points[p1_name] += 1
                stats[p0_name]["D"] += 1
                stats[p1_name]["D"] += 1
            
            matches_played += 1
            if matches_played % 20 == 0:
                print(".", end="", flush=True) # Progress bar
        print(" Done!")

    # ── Final Leaderboard ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print("🏆 FINAL TOURNAMENT STANDINGS 🏆")
    print("="*50)
    print(f"{'Rank':<5} | {'Bot Name':<15} | {'Pts':<5} | {'W':<4} | {'L':<4} | {'D':<4}")
    print("-" * 50)
    
    # Sort by points (highest first)
    ranked_bots = sorted(points.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, pts) in enumerate(ranked_bots, 1):
        w, l, d = stats[name]["W"], stats[name]["L"], stats[name]["D"]
        print(f"#{rank:<4} | {name:<15} | {pts:<5} | {w:<4} | {l:<4} | {d:<4}")
        
    print("="*50)
    print(f"Total time: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()