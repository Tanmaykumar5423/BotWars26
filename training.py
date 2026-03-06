"""
Sample DQN training script for Connect 4 Plus.

This script trains a simple neural network to play Connect 4 Plus using
Double DQN against a mix of opponents (rule-based bot + random).
It's meant as a starting point — feel free to swap in PPO, AlphaZero-style
MCTS, or any other algorithm.

Usage:
    python training.py

After training, your model weights are saved to  weights/model.safetensors.
Logs, plots and gameplay GIFs are written to     logs/

Package your final model class + weights into a submission folder like:

    yourBotName/
        model.py        <- must define a class with act(observation)
        model.safetensors
"""

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file, load_file

from connect4plus.game import env as make_env

# Import rule-based bot as a training opponent
sys.path.insert(0, str(Path(__file__).resolve().parent / "sample_submission"))
from ruleBot.model import RuleBot  # type: ignore  # noqa: E402

# ─── Hyper-parameters ────────────────────────────────────────────────────────

EPISODES = 15_000         # total training games
BATCH_SIZE = 256          # mini-batch size for replay
GAMMA = 0.99              # discount factor
LR = 1e-4                 # learning rate (lower for stability)
EPS_START = 1.0           # initial exploration rate (lowered if warm-starting)
EPS_END = 0.05            # final exploration rate
EPS_DECAY_EPISODES = 5000 # linearly decay epsilon over this many episodes
REPLAY_SIZE = 50_000      # replay buffer capacity
TARGET_UPDATE = 50        # sync target network every N episodes (faster)
TRAIN_STEPS_PER_EP = 4    # gradient steps per episode
SAVE_EVERY = 1000         # save checkpoint every N episodes
EVAL_EVERY = 100          # evaluate & log every N episodes
RECORD_EVERY = 2000       # record gameplay GIFs every N episodes

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
LOGS_DIR = Path(__file__).resolve().parent / "logs"

# Opponent mix (must sum to 1.0) — no self-play for stability
OPP_RULE_PROB = 0.7       # play against the rule-based bot
OPP_RANDOM_PROB = 0.3     # play against a random opponent


# ─── Neural Network ──────────────────────────────────────────────────────────


class DQN(nn.Module):
    """A small CNN that maps a (6,7,3) board observation to Q-values for 7 columns."""

    def __init__(self):
        super().__init__()
        # Input: 3 channels (my pieces, opponent pieces, neutral coin)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 7)  # one Q-value per column

    def forward(self, x):
        # x shape: (batch, 3, 6, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def obs_to_tensor(observation):
    """Convert a PettingZoo observation dict to a (1, 3, 6, 7) float tensor."""
    board = observation["observation"]  # shape (6, 7, 3)
    tensor = torch.from_numpy(board.astype(np.float32))  # (6, 7, 3)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 6, 7)
    return tensor


def select_action(policy_net, observation, epsilon, device: torch.device | str = "cpu"):
    """Epsilon-greedy action selection respecting the action mask."""
    action_mask = observation["action_mask"]
    valid_actions = [i for i, m in enumerate(action_mask) if m == 1]

    if random.random() < epsilon:
        return random.choice(valid_actions)

    with torch.no_grad():
        q_values = policy_net(obs_to_tensor(observation).to(device)).squeeze(0)
    # Mask illegal actions to -inf so they're never picked
    for a in range(7):
        if action_mask[a] == 0:
            q_values[a] = float("-inf")
    return int(q_values.argmax().item())


def random_action(observation):
    """Pick a random legal column."""
    action_mask = observation["action_mask"]
    valid = [i for i, m in enumerate(action_mask) if m == 1]
    return random.choice(valid)


def evaluate(policy_net, device, opponent="random", num_games=30):
    """
    Play num_games as 1st player AND num_games as 2nd player vs the given
    opponent.  Returns (win_rate_as_1st, win_rate_as_2nd).
    """
    rule_bot = RuleBot()

    def _opp_action(obs):
        if opponent == "rule":
            return rule_bot.act(obs)
        return random_action(obs)

    wins_1st = wins_2nd = 0

    for learner_seat in ("player_0", "player_1"):
        for _ in range(num_games):
            ev = make_env()
            ev.reset()
            learner_reward = 0.0
            for agent in ev.agent_iter():
                obs, reward, termination, truncation, info = ev.last()
                done = termination or truncation
                if agent == learner_seat and done:
                    learner_reward = reward
                if done:
                    action = None
                elif agent == learner_seat:
                    action = select_action(policy_net, obs, epsilon=0.0, device=device)
                else:
                    action = _opp_action(obs)
                ev.step(action)
            if learner_reward > 0:
                if learner_seat == "player_0":
                    wins_1st += 1
                else:
                    wins_2nd += 1
            ev.close()

    return wins_1st / num_games, wins_2nd / num_games


# ─── Gameplay Recording ──────────────────────────────────────────────────────


def record_game(policy_net, device, opponent_type, episode, logs_dir):
    """
    Play one game with render_mode='rgb_array' and save as an animated GIF.
    Returns the winner string for logging.
    """
    rule_bot = RuleBot()
    ev = make_env(render_mode="rgb_array")
    ev.reset()
    frames = []

    # Capture initial frame
    frame = ev.render()
    if isinstance(frame, np.ndarray):
        frames.append(frame.copy())

    learner_seat = "player_0"
    learner_reward = 0.0

    for agent in ev.agent_iter():
        obs, reward, termination, truncation, info = ev.last()
        done = termination or truncation
        if agent == learner_seat and done:
            learner_reward = reward
        if done:
            action = None
        elif agent == learner_seat:
            action = select_action(policy_net, obs, epsilon=0.0, device=device)
        else:
            if opponent_type == "rule":
                action = rule_bot.act(obs)
            else:
                action = random_action(obs)
        ev.step(action)
        frame = ev.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame.copy())

    ev.close()

    # Determine result
    if learner_reward > 0:
        result = "win"
    elif learner_reward < 0:
        result = "loss"
    else:
        result = "draw"

    # Save GIF
    if frames:
        gifs_dir = logs_dir / "gameplay"
        gifs_dir.mkdir(exist_ok=True)
        images = [Image.fromarray(f) for f in frames]
        gif_path = gifs_dir / f"ep{episode:05d}_vs_{opponent_type}_{result}.gif"
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       duration=500, loop=0)

    return result


# ─── Plotting ─────────────────────────────────────────────────────────────────


def save_plots(csv_path, plots_dir):
    """
    Read the training CSV and produce matplotlib plots.
    Fails silently if matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib not available, skip plots

    episodes, epsilons = [], []
    rand_1st, rand_2nd, rule_1st, rule_2nd = [], [], [], []
    combined_scores, losses = [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            epsilons.append(float(row["epsilon"]))
            rand_1st.append(float(row["rand_1st"]))
            rand_2nd.append(float(row["rand_2nd"]))
            rule_1st.append(float(row["rule_1st"]))
            rule_2nd.append(float(row["rule_2nd"]))
            combined_scores.append(float(row["combined"]))
            losses.append(float(row["avg_loss"]))

    if not episodes:
        return

    plots_dir.mkdir(exist_ok=True)

    # ── Plot 1: Win rates ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(episodes, rand_1st, label="vs Random (1st)", alpha=0.7)
    ax.plot(episodes, rand_2nd, label="vs Random (2nd)", alpha=0.7)
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate vs Random Opponent")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(episodes, rule_1st, label="vs RuleBot (1st)", color="tab:red", alpha=0.7)
    ax.plot(episodes, rule_2nd, label="vs RuleBot (2nd)", color="tab:orange", alpha=0.7)
    ax.set_ylabel("Win Rate")
    ax.set_xlabel("Episode")
    ax.set_title("Win Rate vs RuleBot")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "win_rates.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Combined score + epsilon ──
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_score = "tab:blue"
    ax1.plot(episodes, combined_scores, color=color_score, label="Combined Score", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Combined Score (30% rand + 70% rule)", color=color_score)
    ax1.tick_params(axis="y", labelcolor=color_score)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_eps = "tab:green"
    ax2.plot(episodes, epsilons, color=color_eps, linestyle="--", alpha=0.6, label="Epsilon")
    ax2.set_ylabel("Epsilon", color=color_eps)
    ax2.tick_params(axis="y", labelcolor=color_eps)
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("Training Progress", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "training_progress.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Loss ──
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(episodes, losses, color="tab:purple", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Loss (Smooth L1)")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss.png", dpi=150)
    plt.close(fig)


# ─── Replay Buffer ───────────────────────────────────────────────────────────


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.cat(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Training Loop ───────────────────────────────────────────────────────────


def optimise(policy_net, target_net, optimizer, replay, device):
    """Run one Double-DQN gradient step.  Returns loss value or None."""
    if len(replay) < BATCH_SIZE:
        return None

    states, actions, rewards_b, next_states, dones_b = replay.sample(BATCH_SIZE)
    states, next_states = states.to(device), next_states.to(device)
    actions = actions.to(device)
    rewards_b = rewards_b.to(device)
    dones_b = dones_b.to(device)

    # Current Q-values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN: use policy net to SELECT actions, target net to EVALUATE them
    with torch.no_grad():
        best_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
    target_q = rewards_b + GAMMA * next_q * (1 - dones_b)

    loss = F.smooth_l1_loss(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    policy_net = DQN().to(device)
    target_net = DQN().to(device)

    # ── Warm start: load existing weights if available ──
    weights_path = WEIGHTS_DIR / "model.safetensors"
    epsilon = EPS_START
    warm_start = False
    if weights_path.exists():
        # Clone tensors so the memory-mapped file is released (Windows lock fix)
        state_dict = {k: v.clone() for k, v in load_file(weights_path).items()}
        policy_net.load_state_dict(state_dict)
        del state_dict
        print(f"  Loaded existing weights from {weights_path}")
        epsilon = 0.3
        warm_start = True
        print(f"  Warm-starting with ε = {epsilon:.2f}")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)

    rule_bot = RuleBot()
    best_eval = 0.0

    WEIGHTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # ── Set up CSV log ──
    csv_path = LOGS_DIR / "training_log.csv"
    csv_fields = [
        "episode", "epsilon", "replay_size", "avg_loss",
        "rand_1st", "rand_2nd", "rule_1st", "rule_2nd",
        "combined", "best_combined", "elapsed_min",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    csv_file.flush()

    start_time = time.time()
    recent_losses: list[float] = []

    for episode in range(1, EPISODES + 1):
        environment = make_env()
        environment.reset()

        # Decide opponent type for this episode
        opp_type = "rule" if random.random() < OPP_RULE_PROB else "random"

        # Alternate which seat the learner occupies
        learner = "player_0" if episode % 2 == 1 else "player_1"

        prev_obs = None
        prev_action = None

        for agent in environment.agent_iter():
            obs, reward, termination, truncation, info = environment.last()
            done = termination or truncation

            # Store the learner's transition (raw game reward, no shaping)
            if agent == learner and prev_obs is not None:
                prev_state = obs_to_tensor(prev_obs)
                curr_state = obs_to_tensor(obs)
                replay.push(prev_state, prev_action, reward, curr_state, done)

            if done:
                action = None
            elif agent == learner:
                action = select_action(policy_net, obs, epsilon, device)
            else:
                if opp_type == "rule":
                    action = rule_bot.act(obs)
                else:
                    action = random_action(obs)

            if agent == learner and not done:
                prev_obs = obs
                prev_action = action

            environment.step(action)

        environment.close()

        # ── Multiple gradient steps per episode ──
        for _ in range(TRAIN_STEPS_PER_EP):
            loss = optimise(policy_net, target_net, optimizer, replay, device)
            if loss is not None:
                recent_losses.append(loss)

        # Linear epsilon decay (respect warm-start value)
        eps_start = 0.3 if warm_start else EPS_START
        epsilon = max(EPS_END, eps_start - (eps_start - EPS_END) * episode / EPS_DECAY_EPISODES)

        # Sync target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # ── Evaluate, log, plot ──
        if episode % EVAL_EVERY == 0:
            r1, r2 = evaluate(policy_net, device, opponent="random")
            b1, b2 = evaluate(policy_net, device, opponent="rule")
            combined = 0.3 * (r1 + r2) / 2 + 0.7 * (b1 + b2) / 2
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            recent_losses.clear()
            elapsed = (time.time() - start_time) / 60

            print(
                f"Episode {episode:>5} | "
                f"ε = {epsilon:.3f} | "
                f"Loss = {avg_loss:.4f} | "
                f"Replay = {len(replay):>6} | "
                f"Rand 1st/2nd {r1:.0%}/{r2:.0%} | "
                f"Rule 1st/2nd {b1:.0%}/{b2:.0%} | "
                f"Combined {combined:.0%} | "
                f"{elapsed:.1f}m"
            )

            # Write CSV row
            csv_writer.writerow({
                "episode": episode,
                "epsilon": f"{epsilon:.4f}",
                "replay_size": len(replay),
                "avg_loss": f"{avg_loss:.6f}",
                "rand_1st": f"{r1:.4f}",
                "rand_2nd": f"{r2:.4f}",
                "rule_1st": f"{b1:.4f}",
                "rule_2nd": f"{b2:.4f}",
                "combined": f"{combined:.4f}",
                "best_combined": f"{best_eval:.4f}",
                "elapsed_min": f"{elapsed:.2f}",
            })
            csv_file.flush()

            # Update plots every eval step
            save_plots(csv_path, LOGS_DIR / "plots")

            # Save best model (70% ruleBot weight, 30% random weight)
            if combined > best_eval:
                best_eval = combined
                save_file(policy_net.state_dict(), WEIGHTS_DIR / "model.safetensors")
                print(f"  ★ New best ({combined:.0%}) → saved")

        # ── Record gameplay GIFs periodically ──
        if episode % RECORD_EVERY == 0:
            for opp in ("rule", "random"):
                result = record_game(policy_net, device, opp, episode, LOGS_DIR)
                print(f"  🎬 Recorded vs {opp}: {result}")

        # Periodic checkpoint (separate file)
        if episode % SAVE_EVERY == 0:
            save_file(policy_net.state_dict(), WEIGHTS_DIR / "checkpoint.safetensors")
            print(f"  → Checkpoint saved")

    csv_file.close()

    # Final save & plots
    save_file(policy_net.state_dict(), WEIGHTS_DIR / "latest.safetensors")
    save_plots(csv_path, LOGS_DIR / "plots")

    elapsed_total = (time.time() - start_time) / 60
    print(
        f"\nTraining complete in {elapsed_total:.1f} minutes."
        f"\n  Best model   → {WEIGHTS_DIR / 'model.safetensors'}"
        f"\n  Latest model → {WEIGHTS_DIR / 'latest.safetensors'}"
        f"\n  Logs & plots → {LOGS_DIR}/"
    )
    print("Now wrap your model in a submission class — see README.md for details.")


if __name__ == "__main__":
    train()
