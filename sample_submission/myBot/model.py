import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file


class DQN(nn.Module):
    """Same architecture used in training.py — must match exactly to load weights."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MyBot:
    def __init__(self):
        self.device = torch.device("cpu")
        self.net = DQN()

        weights_path = Path(__file__).parent / "model.safetensors"
        state_dict = load_file(str(weights_path))
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.net.eval()

    def act(self, observation):
        obs = observation["observation"].astype(np.float32)  # (6, 7, 3)
        action_mask = observation["action_mask"]

        # Convert to (1, 3, 6, 7) tensor
        state = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            q_values = self.net(state).squeeze(0).numpy()

        # Mask illegal actions
        q_values[action_mask == 0] = -np.inf

        return int(np.argmax(q_values))
