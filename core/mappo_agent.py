"""
mappo_agent.py — MAPPO Network Architecture (Extracted from V2 main_train.py)
===============================================================================

Contains:
    - RoleConditionedMAPPOAgentV2: CNN actor with GRU memory + CNN critic (CTDE)
    - embed_goal_batch_v2: Vectorized goal embedding computation
    - _to_chunks: Sequence chunking helper for BPTT
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


def embed_goal_batch_v2(
    obs_all: np.ndarray,
    goal_targets: np.ndarray,
    goal_active: np.ndarray,
) -> np.ndarray:
    """
    Compute goal embedding for each agent in each environment.

    Args:
        obs_all:      [n_envs, 2, 14] observations (4 coords + 10 inventory)
        goal_targets: [n_envs, 2, 2] goal target positions
        goal_active:  [n_envs, 2] bool — whether each agent has an active goal

    Returns:
        goal_emb: [n_envs, 2, 3] — (active_flag, delta_x/60, delta_y/60)
    """
    agent_pos = np.stack([
        obs_all[:, 0, 0:2],
        obs_all[:, 0, 2:4],
    ], axis=1)

    delta = (goal_targets - agent_pos) / 60.0
    delta *= goal_active[:, :, np.newaxis]
    am_target = goal_active.astype(np.float32)

    return np.concatenate([
        am_target[:, :, np.newaxis],
        delta,
    ], axis=2)


class RoleConditionedMAPPOAgentV2(nn.Module):
    """
    Role-conditioned MAPPO agent with:
    - CNN feature extractor for 7x7 FOV observations (actor)
    - CNN feature extractor for 61x61 global map (critic, CTDE)
    - GRU cell for actor memory (handles POMDP)
    - Separate actor heads per role (Lumberjack vs Miner)
    - Optional deep (3-layer) critic MLP
    """

    def __init__(self, cnn_channels: int = 9, goal_dim: int = 3, flag_dim: int = 10,
                 action_dim: int = 5, n_roles: int = 2, deep: bool = False):
        super().__init__()

        self.hidden_size = 256

        # --- ACTOR ---
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        actor_cnn_out = 64 * 7 * 7  # 3136
        actor_input = actor_cnn_out + goal_dim + flag_dim

        hidden = self.hidden_size
        self.actor_mlp = nn.Sequential(
            nn.Linear(actor_input, hidden),
            nn.ReLU()
        )
        self.actor_gru = nn.GRUCell(hidden, hidden)

        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden, action_dim) for _ in range(n_roles)
        ])

        # --- CRITIC (True CTDE, sees full map) ---
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, kernel_size=8, stride=4),  # 61 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 14 -> 6
            nn.ReLU(),
            nn.Flatten()
        )
        critic_cnn_out = 64 * 6 * 6  # 2304
        critic_input = critic_cnn_out + goal_dim + flag_dim

        if deep:
            self.critic_mlp = nn.Sequential(
                nn.Linear(critic_input, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
        else:
            self.critic_mlp = nn.Sequential(
                nn.Linear(critic_input, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
        self.critic = nn.Linear(hidden, 1)

    def get_value(self, global_map: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.critic_cnn(global_map)
        combined = torch.cat([cnn_feat, vec], dim=-1)
        return self.critic(self.critic_mlp(combined))

    def get_action_and_value(self, fov, global_map, vec, role_ids, hx, action=None):
        """Single-step forward for rollout collection (no grad)."""
        # Actor
        a_cnn_feat = self.actor_cnn(fov)
        a_combined = torch.cat([a_cnn_feat, vec], dim=-1)
        a_feat = self.actor_mlp(a_combined)
        hx_out = self.actor_gru(a_feat, hx)

        logits = torch.zeros(
            fov.shape[0], self.actor_heads[0].out_features,
            device=fov.device, dtype=hx_out.dtype,
        )
        for role_idx, head in enumerate(self.actor_heads):
            mask = (role_ids == role_idx)
            if mask.any():
                logits[mask] = head(hx_out[mask])

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        # Critic
        c_cnn_feat = self.critic_cnn(global_map)
        c_combined = torch.cat([c_cnn_feat, vec], dim=-1)
        value = self.critic(self.critic_mlp(c_combined))

        return action, probs.log_prob(action), probs.entropy(), value, hx_out

    def evaluate_sequences(self, fov_seq, gmap_seq, vec_seq, role_ids,
                           hx_init, actions_seq, dones_seq):
        """
        Sequence-chunked evaluation for PPO update with BPTT.

        The CNN and MLP are batched over all B*L frames for efficiency.
        Only the GRU is unrolled sequentially to enable gradient flow through time.

        Args:
            fov_seq:     [B, L, C, H, W]  — actor FOV observations
            gmap_seq:    [B, L, C, H', W'] — critic global map observations
            vec_seq:     [B, L, D]         — vector observations (goal + inventory)
            role_ids:    [B]               — role index per sample
            hx_init:     [B, hidden]       — GRU hidden state at chunk start
            actions_seq: [B, L]            — actions taken during rollout
            dones_seq:   [B, L]            — done flags (1.0 if terminal)
        Returns:
            log_probs:   [B, L]
            entropy:     [B, L]
            values:      [B, L]
        """
        B, L = fov_seq.shape[0], fov_seq.shape[1]
        device = fov_seq.device
        hidden = self.hidden_size

        # --- Actor: batch CNN + MLP, sequential GRU ---
        fov_flat = fov_seq.reshape(B * L, *fov_seq.shape[2:])
        actor_cnn_out = self.actor_cnn(fov_flat)  # [B*L, cnn_out]

        vec_flat = vec_seq.reshape(B * L, -1)
        actor_combined = torch.cat([actor_cnn_out, vec_flat], dim=-1)
        actor_feat = self.actor_mlp(actor_combined).view(B, L, hidden)

        # Sequential GRU unroll with BPTT
        hx = hx_init
        all_logprobs = []
        all_entropy = []

        for t in range(L):
            hx = self.actor_gru(actor_feat[:, t], hx)

            # Role-conditioned action heads
            logits = torch.zeros(B, self.actor_heads[0].out_features, device=device)
            for role_idx, head in enumerate(self.actor_heads):
                mask = (role_ids == role_idx)
                if mask.any():
                    logits[mask] = head(hx[mask])

            dist = Categorical(logits=logits)
            all_logprobs.append(dist.log_prob(actions_seq[:, t]))
            all_entropy.append(dist.entropy())

            # Reset hidden state at episode boundaries for the next step
            if t < L - 1:
                done_mask = dones_seq[:, t].unsqueeze(1)  # [B, 1]
                hx = hx * (1.0 - done_mask)

        log_probs = torch.stack(all_logprobs, dim=1)  # [B, L]
        entropy = torch.stack(all_entropy, dim=1)      # [B, L]

        # --- Critic: fully batched (no recurrence) ---
        gmap_flat = gmap_seq.reshape(B * L, *gmap_seq.shape[2:])
        critic_cnn_out = self.critic_cnn(gmap_flat)
        critic_combined = torch.cat([critic_cnn_out, vec_flat], dim=-1)
        values = self.critic(self.critic_mlp(critic_combined)).view(B, L)

        return log_probs, entropy, values


# ---------------------------------------------------------------------------
# Chunk helper: [T, N, *trailing] -> [N * num_chunks, seq_len, *trailing]
# ---------------------------------------------------------------------------
def _to_chunks(buf, num_chunks, seq_len, N):
    trailing = buf.shape[2:]
    # [T, N, ...] -> [num_chunks, seq_len, N, ...] -> [N, num_chunks, seq_len, ...] -> [N*num_chunks, seq_len, ...]
    dims = [2, 0, 1] + list(range(3, 3 + len(trailing)))
    return (buf.view(num_chunks, seq_len, N, *trailing)
            .permute(*dims)
            .contiguous()
            .view(num_chunks * N, seq_len, *trailing))
