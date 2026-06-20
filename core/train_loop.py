"""
train_loop.py — MAPPO Training Loop with TD Error Tracking & Logging Hooks
===========================================================================

Extracted from V2 main_train.py with the following additions:
    - TD error per-step tracking for critic trigger integration
    - Trajectory logging hooks (calls trajectory_logger at each step)
    - Callback system: on_update_end(update, metrics, td_stats) for external modules
    - Configurable via YAML (config/train_config.yaml)

Usage:
    from core.train_loop import train_mappo_v3
    train_mappo_v3(config, callbacks=[critic_trigger.on_update_end])
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import math
import yaml
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from core.mappo_agent import (
    RoleConditionedMAPPOAgentV2,
    embed_goal_batch_v2,
    _to_chunks,
)
from core.crafting_env import BatchCraftingEnvV2, I_GOLD, F_GAME_OVER, NUM_ITEMS, ITEM_NAMES
from llm.orchestrator import (
    LLMOrchestratorV2,
    compute_shaped_reward_batch_v2,
)

try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])
except AttributeError:
    pass


def load_config(config_path: str = "config/train_config.yaml") -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
def train_mappo_v3(
    # Core options (can be overridden from CLI)
    clear_logs: bool = False,
    clear_checkpoints: bool = False,
    n_envs: int = 128,
    num_steps: int = 256,
    num_updates: int = 2000,
    no_shaping: bool = False,
    llm_dynamic: bool = False,
    llm_interval: int = 10,
    llm_model_name: str = "unknown",
    deep: bool = False,
    seed: int = 42,
    # V3 additions
    config_path: str = "config/train_config.yaml",
    callbacks: list = None,
    trajectory_logger=None,
):
    """
    Main MAPPO training loop with V3 extensions.

    Args:
        callbacks: List of callback functions with signature:
                   callback(update, metrics_dict, td_stats_dict) -> optional dict
                   If callback returns a dict with 'adaptive_weights', those will
                   override the current weights.
        trajectory_logger: Optional TrajectoryLogger instance for per-step state capture.
    """
    if callbacks is None:
        callbacks = []

    num_agents  = 2
    goal_dim    = 3
    vec_dim     = goal_dim + NUM_ITEMS + 4  # 3 + 10 + 4 (absolute positions) = 17
    seq_len     = 16   # BPTT chunk length (standard for recurrent MAPPO)
    hidden_size = 256

    assert num_steps % seq_len == 0, \
        f"num_steps ({num_steps}) must be divisible by seq_len ({seq_len})"

    lambda_initial = 0.0 if no_shaping else 1.0
    lambda_final   = 0.0 if no_shaping else 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V3 Train] Device : {device}")

    if clear_checkpoints:
        import shutil
        ckpt_dir = "checkpoints"
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)

    if clear_logs:
        import shutil, glob
        for d in glob.glob("runs/v3_*"):
            shutil.rmtree(d)

    vec_env = BatchCraftingEnvV2(n_envs=n_envs, seed=seed)
    orchestrator = LLMOrchestratorV2()

    agent = RoleConditionedMAPPOAgentV2(
        cnn_channels=9, goal_dim=goal_dim, flag_dim=NUM_ITEMS + 4, deep=deep,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    print(f"[V3 Train] Network: {'deep (3-layer critic)' if deep else 'standard (2-layer critic)'}")
    print(f"[V3 Train] Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print(f"[V3 Train] Seq chunk length (BPTT): {seq_len}")

    global_step_counter  = 0
    best_avg_env_reward  = -float("inf")
    lambda_t             = lambda_initial
    reward_history       = deque(maxlen=20)

    adaptive_weights = {
        'w_wood': 1.0, 'w_stone': 1.0, 'w_workbench': 1.0,
        'w_iron': 1.0, 'w_bridge': 1.0, 'w_enemy': 1.0, 'w_gold': 1.0,
    }
    prev_llm_metrics = {}
    decay_active = False
    decay_steps = 0

    N = n_envs * num_agents
    step_role_ids = torch.tensor(
        [a % num_agents for _ in range(n_envs) for a in range(num_agents)],
        dtype=torch.long, device=device,
    )

    prefix = "v3_Baseline"
    if llm_dynamic:
        safe_model = llm_model_name.replace(":", "-").replace("/", "-")
        prefix = f"v3_LLMDynamic_{'Deep' if deep else 'Std'}_{safe_model}"
    elif no_shaping:
        prefix = "v3_Baseline_Sparse"

    run_name = f"{prefix}_E{n_envs}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs("runs", exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")
    os.makedirs("checkpoints", exist_ok=True)

    T = num_steps
    num_chunks = T // seq_len

    # ── Rollout Buffers ──────────────────────────────────────────────────
    buf_fov      = torch.zeros(T, N, 9, 7, 7, device=device)
    buf_gmap     = torch.zeros(N, T, 9, 61, 61)          # CPU, N-first
    buf_vec      = torch.zeros(T, N, vec_dim, device=device)
    buf_actions  = torch.zeros(T, N, dtype=torch.long, device=device)
    buf_logprobs = torch.zeros(T, N, device=device)
    buf_rewards  = torch.zeros(T, N, device=device)
    buf_values   = torch.zeros(T, N, device=device)
    buf_dones    = torch.zeros(T, N, device=device)
    buf_rnn      = torch.zeros(T, N, hidden_size, device=device)

    # ── V3: TD Error Buffer (per-step, for critic trigger) ───────────────
    buf_td_errors = torch.zeros(T, N, device=device)

    all_obs, _ = vec_env.reset()
    all_fov, all_gmap = vec_env._get_obs_batch_fov()

    inventory = all_obs[:, 0, 4:4+NUM_ITEMS]

    if not no_shaping:
        goal_zone_indices, goal_active = orchestrator.batch_lookup_goals_v2_randomized(
            inventory, subtask_weights=adaptive_weights if llm_dynamic else None
        )
    else:
        goal_zone_indices = np.full((n_envs, 2), -1, dtype=np.int32)
        goal_active = np.zeros((n_envs, 2), dtype=bool)

    rnn_state = torch.zeros(N, hidden_size, device=device)

    # ── PPO Hyperparameters ──────────────────────────────────────────────
    num_minibatches = 8
    ppo_epochs      = 4
    clip_coef       = 0.2
    vf_coef         = 0.5
    ent_coef        = 0.05
    gamma           = 0.99
    gae_lambda      = 0.95
    max_grad_norm   = 0.5

    lr_initial     = 3e-4
    lr_final       = 1e-5
    warmup_updates = 20
    for pg in optimizer.param_groups:
        pg['lr'] = lr_initial

    # ── V3: Trajectory logging setup ─────────────────────────────────────
    # Determine which envs to log (every Kth)
    log_every_k = 8
    logged_env_ids = np.arange(0, n_envs, log_every_k)

    # ═══════════════════════════════════════════════════════════════════
    for update in range(1, num_updates + 1):
        update_start = time.perf_counter()

        epoch_episode_count    = 0
        epoch_success_count    = 0
        epoch_subtask_steps    = np.zeros(NUM_ITEMS, dtype=np.int64)
        epoch_env_reward_sum   = 0.0
        epoch_env_reward_count = 0

        # ══════════════════════════════════════════════════════════
        # ROLLOUT
        # ══════════════════════════════════════════════════════════
        for step in range(num_steps):
            global_step_counter += n_envs

            # Compute goal targets from dynamic zone locations
            env_idx = np.arange(n_envs)[:, np.newaxis]
            safe_zone_indices = np.clip(goal_zone_indices, 0, 6)
            goal_targets = vec_env.zones[env_idx, safe_zone_indices]

            goal_emb = embed_goal_batch_v2(all_obs, goal_targets, goal_active)
            inv_repeat = np.stack([all_obs[:, 0, 4:4+NUM_ITEMS],
                                   all_obs[:, 0, 4:4+NUM_ITEMS]], axis=1)
            pos_a0 = all_obs[:, 0, 0:2] / 60.0
            pos_a1 = all_obs[:, 0, 2:4] / 60.0
            pos_for_a0 = np.concatenate([pos_a0, pos_a1], axis=1)
            pos_for_a1 = np.concatenate([pos_a1, pos_a0], axis=1)
            pos_repeat = np.stack([pos_for_a0, pos_for_a1], axis=1)
            vec_input = np.concatenate([pos_repeat, goal_emb, inv_repeat], axis=2)

            fov_t = torch.from_numpy(
                all_fov.reshape(N, 9, 7, 7)
            ).to(device, non_blocking=True)
            gmap_repeat = np.stack([all_gmap, all_gmap], axis=1)
            gmap_t = torch.from_numpy(
                gmap_repeat.reshape(N, 9, 61, 61)
            ).to(device, non_blocking=True)
            vec_t = torch.from_numpy(
                vec_input.reshape(N, vec_dim)
            ).to(device, non_blocking=True)

            with torch.no_grad():
                action, logprob, _, value, rnn_state_out = agent.get_action_and_value(
                    fov_t, gmap_t, vec_t, step_role_ids, rnn_state
                )

            buf_fov[step]      = fov_t
            buf_gmap[:, step]  = gmap_t.cpu()          # Store on CPU (N-first)
            buf_vec[step]      = vec_t
            buf_actions[step]  = action
            buf_logprobs[step] = logprob
            buf_values[step]   = value.flatten()
            buf_rnn[step]      = rnn_state             # Hidden state BEFORE GRU

            actions_np = action.cpu().numpy().reshape(n_envs, num_agents)
            next_obs, env_rewards, dones, truncs, info = vec_env.step(actions_np)
            terminal = dones | truncs

            # Zero out rnn hidden state at episode boundaries
            terminal_expanded = np.stack([terminal, terminal], axis=1).reshape(-1)
            terminal_mask = torch.from_numpy(
                terminal_expanded.astype(np.float32)
            ).to(device)
            rnn_state = rnn_state_out * (1.0 - terminal_mask).unsqueeze(1)

            sw = adaptive_weights if (llm_dynamic and not no_shaping) else None
            active_for_shaping = goal_active.copy()
            active_for_shaping[terminal] = False

            shaped_r = compute_shaped_reward_batch_v2(
                next_obs, all_obs, goal_zone_indices, active_for_shaping,
                vec_env.zones, gamma=gamma, subtask_weights=sw,
            )

            total_r = env_rewards + lambda_t * shaped_r
            buf_rewards[step] = torch.from_numpy(
                total_r.reshape(N)
            ).to(device, non_blocking=True)
            buf_dones[step] = terminal_mask

            # ── V3: Trajectory Logging ────────────────────────────────
            if trajectory_logger is not None and step % 4 == 0:
                # Log every 4th step to reduce I/O overhead
                trajectory_logger.log_step(
                    update=update,
                    step=step,
                    env_ids=logged_env_ids,
                    snapshot=vec_env.get_state_snapshot(logged_env_ids),
                    actions=actions_np[logged_env_ids],
                    env_rewards=env_rewards[logged_env_ids],
                    shaped_rewards=shaped_r[logged_env_ids],
                    goal_zones=goal_zone_indices[logged_env_ids],
                    goal_active=goal_active[logged_env_ids],
                    terminal=terminal[logged_env_ids],
                )

            if terminal.any():
                epoch_episode_count += int(terminal.sum())
                term_flags = info['terminal_flags']
                gold_mined_mask = term_flags[terminal, I_GOLD] > 0
                epoch_success_count += int(gold_mined_mask.sum())
                for fi in range(NUM_ITEMS):
                    epoch_subtask_steps[fi] += int(
                        (term_flags[terminal, fi] > 0).sum()
                    )

                # ── V3: Log terminal episodes ────────────────────────
                if trajectory_logger is not None:
                    terminal_ids = np.where(terminal)[0]
                    logged_terminal = np.intersect1d(terminal_ids, logged_env_ids)
                    if len(logged_terminal) > 0:
                        trajectory_logger.log_episode_end(
                            env_ids=logged_terminal,
                            terminal_flags=term_flags[logged_terminal],
                            success=(term_flags[logged_terminal, I_GOLD] > 0),
                        )

            epoch_env_reward_sum   += float(env_rewards.sum())
            epoch_env_reward_count += n_envs * num_agents

            new_inv = next_obs[:, 0, 4:4+NUM_ITEMS]
            old_inv = all_obs[:, 0, 4:4+NUM_ITEMS]
            inv_changed = (old_inv != new_inv).any(axis=1) | terminal
            if inv_changed.any() and not no_shaping:
                gt_new, ga_new = orchestrator.batch_lookup_goals_v2_randomized(
                    new_inv, subtask_weights=adaptive_weights if llm_dynamic else None
                )
                goal_zone_indices[inv_changed] = gt_new[inv_changed]
                goal_active[inv_changed]       = ga_new[inv_changed]

            all_obs = next_obs
            all_fov, all_gmap = vec_env._get_obs_batch_fov()

        # ══════════════════════════════════════════════════════════
        # GAE (Generalized Advantage Estimation)
        # ══════════════════════════════════════════════════════════
        advantages = torch.zeros(T, N, device=device)
        gae        = torch.zeros(N, device=device)
        with torch.no_grad():
            next_gmap_repeat = np.stack([all_gmap, all_gmap], axis=1)
            next_gmap_t = torch.from_numpy(
                next_gmap_repeat.reshape(N, 9, 61, 61)
            ).to(device)

            env_idx = np.arange(n_envs)[:, np.newaxis]
            safe_zone_indices = np.clip(goal_zone_indices, 0, 6)
            goal_targets_next = vec_env.zones[env_idx, safe_zone_indices]

            goal_emb_next = embed_goal_batch_v2(
                all_obs, goal_targets_next, goal_active
            )
            inv_repeat_next = np.stack([all_obs[:, 0, 4:4+NUM_ITEMS],
                                        all_obs[:, 0, 4:4+NUM_ITEMS]], axis=1)
            pos_next_a0 = all_obs[:, 0, 0:2] / 60.0
            pos_next_a1 = all_obs[:, 0, 2:4] / 60.0
            pos_next_for_a0 = np.concatenate([pos_next_a0, pos_next_a1], axis=1)
            pos_next_for_a1 = np.concatenate([pos_next_a1, pos_next_a0], axis=1)
            pos_repeat_next = np.stack([pos_next_for_a0, pos_next_for_a1], axis=1)
            vec_input_next = np.concatenate(
                [pos_repeat_next, goal_emb_next, inv_repeat_next], axis=2
            )
            next_vec_t = torch.from_numpy(
                vec_input_next.reshape(N, vec_dim)
            ).to(device)

            next_value = agent.get_value(next_gmap_t, next_vec_t).flatten()

        for t in reversed(range(T)):
            not_done    = 1.0 - buf_dones[t]
            next_masked = next_value * not_done
            delta       = buf_rewards[t] + gamma * next_masked - buf_values[t]
            gae         = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
            next_value  = buf_values[t]

            # ── V3: Store per-step TD error ──────────────────────────
            buf_td_errors[t] = delta

        returns = advantages + buf_values  # [T, N]

        # ══════════════════════════════════════════════════════════
        # V3: Compute TD Error Statistics for this update
        # ══════════════════════════════════════════════════════════
        td_errors_flat = buf_td_errors.detach()
        td_stats = {
            "mean_td_error":     float(td_errors_flat.mean()),
            "std_td_error":      float(td_errors_flat.std()),
            "abs_mean_td_error": float(td_errors_flat.abs().mean()),
            "max_td_error":      float(td_errors_flat.abs().max()),
            "variance_td_error": float(td_errors_flat.var()),
            # Per-agent TD error stats (agent 0 = even indices, agent 1 = odd)
            "td_error_agent0_mean": float(td_errors_flat[:, 0::2].mean()),
            "td_error_agent1_mean": float(td_errors_flat[:, 1::2].mean()),
        }

        # ══════════════════════════════════════════════════════════
        # PPO UPDATE — Sequence-Chunked with BPTT
        # ══════════════════════════════════════════════════════════
        total_samples = num_chunks * N
        mb_chunk_size = total_samples // num_minibatches

        # Reshape GPU buffers: [T, N, ...] -> [total_samples, seq_len, ...]
        c_fov      = _to_chunks(buf_fov,      num_chunks, seq_len, N)
        c_vec      = _to_chunks(buf_vec,       num_chunks, seq_len, N)
        c_actions  = _to_chunks(buf_actions,   num_chunks, seq_len, N)
        c_logprobs = _to_chunks(buf_logprobs,  num_chunks, seq_len, N)
        c_values   = _to_chunks(buf_values,    num_chunks, seq_len, N)
        c_dones    = _to_chunks(buf_dones,     num_chunks, seq_len, N)
        c_adv      = _to_chunks(advantages,    num_chunks, seq_len, N)
        c_returns  = _to_chunks(returns,       num_chunks, seq_len, N)

        # Global map: CPU, N-first -> chunk directly (no copy needed)
        c_gmap = buf_gmap.view(
            N, num_chunks, seq_len, 9, 61, 61
        ).reshape(total_samples, seq_len, 9, 61, 61)

        # Initial GRU hidden state for each chunk
        c_rnn_init = buf_rnn[::seq_len]  # [num_chunks, N, hidden]
        c_rnn_init = (c_rnn_init.permute(1, 0, 2)
                      .contiguous()
                      .view(total_samples, hidden_size))

        # Role IDs per chunk-sample
        c_role_ids = (step_role_ids.unsqueeze(1)
                      .expand(-1, num_chunks)
                      .contiguous()
                      .view(total_samples))

        # Per-role advantage normalization
        c_adv_flat = c_adv.reshape(-1)
        c_roles_exp = (c_role_ids.unsqueeze(1)
                       .expand(-1, seq_len)
                       .reshape(-1))
        for role_idx in range(num_agents):
            mask = (c_roles_exp == role_idx)
            adv_role = c_adv_flat[mask]
            if adv_role.numel() > 1:
                std_role = adv_role.std()
                if std_role > 1e-8:
                    c_adv_flat[mask] = (
                        (adv_role - adv_role.mean()) / (std_role + 1e-8)
                    )
                else:
                    c_adv_flat[mask] = adv_role - adv_role.mean()
        c_adv = c_adv_flat.view(total_samples, seq_len)

        epoch_losses, epoch_v_losses, epoch_pg_losses, epoch_ent = [], [], [], []

        for _ppo_epoch in range(ppo_epochs):
            indices = np.random.permutation(total_samples)
            for start in range(0, total_samples, mb_chunk_size):
                end    = min(start + mb_chunk_size, total_samples)
                mb_idx = indices[start:end]

                mb_fov      = c_fov[mb_idx]
                mb_gmap     = c_gmap[mb_idx].to(device, non_blocking=True)
                mb_vec      = c_vec[mb_idx]
                mb_actions  = c_actions[mb_idx]
                mb_logprobs = c_logprobs[mb_idx]
                mb_adv      = c_adv[mb_idx]
                mb_returns  = c_returns[mb_idx]
                mb_values   = c_values[mb_idx]
                mb_roles    = c_role_ids[mb_idx]
                mb_rnn      = c_rnn_init[mb_idx]
                mb_dones    = c_dones[mb_idx]

                new_logprobs, new_entropy, new_values = agent.evaluate_sequences(
                    mb_fov, mb_gmap, mb_vec, mb_roles,
                    mb_rnn, mb_actions, mb_dones,
                )

                # Flatten sequence dim for loss computation
                new_lp   = new_logprobs.reshape(-1)
                new_ent  = new_entropy.reshape(-1)
                new_val  = new_values.reshape(-1)
                old_lp   = mb_logprobs.reshape(-1)
                old_val  = mb_values.reshape(-1)
                adv_flat = mb_adv.reshape(-1)
                ret_flat = mb_returns.reshape(-1)

                # ── Policy loss ──
                logratio = new_lp - old_lp
                ratio    = logratio.exp()
                pg_loss1 = -adv_flat * ratio
                pg_loss2 = -adv_flat * torch.clamp(
                    ratio, 1.0 - clip_coef, 1.0 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ── Value loss with clipping ──
                v_clipped = old_val + torch.clamp(
                    new_val - old_val, -clip_coef, clip_coef
                )
                v_loss_unclipped = (new_val - ret_flat) ** 2
                v_loss_clipped   = (v_clipped - ret_flat) ** 2
                v_loss = 0.5 * torch.max(
                    v_loss_unclipped, v_loss_clipped
                ).mean()

                # ── Entropy bonus ──
                entropy_loss = new_ent.mean()

                # ── Total loss ──
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_v_losses.append(v_loss.item())
                epoch_pg_losses.append(pg_loss.item())
                epoch_ent.append(entropy_loss.item())

        # ══════════════════════════════════════════════════════════
        # LOGGING & SCHEDULING
        # ══════════════════════════════════════════════════════════
        avg_loss       = float(np.mean(epoch_losses))
        avg_v_loss     = float(np.mean(epoch_v_losses))
        avg_pg_loss    = float(np.mean(epoch_pg_losses))
        avg_entropy    = float(np.mean(epoch_ent))
        avg_env_reward = epoch_env_reward_sum / max(epoch_env_reward_count, 1)
        success_rate   = epoch_success_count / max(epoch_episode_count, 1)
        update_time    = time.perf_counter() - update_start

        # Learning rate: warmup + cosine decay
        if update <= warmup_updates:
            lr_now = lr_initial * (update / warmup_updates)
        else:
            progress = (update - warmup_updates) / max(
                num_updates - warmup_updates, 1
            )
            lr_now = lr_final + 0.5 * (lr_initial - lr_final) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # Lambda decay: Conditional linear annealing
        pct_bridge = epoch_subtask_steps[7] / max(epoch_episode_count, 1)
        if pct_bridge > 0.4 and not no_shaping:
            decay_active = True
            
        if decay_active:
            decay_steps += 1
            progress = min(decay_steps / 500.0, 1.0)
            lambda_t = lambda_initial + (lambda_final - lambda_initial) * progress
        else:
            lambda_t = lambda_initial

        # Reward tracking
        reward_history.append(avg_env_reward)

        # ── Build metrics dict for callbacks & TB ────────────────────
        metrics = {
            "update": update,
            "global_step": global_step_counter,
            "avg_loss": avg_loss,
            "avg_v_loss": avg_v_loss,
            "avg_pg_loss": avg_pg_loss,
            "avg_entropy": avg_entropy,
            "avg_env_reward": avg_env_reward,
            "success_rate": success_rate,
            "lr_now": lr_now,
            "lambda_t": lambda_t,
            "epoch_episode_count": epoch_episode_count,
            "epoch_success_count": epoch_success_count,
            "update_time": update_time,
            "adaptive_weights": dict(adaptive_weights),
        }

        # Per-subtask completion percentages
        subtask_pcts = {}
        for fi, item_name in enumerate(ITEM_NAMES):
            pct = epoch_subtask_steps[fi] / max(epoch_episode_count, 1)
            subtask_pcts[item_name.lower()] = pct
        metrics["subtask_pcts"] = subtask_pcts

        # ── TensorBoard logging ──────────────────────────────────────
        writer.add_scalar("Training/Total_Loss",    avg_loss,       global_step_counter)
        writer.add_scalar("Training/Value_Loss",    avg_v_loss,     global_step_counter)
        writer.add_scalar("Training/Policy_Loss",   avg_pg_loss,    global_step_counter)
        writer.add_scalar("Training/Entropy",       avg_entropy,    global_step_counter)
        writer.add_scalar("Training/Learning_Rate", lr_now,         global_step_counter)
        writer.add_scalar("Training/Lambda",        lambda_t,       global_step_counter)
        writer.add_scalar("Rewards/Avg_Env_Reward", avg_env_reward, global_step_counter)
        writer.add_scalar("Episodes/Success_Rate",  success_rate,   global_step_counter)

        # V3: TD Error TensorBoard logging
        writer.add_scalar("TD_Error/Mean",          td_stats["mean_td_error"],     global_step_counter)
        writer.add_scalar("TD_Error/Abs_Mean",      td_stats["abs_mean_td_error"], global_step_counter)
        writer.add_scalar("TD_Error/Std",           td_stats["std_td_error"],      global_step_counter)
        writer.add_scalar("TD_Error/Variance",      td_stats["variance_td_error"], global_step_counter)
        writer.add_scalar("TD_Error/Agent0_Mean",   td_stats["td_error_agent0_mean"], global_step_counter)
        writer.add_scalar("TD_Error/Agent1_Mean",   td_stats["td_error_agent1_mean"], global_step_counter)

        for fi, item_name in enumerate(ITEM_NAMES):
            pct = epoch_subtask_steps[fi] / max(epoch_episode_count, 1)
            writer.add_scalar(
                f"Subtasks/{item_name}_Pct", pct, global_step_counter
            )

        for k, v in adaptive_weights.items():
            writer.add_scalar(f"LLM_Weights/{k}", v, global_step_counter)

        if update % 10 == 0:
            sps = int(n_envs * num_steps / max(update_time, 1e-6))
            p_pick = epoch_subtask_steps[3] / max(epoch_episode_count, 1)
            p_sw   = epoch_subtask_steps[4] / max(epoch_episode_count, 1)
            p_br   = epoch_subtask_steps[7] / max(epoch_episode_count, 1)
            
            print(
                f"Epoch {update:>4}/{num_updates} | "
                f"Loss: {avg_loss:.4f} "
                f"(v:{avg_v_loss:.3f} pg:{avg_pg_loss:.3f}) | "
                f"R_env: {avg_env_reward:.4f} | "
                f"Craft: [P:{p_pick:.0%} S:{p_sw:.0%} B:{p_br:.0%}] | "
                f"Gold: {success_rate:.0%} | "
                f"\u03bb: {lambda_t:.3f} | "
                f"TD: {td_stats['abs_mean_td_error']:.4f} | "
                f"{update_time:.2f}s ({sps} sps)"
            )

        # ══════════════════════════════════════════════════════════
        # V3: CALLBACK SYSTEM — fire after each update
        # ══════════════════════════════════════════════════════════
        for callback in callbacks:
            result = callback(update, metrics, td_stats)
            if result and isinstance(result, dict):
                # Callbacks can override adaptive weights
                if "adaptive_weights" in result:
                    adaptive_weights = result["adaptive_weights"]
                    w_str = ", ".join(
                        f"{k.replace('w_', '')}={v:.2f}"
                        for k, v in adaptive_weights.items()
                    )
                    print(f"      [Callback] Weights updated: {w_str}")

        # ══════════════════════════════════════════════════════════
        # CHECKPOINTING
        # ══════════════════════════════════════════════════════════
        if update % 50 == 0:
            ckpt_path = f"checkpoints/agent_update_{update}.pt"
            torch.save({
                'update': update,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_env_reward': avg_env_reward,
                'success_rate': success_rate,
                'lambda_t': lambda_t,
                'adaptive_weights': adaptive_weights,
                'td_stats': td_stats,
            }, ckpt_path)

        if avg_env_reward > best_avg_env_reward:
            best_avg_env_reward = avg_env_reward
            torch.save({
                'update': update,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_env_reward': avg_env_reward,
                'success_rate': success_rate,
            }, f"checkpoints/best_agent_{prefix}.pt")

        # ══════════════════════════════════════════════════════════
        # LLM ADAPTIVE WEIGHTS (legacy V2 behavior, still supported)
        # ══════════════════════════════════════════════════════════
        if llm_dynamic and update % llm_interval == 0:
            curr_metrics = {
                "wood": 100 * epoch_subtask_steps[0] / max(epoch_episode_count, 1),
                "stone": 100 * epoch_subtask_steps[1] / max(epoch_episode_count, 1),
                "iron": 100 * epoch_subtask_steps[2] / max(epoch_episode_count, 1),
                "pickaxe": 100 * epoch_subtask_steps[3] / max(epoch_episode_count, 1),
                "sword": 100 * epoch_subtask_steps[4] / max(epoch_episode_count, 1),
                "armor": 100 * epoch_subtask_steps[5] / max(epoch_episode_count, 1),
                "bridge": 100 * epoch_subtask_steps[7] / max(epoch_episode_count, 1),
                "enemy": 100 * epoch_subtask_steps[8] / max(epoch_episode_count, 1),
                "gold": 100 * epoch_subtask_steps[6] / max(epoch_episode_count, 1),
            }
            
            delta_metrics = {}
            for k, v in curr_metrics.items():
                delta_metrics[k] = v - prev_llm_metrics.get(k, 0.0)
                
            print("[LLM] Querying Adaptive Weights...")
            adaptive_weights = orchestrator.query_adaptive_weights(
                curr_metrics, delta_metrics, prev_weights=adaptive_weights
            )
            
            w_str = ", ".join(f"{k.replace('w_','')}={v:.2f}" for k, v in adaptive_weights.items())
            print(f"      Weights: {w_str}")
            
            prev_llm_metrics = curr_metrics

    # Save final model
    torch.save({
        'update': num_updates,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_env_reward': avg_env_reward,
        'success_rate': success_rate,
    }, "checkpoints/agent_final.pt")

    vec_env.close()
    writer.close()
    print(f"[V3 Train] Training complete. Best R_env: {best_avg_env_reward:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MAPPO V3 — Critic-Triggered LLM Training"
    )
    parser.add_argument("--clear-logs", action="store_true")
    parser.add_argument("--clear-checkpoints", action="store_true")
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-updates", type=int, default=2000)
    parser.add_argument("--baseline", action="store_true",
                        help="Sparse baseline: no LLM shaping (lambda=0).")
    parser.add_argument("--llm-dynamic", action="store_true",
                        help="Enable LLM Dynamic Critic with 7 adaptive sub-task weights.")
    parser.add_argument("--llm-interval", type=int, default=50)
    parser.add_argument("--deep", action="store_true",
                        help="Use 3-layer Critic MLP (deeper value function).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-logging", action="store_true",
                        help="Enable trajectory logging to data/trajectories/")
    args = parser.parse_args()

    # Optionally set up trajectory logger
    traj_logger = None
    if args.enable_logging:
        from logging_utils.trajectory_logger import TrajectoryLogger
        traj_logger = TrajectoryLogger(output_dir="data/trajectories")

    train_mappo_v3(
        clear_logs=args.clear_logs,
        clear_checkpoints=args.clear_checkpoints,
        n_envs=args.n_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        no_shaping=args.baseline,
        llm_dynamic=args.llm_dynamic,
        llm_interval=args.llm_interval,
        deep=args.deep,
        seed=args.seed,
        trajectory_logger=traj_logger,
    )
