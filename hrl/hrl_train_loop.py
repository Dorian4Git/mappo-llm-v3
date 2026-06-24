"""
hrl_train_loop.py — MAPPO Training Loop with Options Framework (Phase 4)
========================================================================

Extends the core training loop to use the HRL Options Framework.
Instead of directly shaping rewards with LLM weights, the LLM assigns
high-level Options, and the OptionController provides intrinsic rewards
to the low-level MAPPO agents.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import json
import math
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from core.mappo_agent import (
    RoleConditionedMAPPOAgentV2,
    embed_goal_batch_v2,
    _to_chunks,
)
from core.crafting_env import BatchCraftingEnvV2, I_GOLD, NUM_ITEMS, ITEM_NAMES
from llm.prompt_builder import PromptBuilder
from llm.async_bridge import LLMBridge
from hrl.option_controller import OptionController, NUM_OPTIONS

def get_option_target_zone(opt_strs):
    mapping = {
        "COLLECT_WOOD": 0, "COLLECT_STONE": 1,
        "CRAFT_PICKAXE": 2, "MINE_IRON": 3,
        "CRAFT_SWORD": 2, "CRAFT_ARMOR": 2, 
        "BUILD_BRIDGE": 4, "FIGHT_ENEMY": 5, "COLLECT_GOLD": 6, "IDLE": -1
    }
    if isinstance(opt_strs, str):
        return mapping.get(opt_strs, -1)
    return np.array([mapping.get(s, -1) for s in opt_strs])

def check_option_success(opt_strs, inv_prev, inv_next):
    # I_WOOD=0, I_STONE=1, I_IRON=2, I_PICKAXE=3, I_SWORD=4, I_ARMOR=5, I_GOLD=6, F_BRIDGE=7, F_ENEMY_DEFEATED=8
    n_envs = inv_prev.shape[0]
    success = np.zeros(n_envs, dtype=bool)
    if isinstance(opt_strs, str):
        opt_strs = [opt_strs] * n_envs
        
    for i in range(n_envs):
        opt = opt_strs[i]
        if opt == "COLLECT_WOOD": success[i] = inv_next[i, 0] > inv_prev[i, 0]
        elif opt == "COLLECT_STONE": success[i] = inv_next[i, 1] > inv_prev[i, 1]
        elif opt == "CRAFT_PICKAXE": success[i] = inv_next[i, 3] > inv_prev[i, 3]
        elif opt == "MINE_IRON": success[i] = inv_next[i, 2] > inv_prev[i, 2]
        elif opt == "CRAFT_SWORD": success[i] = inv_next[i, 4] > inv_prev[i, 4]
        elif opt == "CRAFT_ARMOR": success[i] = inv_next[i, 5] > inv_prev[i, 5]
        elif opt == "BUILD_BRIDGE": success[i] = inv_next[i, 7] > inv_prev[i, 7]
        elif opt == "FIGHT_ENEMY": success[i] = inv_next[i, 8] > inv_prev[i, 8]
        elif opt == "COLLECT_GOLD": success[i] = inv_next[i, 6] > inv_prev[i, 6]
        elif opt == "IDLE": success[i] = False
    return success

def train_mappo_hrl(
    n_envs: int = 128,
    num_steps: int = 256,
    num_updates: int = 2000,
    deep: bool = False,
    seed: int = 42,
    llm_backend: str = "ollama",
    llm_model: str = "qwen2.5:7b",
    disable_lora: bool = False,
):
    num_agents = 2
    # 2 (pos) + 3 (goal padding) + 10 (inv) + NUM_OPTIONS (one-hot option)
    vec_dim = 2 + 3 + NUM_ITEMS + NUM_OPTIONS
    seq_len = 16
    hidden_size = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HRL Train] Device : {device}")

    vec_env = BatchCraftingEnvV2(n_envs=n_envs, seed=seed)
    
    # Init LLM & Option Controller
    bridge = LLMBridge(backend=llm_backend, model_name=llm_model)
    if llm_backend.startswith("huggingface"):
        bridge.swap_model(llm_model, backend=llm_backend)
        if disable_lora:
            bridge.disable_lora()
        
    prompt_builder = PromptBuilder()
    option_controller = OptionController(n_envs=n_envs)

    agent = RoleConditionedMAPPOAgentV2(
        cnn_channels=9, goal_dim=3, flag_dim=2 + NUM_ITEMS + NUM_OPTIONS, deep=deep,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    print(f"[HRL Train] Using HRL Options Framework")
    print(f"[HRL Train] Network: {'deep (3-layer)' if deep else 'standard (2-layer)'}")

    global_step_counter = 0
    best_avg_env_reward = -float("inf")

    N = n_envs * num_agents
    step_role_ids = torch.tensor(
        [a % num_agents for _ in range(n_envs) for a in range(num_agents)],
        dtype=torch.long, device=device,
    )

    if llm_backend == "gemini":
        model_str = "Gemini"
    elif llm_backend == "ollama":
        model_str = "Ollama"
    else:
        model_str = "NoLoRA" if disable_lora else "LoRA"
        
    run_name = f"v3_HRL_{'Deep' if deep else 'Std'}_{model_str}_E{n_envs}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs("runs", exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")
    os.makedirs("checkpoints", exist_ok=True)
    
    llm_log_path = os.path.join(f"runs/{run_name}", "llm_queries.jsonl")

    T = num_steps
    num_chunks = T // seq_len

    buf_fov = torch.zeros(T, N, 9, 7, 7, device=device)
    buf_gmap = torch.zeros(N, T, 9, 61, 61)
    buf_vec = torch.zeros(T, N, vec_dim, device=device)
    buf_actions = torch.zeros(T, N, dtype=torch.long, device=device)
    buf_logprobs = torch.zeros(T, N, device=device)
    buf_rewards = torch.zeros(T, N, device=device)
    buf_values = torch.zeros(T, N, device=device)
    buf_dones = torch.zeros(T, N, device=device)
    buf_rnn = torch.zeros(T, N, hidden_size, device=device)
    buf_td_errors = torch.zeros(T, N, device=device)

    all_obs, _ = vec_env.reset()
    all_fov, all_gmap = vec_env._get_obs_batch_fov()
    rnn_state = torch.zeros(N, hidden_size, device=device)

    # PPO Hyperparameters
    num_minibatches, ppo_epochs, clip_coef = 8, 4, 0.2
    vf_coef, ent_coef, gamma, gae_lambda = 0.5, 0.05, 0.99, 0.95
    max_grad_norm = 0.5
    lr_initial, lr_final, warmup_updates = 3e-4, 1e-5, 20

    for update in range(1, num_updates + 1):
        update_start = time.perf_counter()
        epoch_episode_count, epoch_success_count = 0, 0
        epoch_subtask_steps = np.zeros(NUM_ITEMS, dtype=np.int64)
        epoch_env_reward_sum, epoch_env_reward_count = 0.0, 0

        for step in range(num_steps):
            global_step_counter += n_envs

            goal_emb = np.zeros((n_envs, 2, 3), dtype=np.float32) # Padding to match vec_dim
            inv_repeat = np.stack([all_obs[:, 0, 4:4+NUM_ITEMS], all_obs[:, 0, 4:4+NUM_ITEMS]], axis=1)
            opt_repeat = option_controller.get_option_embeddings()
            pos_repeat = vec_env.pos.copy()
            
            vec_input = np.concatenate([pos_repeat, goal_emb, inv_repeat, opt_repeat], axis=2)

            fov_t = torch.from_numpy(all_fov.reshape(N, 9, 7, 7)).to(device, non_blocking=True)
            gmap_repeat = np.stack([all_gmap, all_gmap], axis=1)
            gmap_t = torch.from_numpy(gmap_repeat.reshape(N, 9, 61, 61)).to(device, non_blocking=True)
            vec_t = torch.from_numpy(vec_input.reshape(N, vec_dim)).to(device, non_blocking=True)

            with torch.no_grad():
                action, logprob, _, value, rnn_state_out = agent.get_action_and_value(
                    fov_t, gmap_t, vec_t, step_role_ids, rnn_state
                )

            buf_fov[step] = fov_t
            buf_gmap[:, step] = gmap_t.cpu()
            buf_vec[step] = vec_t
            buf_actions[step] = action
            buf_logprobs[step] = logprob
            buf_values[step] = value.flatten()
            buf_rnn[step] = rnn_state

            actions_np = action.cpu().numpy().reshape(n_envs, num_agents)
            pos_prev = vec_env.pos.copy()
            next_obs, env_rewards, dones, truncs, info = vec_env.step(actions_np)
            terminal = dones | truncs

            terminal_expanded = np.stack([terminal, terminal], axis=1).reshape(-1)
            terminal_mask = torch.from_numpy(terminal_expanded.astype(np.float32)).to(device)
            rnn_state = rnn_state_out * (1.0 - terminal_mask).unsqueeze(1)
            
            # --- HRL Reward Function & LLM Trigger Logic ---
            inv_prev = all_obs[:, 0, 4:4+NUM_ITEMS]
            inv_next = next_obs[:, 0, 4:4+NUM_ITEMS]
            
            a0_opt = option_controller.get_active_option(0)
            a1_opt = option_controller.get_active_option(1)
            
            a0_success = check_option_success(a0_opt, inv_prev, inv_next)
            a1_success = check_option_success(a1_opt, inv_prev, inv_next)
            
            intrinsic_r = np.zeros((n_envs, 2), dtype=np.float32)
            step_penalty = -0.01
            MAX_DIST = 85.0
            
            # Agent 0
            z0 = get_option_target_zone(a0_opt)
            dist0_prev = np.linalg.norm(pos_prev[:, 0] - vec_env.zones[np.arange(n_envs), z0], axis=1) 
            dist0_next = np.linalg.norm(vec_env.pos[:, 0] - vec_env.zones[np.arange(n_envs), z0], axis=1) 
            phi0_prev = MAX_DIST - dist0_prev 
            phi0_next = MAX_DIST - dist0_next 
            shaping0 = (0.99 * phi0_next) - phi0_prev
            intrinsic_r[:, 0] = np.where(a0_success, 1.0 + step_penalty, step_penalty + (shaping0 * 0.05))
            intrinsic_r[:, 0] = np.where(z0 == -1, step_penalty, intrinsic_r[:, 0])
            
            # Agent 1
            z1 = get_option_target_zone(a1_opt)
            dist1_prev = np.linalg.norm(pos_prev[:, 1] - vec_env.zones[np.arange(n_envs), z1], axis=1)
            dist1_next = np.linalg.norm(vec_env.pos[:, 1] - vec_env.zones[np.arange(n_envs), z1], axis=1)
            phi1_prev = MAX_DIST - dist1_prev
            phi1_next = MAX_DIST - dist1_next
            shaping1 = (0.99 * phi1_next) - phi1_prev
            intrinsic_r[:, 1] = np.where(a1_success, 1.0 + step_penalty, step_penalty + (shaping1 * 0.05))
            intrinsic_r[:, 1] = np.where(z1 == -1, step_penalty, intrinsic_r[:, 1])
            
            # --- HRL Vectorized LLM Trigger Logic ---
            
            # Decrement cooldowns
            option_controller.cooldown_counter = np.maximum(0, option_controller.cooldown_counter - 1)
            
            # 1. Identify environments that succeeded
            success_mask = a0_success | a1_success
            
            # 2. Filter out envs that are already pending or on cooldown
            ready_mask = success_mask & (~option_controller.llm_pending) & (option_controller.cooldown_counter == 0)
            trigger_envs = np.where(ready_mask)[0]
            
            if len(trigger_envs) > 0:
                print(f"Batching LLM queries for {len(trigger_envs)} environments...")
                
                # Lock these environments
                option_controller.set_pending(trigger_envs, True)
                option_controller.cooldown_counter[trigger_envs] = 50 
                
                inv_batch = inv_next[trigger_envs].astype(int)
                unique_invs, inverse_indices = np.unique(inv_batch, axis=0, return_inverse=True)
                
                batched_prompts = []
                batched_inventories = []
                unique_to_envs = [trigger_envs[inverse_indices == i] for i in range(len(unique_invs))]
                
                for state_idx, unique_state in enumerate(unique_invs):
                    inv_dict = {
                        "wood": int(unique_state[0]), "stone": int(unique_state[1]), "iron": int(unique_state[2]),
                        "pickaxe": int(unique_state[3]), "sword": int(unique_state[4]), "armor": int(unique_state[5]),
                        "gold": int(unique_state[6]), "bridge": int(unique_state[7]), "enemy": int(unique_state[8]),
                    }
                    batched_inventories.append(inv_dict)
                    
                    rep_env = unique_to_envs[state_idx][0]
                    a0_stat = "Finished" if a0_success[rep_env] else f"Working on {a0_opt[rep_env]}"
                    a1_stat = "Finished" if a1_success[rep_env] else f"Working on {a1_opt[rep_env]}"
                    
                    prompt = prompt_builder.build_hrl_prompt(inv_dict, a0_stat, a1_stat)
                    batched_prompts.append(prompt)
                
                # 3. Define the asynchronous callback for the batch
                def make_batch_cb(unique_env_groups, prompts, inventories, current_step):
                    def _cb(batch_responses):
                        for state_idx, response in enumerate(batch_responses):
                            env_group = unique_env_groups[state_idx]
                            # Apply the response to all environments sharing this state
                            option_controller.update_options_from_llm(response, env_group)
                            option_controller.set_pending(env_group, False)
                            
                            try:
                                with open(llm_log_path, "a") as f:
                                    for env_idx in env_group:
                                        log_entry = {
                                            "global_step": current_step,
                                            "env_id": int(env_idx),
                                            "prompt": prompts[state_idx],
                                            "raw_response": response,
                                            "parsed_a0_option": option_controller.get_active_option(0, env_idx),
                                            "parsed_a1_option": option_controller.get_active_option(1, env_idx),
                                            "inventory": inventories[state_idx]
                                        }
                                        f.write(json.dumps(log_entry) + "\n")
                            except Exception as e:
                                print(f"[Logging Error] Failed to write LLM log: {e}")
                    return _cb
                    
                # 4. Dispatch to the async bridge
                bridge.query_batch_async(batched_prompts, callback=make_batch_cb(unique_to_envs, batched_prompts, batched_inventories, global_step_counter))

            total_r = env_rewards + intrinsic_r
            buf_rewards[step] = torch.from_numpy(total_r.reshape(N)).to(device, non_blocking=True)
            buf_dones[step] = terminal_mask

            if terminal.any():
                epoch_episode_count += int(terminal.sum())
                term_flags = info['terminal_flags']
                gold_mined_mask = term_flags[terminal, I_GOLD] > 0
                epoch_success_count += int(gold_mined_mask.sum())
                for fi in range(NUM_ITEMS):
                    epoch_subtask_steps[fi] += int((term_flags[terminal, fi] > 0).sum())

            epoch_env_reward_sum += float(env_rewards.sum())
            epoch_env_reward_count += n_envs * num_agents

            all_obs = next_obs
            all_fov, all_gmap = vec_env._get_obs_batch_fov()

        # --- GAE ---
        advantages = torch.zeros(T, N, device=device)
        gae = torch.zeros(N, device=device)
        with torch.no_grad():
            next_gmap_repeat = np.stack([all_gmap, all_gmap], axis=1)
            next_gmap_t = torch.from_numpy(next_gmap_repeat.reshape(N, 9, 61, 61)).to(device)
            
            goal_emb_next = np.zeros((n_envs, 2, 3), dtype=np.float32)
            inv_repeat_next = np.stack([all_obs[:, 0, 4:4+NUM_ITEMS], all_obs[:, 0, 4:4+NUM_ITEMS]], axis=1)
            opt_repeat_next = option_controller.get_option_embeddings()
            pos_repeat_next = vec_env.pos.copy()
            vec_input_next = np.concatenate([pos_repeat_next, goal_emb_next, inv_repeat_next, opt_repeat_next], axis=2)
            next_vec_t = torch.from_numpy(vec_input_next.reshape(N, vec_dim)).to(device)

            next_value = agent.get_value(next_gmap_t, next_vec_t).flatten()

        for t in reversed(range(T)):
            not_done = 1.0 - buf_dones[t]
            next_masked = next_value * not_done
            delta = buf_rewards[t] + gamma * next_masked - buf_values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
            next_value = buf_values[t]
            buf_td_errors[t] = delta

        returns = advantages + buf_values

        td_errors_flat = buf_td_errors.detach()
        td_stats = {
            "mean_td_error": float(td_errors_flat.mean()),
            "std_td_error": float(td_errors_flat.std()),
            "abs_mean_td_error": float(td_errors_flat.abs().mean()),
            "variance_td_error": float(td_errors_flat.var()),
            "td_error_agent0_mean": float(td_errors_flat[:, 0::2].mean()),
            "td_error_agent1_mean": float(td_errors_flat[:, 1::2].mean()),
        }

        # --- PPO UPDATE ---
        total_samples = num_chunks * N
        mb_chunk_size = total_samples // num_minibatches

        c_fov = _to_chunks(buf_fov, num_chunks, seq_len, N)
        c_vec = _to_chunks(buf_vec, num_chunks, seq_len, N)
        c_actions = _to_chunks(buf_actions, num_chunks, seq_len, N)
        c_logprobs = _to_chunks(buf_logprobs, num_chunks, seq_len, N)
        c_values = _to_chunks(buf_values, num_chunks, seq_len, N)
        c_dones = _to_chunks(buf_dones, num_chunks, seq_len, N)
        c_adv = _to_chunks(advantages, num_chunks, seq_len, N)
        c_returns = _to_chunks(returns, num_chunks, seq_len, N)

        c_gmap = buf_gmap.view(N, num_chunks, seq_len, 9, 61, 61).reshape(total_samples, seq_len, 9, 61, 61)
        c_rnn_init = buf_rnn[::seq_len].permute(1, 0, 2).contiguous().view(total_samples, hidden_size)
        c_role_ids = step_role_ids.unsqueeze(1).expand(-1, num_chunks).contiguous().view(total_samples)

        c_adv_flat = c_adv.reshape(-1)
        c_roles_exp = c_role_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)
        for role_idx in range(num_agents):
            mask = (c_roles_exp == role_idx)
            adv_role = c_adv_flat[mask]
            if adv_role.numel() > 1:
                std_role = adv_role.std()
                if std_role > 1e-8:
                    c_adv_flat[mask] = (adv_role - adv_role.mean()) / (std_role + 1e-8)
                else:
                    c_adv_flat[mask] = adv_role - adv_role.mean()
        c_adv = c_adv_flat.view(total_samples, seq_len)

        epoch_losses, epoch_v_losses, epoch_pg_losses, epoch_ent = [], [], [], []

        for _ppo_epoch in range(ppo_epochs):
            target_kl_reached = False
            indices = np.random.permutation(total_samples)
            for start in range(0, total_samples, mb_chunk_size):
                end = min(start + mb_chunk_size, total_samples)
                mb_idx = indices[start:end]

                new_logprobs, new_entropy, new_values = agent.evaluate_sequences(
                    c_fov[mb_idx], c_gmap[mb_idx].to(device, non_blocking=True), c_vec[mb_idx],
                    c_role_ids[mb_idx], c_rnn_init[mb_idx], c_actions[mb_idx], c_dones[mb_idx],
                )

                new_lp, new_ent, new_val = new_logprobs.reshape(-1), new_entropy.reshape(-1), new_values.reshape(-1)
                old_lp, old_val = c_logprobs[mb_idx].reshape(-1), c_values[mb_idx].reshape(-1)
                adv_flat, ret_flat = c_adv[mb_idx].reshape(-1), c_returns[mb_idx].reshape(-1)

                logratio = new_lp - old_lp
                ratio = logratio.exp()
                
                # --- NEW TARGET KL GUARD ---
                with torch.no_grad():
                    # Calculate approximate KL Divergence
                    approx_kl = (-logratio).mean().item()
                
                # If the policy has shifted too far, abort the remaining epochs for this batch
                target_kl = 0.015
                if approx_kl > target_kl:
                    target_kl_reached = True
                    break
                # ---------------------------

                pg_loss1 = -adv_flat * ratio
                pg_loss2 = -adv_flat * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_clipped = old_val + torch.clamp(new_val - old_val, -clip_coef, clip_coef)
                v_loss = 0.5 * torch.max((new_val - ret_flat) ** 2, (v_clipped - ret_flat) ** 2).mean()
                entropy_loss = new_ent.mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_v_losses.append(v_loss.item())
                epoch_pg_losses.append(pg_loss.item())
                epoch_ent.append(entropy_loss.item())

            if target_kl_reached:
                break

        # Logging & Scheduling
        avg_loss = float(np.mean(epoch_losses))
        avg_env_reward = epoch_env_reward_sum / max(epoch_env_reward_count, 1)
        success_rate = epoch_success_count / max(epoch_episode_count, 1)
        update_time = time.perf_counter() - update_start

        if update <= warmup_updates:
            lr_now = lr_initial * (update / warmup_updates)
        else:
            progress = (update - warmup_updates) / max(num_updates - warmup_updates, 1)
            lr_now = lr_final + 0.5 * (lr_initial - lr_final) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        writer.add_scalar("Rewards/Avg_Env_Reward", avg_env_reward, global_step_counter)
        writer.add_scalar("Episodes/Success_Rate", success_rate, global_step_counter)
        writer.add_scalar("TD_Error/Abs_Mean", td_stats["abs_mean_td_error"], global_step_counter)

        for fi in range(NUM_ITEMS):
            writer.add_scalar(f"Subtasks/{ITEM_NAMES[fi]}_Pct", epoch_subtask_steps[fi] / max(epoch_episode_count, 1), global_step_counter)

        if update % 10 == 0:
            print(f"Epoch {update:>4}/{num_updates} | Loss: {avg_loss:.4f} | "
                  f"R_env: {avg_env_reward:.4f} | Gold: {success_rate:.0%} | "
                  f"TD: {td_stats['abs_mean_td_error']:.4f} | {update_time:.2f}s")

        if avg_env_reward > best_avg_env_reward:
            best_avg_env_reward = avg_env_reward
            ckpt_name = "checkpoints/best_agent_nolora.pt" if disable_lora else "checkpoints/best_agent.pt"
            torch.save({'update': update, 'model_state_dict': agent.state_dict()}, ckpt_name)

    vec_env.close()
    writer.close()
    bridge.close()
    print(f"[HRL Train] Training complete. Best R_env: {best_avg_env_reward:.4f}")

if __name__ == "__main__":
    train_mappo_hrl()
