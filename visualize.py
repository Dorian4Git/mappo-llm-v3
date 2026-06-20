import pygame
import torch
import numpy as np
import time
import sys
import os
import argparse

from core.crafting_env import BatchCraftingEnvV2, NUM_ITEMS, ZONES, GRID_LIMIT, DIST_THRESHOLD
from core.mappo_agent import RoleConditionedMAPPOAgentV2
from hrl.option_controller import OptionController, NUM_OPTIONS
from llm.prompt_builder import PromptBuilder
from llm.async_bridge import LLMBridge
from hrl.hrl_train_loop import check_option_success

# Colors
COLOR_BG = (100, 150, 80)
COLOR_GRID = (50, 50, 50)
COLOR_A0 = (0, 255, 255)       # Cyan
COLOR_A1 = (255, 0, 255)       # Magenta
COLOR_OBSTACLE = (100, 100, 100)
COLOR_TEXT = (255, 255, 255)

# Zone Colors
ZONE_COLORS = {
    "wood": (139, 69, 19),      # Saddle Brown
    "stone": (119, 136, 153),   # Light Slate Gray
    "workbench": (147, 112, 219), # Medium Purple
    "iron": (169, 169, 169),    # Dark Gray
    "bridge": (139, 105, 20),   # Dark Goldenrod
    "enemy": (200, 50, 50),     # Red
    "gold": (255, 215, 0),      # Gold
}

# Drawing settings
SCALE_X = 12
SCALE_Y = 6  # Squash Y-axis for oblique perspective
WALL_HEIGHT = 20
GRID_PIXELS_X = int(GRID_LIMIT * SCALE_X)
GRID_PIXELS_Y = int(GRID_LIMIT * SCALE_Y)
PANEL_WIDTH = 350
WINDOW_WIDTH = GRID_PIXELS_X + PANEL_WIDTH
WINDOW_HEIGHT = GRID_PIXELS_Y + 80

def draw_env(screen, env, font, current_options=None, visual_effects=None):
    screen.fill(COLOR_BG)

    from core.crafting_env import RIVER_X_MIN, RIVER_X_MAX, BRIDGE_Y_MIN, BRIDGE_Y_MAX
    from core.crafting_env import I_WOOD, I_STONE, I_IRON, I_PICKAXE, I_SWORD, I_ARMOR, I_GOLD, F_BRIDGE, F_ENEMY_DEFEATED, F_GAME_OVER
    
    inv = env.inventory[0]
    pos = env.pos[0]
    
    # Paths
    path_color = (130, 100, 60)
    pygame.draw.rect(screen, path_color, pygame.Rect(20 * SCALE_X, 39 * SCALE_Y, 15 * SCALE_X, 4 * SCALE_Y))
    pygame.draw.rect(screen, path_color, pygame.Rect(36 * SCALE_X, 39 * SCALE_Y, 10 * SCALE_X, 4 * SCALE_Y))
    pygame.draw.rect(screen, path_color, pygame.Rect(45 * SCALE_X, 20 * SCALE_Y, 4 * SCALE_X, 20 * SCALE_Y))
    pygame.draw.rect(screen, path_color, pygame.Rect(45 * SCALE_X, 20 * SCALE_Y, 5 * SCALE_X, 4 * SCALE_Y))

    # River
    river_rect = pygame.Rect(RIVER_X_MIN * SCALE_X, 0, (RIVER_X_MAX - RIVER_X_MIN) * SCALE_X, GRID_PIXELS_Y)
    pygame.draw.rect(screen, (30, 80, 150), river_rect)

    if inv[F_BRIDGE] > 0:
        bridge_rect = pygame.Rect(RIVER_X_MIN * SCALE_X, BRIDGE_Y_MIN * SCALE_Y, 
                                  (RIVER_X_MAX - RIVER_X_MIN) * SCALE_X, (BRIDGE_Y_MAX - BRIDGE_Y_MIN) * SCALE_Y)
        pygame.draw.rect(screen, (139, 105, 20), bridge_rect)
        for y_line in range(int(BRIDGE_Y_MIN * SCALE_Y), int(BRIDGE_Y_MAX * SCALE_Y), 4):
            pygame.draw.line(screen, (100, 70, 10), (RIVER_X_MIN * SCALE_X, y_line), (RIVER_X_MAX * SCALE_X, y_line), 1)
    else:
        bx, by = ZONES["bridge"]
        pygame.draw.circle(screen, (255, 255, 0, 100), (int((bx - 2) * SCALE_X), int(by * SCALE_Y)), 15, 2)
        pygame.draw.circle(screen, (255, 255, 0, 100), (int((bx + 2) * SCALE_X), int(by * SCALE_Y)), 15, 2)
        stand_text = font.render("Stand Here", True, (255, 255, 0))
        screen.blit(stand_text, (int((bx - 5) * SCALE_X), int((by - 3) * SCALE_Y)))

    # Highlight workbench when both agents are near
    dist_a0 = np.linalg.norm(pos[0] - ZONES["workbench"])
    dist_a1 = np.linalg.norm(pos[1] - ZONES["workbench"])
    if dist_a0 < DIST_THRESHOLD and dist_a1 < DIST_THRESHOLD:
        wx, wy = ZONES["workbench"]
        wx_px, wy_px = int(wx * SCALE_X), int(wy * SCALE_Y)
        a0_px = (int(pos[0][0] * SCALE_X), int(pos[0][1] * SCALE_Y))
        a1_px = (int(pos[1][0] * SCALE_X), int(pos[1][1] * SCALE_Y))
        pygame.draw.line(screen, (255, 255, 100), a0_px, (wx_px, wy_px), 2)
        pygame.draw.line(screen, (255, 255, 100), a1_px, (wx_px, wy_px), 2)
        pygame.draw.circle(screen, (255, 255, 100), (wx_px, wy_px), 20, 2)

    draw_queue = []
    from core.crafting_env import OBSTACLES
    for (x0, y0, x1, y1) in OBSTACLES:
        draw_queue.append({'type': 'wall', 'y_sort': y1, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})

    zone_names = ["wood", "stone", "workbench", "iron", "bridge", "enemy", "gold"]
    zone_flag_map = {
        "wood": I_WOOD, "stone": I_STONE, "workbench": I_ARMOR,
        "iron": I_IRON, "enemy": F_ENEMY_DEFEATED, "gold": I_GOLD
    }
    
    for name in zone_names:
        if name == "bridge": continue
        zx, zy = ZONES[name]
        flag_idx = zone_flag_map.get(name, 0)
        draw_queue.append({'type': 'zone', 'y_sort': zy + 1.5, 'name': name, 'x': zx, 'y': zy, 'flag_idx': flag_idx})

    for a_idx in range(2):
        ax, ay = pos[a_idx]
        draw_queue.append({'type': 'agent', 'y_sort': ay, 'id': a_idx, 'x': ax, 'y': ay})

    draw_queue.sort(key=lambda item: item['y_sort'])

    for item in draw_queue:
        if item['type'] == 'wall':
            rx0, ry0 = item['x0'] * SCALE_X, item['y0'] * SCALE_Y
            rw, rh = (item['x1'] - item['x0']) * SCALE_X, (item['y1'] - item['y0']) * SCALE_Y
            front_rect = pygame.Rect(rx0, ry0 + rh - WALL_HEIGHT, rw, WALL_HEIGHT)
            pygame.draw.rect(screen, (80, 80, 80), front_rect)
            pygame.draw.rect(screen, (0, 0, 0), front_rect, 1)
            top_rect = pygame.Rect(rx0, ry0 - WALL_HEIGHT, rw, rh)
            pygame.draw.rect(screen, COLOR_OBSTACLE, top_rect)
            pygame.draw.rect(screen, (0, 0, 0), top_rect, 1)
            
        elif item['type'] == 'zone':
            zx, zy, name, flag_idx = item['x'], item['y'], item['name'], item['flag_idx']
            color = ZONE_COLORS[name]
            
            if name == "enemy":
                ex_px, ey_px = int(zx * SCALE_X), int(zy * SCALE_Y)
                if inv[flag_idx] == 0:
                    pygame.draw.ellipse(screen, (20, 20, 20), (ex_px - 15, ey_px - 5, 30, 10))
                    pygame.draw.circle(screen, (220, 30, 30), (ex_px, ey_px - 15), 18)
                    pygame.draw.circle(screen, (100, 10, 10), (ex_px, ey_px - 15), 18, 2)
                    label = font.render("Enemy", True, COLOR_TEXT)
                    screen.blit(label, (ex_px - 20, ey_px - 45))
                else:
                    pygame.draw.circle(screen, (100, 30, 30), (ex_px, ey_px - 10), 12)
                    done_label = font.render("Enemy (Dead)", True, (150, 150, 150))
                    screen.blit(done_label, (ex_px - 30, ey_px + 5))
                continue
                
            if inv[flag_idx] > 0:
                color = (max(0, color[0]-100), max(0, color[1]-100), max(0, color[2]-100))
            rx, ry = (zx - 1.5) * SCALE_X, (zy - 1.5) * SCALE_Y
            rw, rh = 3 * SCALE_X, 3 * SCALE_Y
            zone_h = 4
            front_rect = pygame.Rect(rx, ry + rh - zone_h, rw, zone_h)
            pygame.draw.rect(screen, (max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50)), front_rect)
            pygame.draw.rect(screen, (0, 0, 0), front_rect, 1)
            top_rect = pygame.Rect(rx, ry - zone_h, rw, rh)
            pygame.draw.rect(screen, color, top_rect)
            pygame.draw.rect(screen, (0, 0, 0), top_rect, 1)
            label = font.render(name.capitalize(), True, COLOR_TEXT)
            screen.blit(label, (rx, ry - zone_h - 15))
            if inv[flag_idx] > 0:
                done_label = font.render("(Done)", True, (150, 150, 150))
                screen.blit(done_label, (rx, ry - zone_h + rh + 2))
                
        elif item['type'] == 'agent':
            ax, ay, a_idx = item['x'], item['y'], item['id']
            sx, sy = int(ax * SCALE_X), int(ay * SCALE_Y)
            color = COLOR_A0 if a_idx == 0 else COLOR_A1
            pygame.draw.ellipse(screen, (20, 20, 20), (sx - 8, sy - 4, 16, 8))
            body_y = sy - 10
            pygame.draw.circle(screen, color, (sx, body_y), 8)
            
            if inv[I_ARMOR] > 0:
                pygame.draw.circle(screen, (200, 200, 200), (sx, body_y), 10, 3) 
            else:
                pygame.draw.circle(screen, (0, 0, 0), (sx, body_y), 8, 2)
                
            a_label = font.render(f"A{a_idx}", True, (0, 0, 0))
            screen.blit(a_label, (sx - 6, body_y - 6))

    if visual_effects:
        font_fx = pygame.font.SysFont(None, 24, bold=True)
        for fx in visual_effects:
            fx_text = font_fx.render(fx['text'], True, (255, 255, 50))
            screen.blit(fx_text, (fx['x'], fx['y']))

    panel_rect = pygame.Rect(GRID_PIXELS_X, 0, PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(screen, (40, 40, 45), panel_rect)
    pygame.draw.line(screen, (100, 100, 100), (GRID_PIXELS_X, 0), (GRID_PIXELS_X, WINDOW_HEIGHT), 2)

    header_font = pygame.font.SysFont(None, 22, bold=True)
    y_offset = 20
    screen.blit(header_font.render("MAPPO V3 Dashboard", True, (200, 200, 255)), (GRID_PIXELS_X + 20, y_offset))
    y_offset += 40
    screen.blit(font.render(f"Step Count: {env.step_counts[0]}", True, COLOR_TEXT), (GRID_PIXELS_X + 20, y_offset))
    y_offset += 30

    screen.blit(header_font.render("LLM Active Options:", True, (255, 200, 100)), (GRID_PIXELS_X + 20, y_offset))
    y_offset += 25
    
    if current_options:
        a0_g = current_options[0]
        a1_g = current_options[1]
        screen.blit(font.render(f"A0: {a0_g}", True, COLOR_A0), (GRID_PIXELS_X + 20, y_offset))
        y_offset += 20
        screen.blit(font.render(f"A1: {a1_g}", True, COLOR_A1), (GRID_PIXELS_X + 20, y_offset))
    else:
        screen.blit(font.render("No active options", True, COLOR_TEXT), (GRID_PIXELS_X + 20, y_offset))
        
    y_offset += 40
    screen.blit(header_font.render("Team Inventory:", True, (255, 200, 100)), (GRID_PIXELS_X + 20, y_offset))
    y_offset += 25
    
    flag_labels = ["Wood", "Stone", "Iron", "Pickaxe", "Sword", "Armor", "Gold", "Bridge", "EnemyDef", "GameOver"]
    for i, name in enumerate(flag_labels):
        val = int(inv[i])
        status = f"[{val}]" if val > 0 else "[ 0 ]"
        c = (100, 255, 100) if val > 0 else (200, 200, 200)
        if name == "GameOver" and val > 0:
            c = (255, 50, 50)
        screen.blit(font.render(f"{status} {name}", True, c), (GRID_PIXELS_X + 20, y_offset))
        y_offset += 20

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_agent.pt")
    parser.add_argument("--llm-backend", type=str, default="huggingface_peft")
    parser.add_argument("--llm-model", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "models", "qlora_adapter")))
    parser.add_argument("--disable-lora", action="store_true", help="Run the base Qwen model without the LoRA adapter")
    args = parser.parse_args()

    print("Initializing Pygame...")
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("MAPPO-LLM-V3 HRL Visualization")
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.checkpoint
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    agent_state = state_dict['model_state_dict']
    
    is_deep = "critic_mlp.4.weight" in agent_state

    # V3 uses 3 + NUM_ITEMS + NUM_OPTIONS for flag_dim
    agent = RoleConditionedMAPPOAgentV2(cnn_channels=9, goal_dim=3, flag_dim=NUM_ITEMS + NUM_OPTIONS, deep=is_deep)
    agent.load_state_dict(agent_state)
    agent.eval()
    agent.to(device)

    env = BatchCraftingEnvV2(n_envs=1, seed=42)
    
    print("Loading LLM Orchestrator...")
    bridge = LLMBridge(backend=args.llm_backend, model_name=args.llm_model)
    if args.llm_backend.startswith("huggingface"):
        bridge.swap_model(args.llm_model, backend=args.llm_backend)
        if args.disable_lora:
            bridge.disable_lora()
        
    prompt_builder = PromptBuilder()
    option_controller = OptionController(n_envs=1)
    
    # Trigger initial prompt
    print("Fetching initial option...")
    initial_prompt = prompt_builder.build_hrl_prompt(
        {"wood":0, "stone":0, "iron":0, "pickaxe":0, "sword":0, "armor":0, "gold":0, "bridge":0, "enemy":0},
        "Starting", "Starting"
    )
    res = bridge.query_sync(initial_prompt)
    option_controller.update_options_from_llm(res)
    
    print("Starting visualization loop...")
    running = True
    obs_raw, _ = env.reset()
    
    rnn_state = torch.zeros(2, 256, device=device)
    step_role_ids = torch.tensor([0, 1], dtype=torch.long, device=device)
    
    visual_effects = []
    rev_map_inv = {0: "Wood", 1: "Stone", 2: "Iron", 3: "Pickaxe", 4: "Sword", 5: "Armor", 6: "Gold", 7: "Bridge", 8: "Enemy"}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs_raw, _ = env.reset()
                    rnn_state = torch.zeros(2, 256, device=device)
                    visual_effects.clear()
                    print("Environment reset manually.")

        # Update effects
        alive_effects = []
        for fx in visual_effects:
            fx['y'] -= 2
            fx['timer'] -= 1
            if fx['timer'] > 0:
                alive_effects.append(fx)
        visual_effects = alive_effects

        all_fov, all_gmap = env._get_obs_batch_fov()
        fov_t = torch.from_numpy(all_fov.reshape(2, 9, 7, 7)).to(device)
        
        inv = obs_raw[:, 0, 4:4+NUM_ITEMS]
        prev_inv = inv[0].copy()
        
        # Check success to trigger LLM
        a0_opt = option_controller.get_active_option(0)
        a1_opt = option_controller.get_active_option(1)
        
        a0_success = check_option_success([a0_opt], np.expand_dims(prev_inv, 0), np.expand_dims(prev_inv, 0)) # Fake previous step success to trigger correctly
        # Wait, check_option_success takes inv_prev, inv_next. We check after step!
        
        goal_emb = np.zeros((1, 2, 3), dtype=np.float32)
        inv_repeat = np.stack([inv, inv], axis=1)
        opt_repeat = option_controller.get_option_embeddings()
        
        vec_input = np.concatenate([goal_emb, inv_repeat, opt_repeat], axis=2)
        vec_t = torch.from_numpy(vec_input.reshape(2, 3 + NUM_ITEMS + NUM_OPTIONS)).to(device)
        gmap_t = torch.zeros(2, 9, 61, 61, device=device)

        with torch.no_grad():
            action, logprob, _, value, rnn_state_out = agent.get_action_and_value(
                fov_t, gmap_t, vec_t, step_role_ids, rnn_state
            )
            rnn_state = rnn_state_out
            
        actions_np = action.cpu().numpy().reshape(1, 2)
        obs_raw, rewards, done, trunc, _ = env.step(actions_np)
        
        new_inv = obs_raw[0, 0, 4:4+NUM_ITEMS]
        
        # Now check success!
        a0_success = check_option_success([a0_opt], np.expand_dims(prev_inv, 0), np.expand_dims(new_inv, 0))
        a1_success = check_option_success([a1_opt], np.expand_dims(prev_inv, 0), np.expand_dims(new_inv, 0))
        
        if option_controller.cooldown_counter[0] > 0:
            option_controller.cooldown_counter[0] -= 1
            
        if (a0_success.any() or a1_success.any() or "IDLE" in [a0_opt, a1_opt]) and not option_controller.llm_pending[0] and option_controller.cooldown_counter[0] == 0:
            print(f"Option terminated. Triggering LLM Orchestrator...")
            option_controller.set_pending([0], True)
            option_controller.cooldown_counter[0] = 50
            inv_arr = new_inv.astype(int)
            inv_dict = {
                "wood": int(inv_arr[0]),
                "stone": int(inv_arr[1]),
                "iron": int(inv_arr[2]),
                "pickaxe": int(inv_arr[3]),
                "sword": int(inv_arr[4]),
                "armor": int(inv_arr[5]),
                "gold": int(inv_arr[6]),
                "bridge": int(inv_arr[7]),
                "enemy": int(inv_arr[8]),
            }
            a0_stat = "Idle/Finished" if a0_success[0] else f"Working on {a0_opt}"
            a1_stat = "Idle/Finished" if a1_success[0] else f"Working on {a1_opt}"
            prompt = prompt_builder.build_hrl_prompt(inv_dict, a0_stat, a1_stat)
            def _cb(res):
                option_controller.update_options_from_llm(res, env_indices=[0])
                option_controller.set_pending([0], False)
            bridge.query_async(prompt, callback=_cb)
        
        diff = new_inv - prev_inv
        for i in range(9):
            if diff[i] > 0:
                name = rev_map_inv[i].lower()
                if name in ["pickaxe", "sword", "armor"]:
                    zx, zy = ZONES["workbench"]
                elif name == "bridge":
                    zx, zy = ZONES["bridge"]
                elif name == "enemy":
                    zx, zy = ZONES["enemy"]
                else:
                    zx, zy = ZONES.get(name, env.pos[0][0])
                
                x_px, y_px = int(zx * SCALE_X), int(zy * SCALE_Y)
                visual_effects.append({
                    'text': f"+1 {rev_map_inv[i]}",
                    'x': x_px - 15,
                    'y': y_px - 20,
                    'timer': 20
                })

        current_options = [option_controller.get_active_option(0), option_controller.get_active_option(1)]

        draw_env(screen, env, font, current_options=current_options, visual_effects=visual_effects)
        pygame.display.flip()

        if done[0] or trunc[0]:
            print(f"Episode finished at step {env.step_counts[0]}. Gold: {obs_raw[0, 0, 4+6] > 0}")
            pygame.time.delay(1000)
            obs_raw, _ = env.reset()
            rnn_state = torch.zeros(2, 256, device=device)
            visual_effects.clear()

        clock.tick(15)

    pygame.quit()
    bridge.close()

if __name__ == "__main__":
    main()
