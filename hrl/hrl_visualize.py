import pygame
import torch
import numpy as np
import time
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.crafting_env import NUM_ITEMS, ZONES, GRID_LIMIT, DIST_THRESHOLD
from hrl.hrl_crafting_env import HRLCraftingEnv
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

# Drawing settings
SCALE_X = 24
SCALE_Y = 12  # Squash Y-axis for oblique perspective
WALL_HEIGHT = 40
GRID_PIXELS_X = int(GRID_LIMIT * SCALE_X)
GRID_PIXELS_Y = int(GRID_LIMIT * SCALE_Y)
PANEL_WIDTH = 700
WINDOW_WIDTH = GRID_PIXELS_X + PANEL_WIDTH
WINDOW_HEIGHT = GRID_PIXELS_Y + 80

ASSETS = {}
def load_assets():
    global ASSETS
    asset_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    
    def load(name, scale=None):
        path = os.path.join(asset_dir, name)
        try:
            img = pygame.image.load(path).convert_alpha()
            if scale: img = pygame.transform.scale(img, scale)
            return img
        except Exception as e:
            print(f"Warning: could not load {name}")
            s = pygame.Surface(scale if scale else (24, 24))
            s.fill((255, 0, 0))
            return s

    sw, sh = int(SCALE_X * 3), int(SCALE_Y * 5)
    
    # Backgrounds
    ASSETS['grass'] = load("grass_tile.png", scale=(128, 128))
    ASSETS['water'] = load("water_tile.png", scale=(128, 128))
    
    # Bridge
    ASSETS['bridge'] = load("bridge_tile.png", scale=(int(SCALE_X*6), int(SCALE_Y*6)))
    
    # Entities
    ASSETS['agent_0'] = load("agent_0.png", scale=(sw, sh))
    ASSETS['agent_1'] = load("agent_1.png", scale=(sw, sh))
    ASSETS['enemy_alive'] = load("enemy_alive.png", scale=(sw*2, int(sh*1.5)))
    ASSETS['enemy_dead'] = load("enemy_dead.png", scale=(sw*2, int(sh*1.5)))
    
    # Resources
    ASSETS['wood'] = load("wood_resource.png", scale=(sw, sw))
    ASSETS['stone'] = load("stone_resource.png", scale=(sw, sw))
    ASSETS['iron'] = load("iron_resource.png", scale=(sw, sw))
    ASSETS['gold'] = load("gold_resource.png", scale=(sw, sw))
    ASSETS['workbench'] = load("workbench.png", scale=(sw, sw))
    
    # Obstacle
    ASSETS['obstacle'] = load("obstacle_rock.png", scale=(int(sw*1.5), int(sh*1.2)))

AGENT_FACING_LEFT = [False, False]

def draw_env(screen, env, font, current_options=None, visual_effects=None, prev_pos=None):
    from core.crafting_env import RIVER_X_MIN, RIVER_X_MAX, BRIDGE_Y_MIN, BRIDGE_Y_MAX
    from core.crafting_env import I_WOOD, I_STONE, I_IRON, I_PICKAXE, I_SWORD, I_ARMOR, I_GOLD, F_BRIDGE, F_ENEMY_DEFEATED, F_GAME_OVER
    
    inv = env.inventory[0]
    pos = env.pos[0]
    
    # Update facing direction
    if prev_pos is not None:
        for i in range(2):
            if pos[i][0] < prev_pos[i][0]:
                AGENT_FACING_LEFT[i] = True
            elif pos[i][0] > prev_pos[i][0]:
                AGENT_FACING_LEFT[i] = False
    
    # Draw Background Tiles
    if 'grass' in ASSETS:
        gw, gh = ASSETS['grass'].get_size()
        for x in range(0, GRID_PIXELS_X, gw):
            for y in range(0, GRID_PIXELS_Y, gh):
                screen.blit(ASSETS['grass'], (x, y))
    else:
        screen.fill(COLOR_BG)
        
    # Draw Paths
    path_color = (130, 100, 60)
    pygame.draw.rect(screen, path_color, pygame.Rect(20 * SCALE_X, 39 * SCALE_Y, 15 * SCALE_X, 4 * SCALE_Y))
    pygame.draw.rect(screen, path_color, pygame.Rect(36 * SCALE_X, 39 * SCALE_Y, 10 * SCALE_X, 4 * SCALE_Y))
    pygame.draw.rect(screen, path_color, pygame.Rect(45 * SCALE_X, 20 * SCALE_Y, 4 * SCALE_X, 20 * SCALE_Y))
    pygame.draw.rect(screen, path_color, pygame.Rect(45 * SCALE_X, 20 * SCALE_Y, 5 * SCALE_X, 4 * SCALE_Y))

    # River
    river_rect = pygame.Rect(RIVER_X_MIN * SCALE_X, 0, (RIVER_X_MAX - RIVER_X_MIN) * SCALE_X, GRID_PIXELS_Y)
    if 'water' in ASSETS:
        ww, wh = ASSETS['water'].get_size()
        for x in range(int(RIVER_X_MIN * SCALE_X), int(RIVER_X_MAX * SCALE_X), ww):
            for y in range(0, GRID_PIXELS_Y, wh):
                # Clip width to river boundary
                blit_w = min(ww, int(RIVER_X_MAX * SCALE_X) - x)
                screen.blit(ASSETS['water'], (x, y), (0, 0, blit_w, wh))
    else:
        pygame.draw.rect(screen, (30, 80, 150), river_rect)

    if inv[F_BRIDGE] > 0:
        bx = RIVER_X_MIN * SCALE_X
        by = BRIDGE_Y_MIN * SCALE_Y
        bh = (BRIDGE_Y_MAX - BRIDGE_Y_MIN) * SCALE_Y
        if 'bridge' in ASSETS:
            # blit multiple bridges if needed
            bw, bh_img = ASSETS['bridge'].get_size()
            for y_off in range(int(by), int(by + bh), bh_img):
                screen.blit(ASSETS['bridge'], (bx - bw//4, y_off))
        else:
            bridge_rect = pygame.Rect(bx, by, (RIVER_X_MAX - RIVER_X_MIN) * SCALE_X, bh)
            pygame.draw.rect(screen, (139, 105, 20), bridge_rect)
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
        if name == "enemy" and hasattr(env, 'enemy_pos'):
            zx, zy = env.enemy_pos[0][0], env.enemy_pos[0][1]
        else:
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
            if 'obstacle' in ASSETS:
                ow, oh = ASSETS['obstacle'].get_size()
                # Tile obstacles
                for ox in range(int(rx0), int(rx0 + rw), int(ow * 0.8)):
                    screen.blit(ASSETS['obstacle'], (ox, ry0 + rh - oh))
            else:
                front_rect = pygame.Rect(rx0, ry0 + rh - WALL_HEIGHT, rw, WALL_HEIGHT)
                pygame.draw.rect(screen, (80, 80, 80), front_rect)
                top_rect = pygame.Rect(rx0, ry0 - WALL_HEIGHT, rw, rh)
                pygame.draw.rect(screen, COLOR_OBSTACLE, top_rect)
            
        elif item['type'] == 'zone':
            zx, zy, name, flag_idx = item['x'], item['y'], item['name'], item['flag_idx']
            
            ex_px, ey_px = int(zx * SCALE_X), int(zy * SCALE_Y)
            if name == "enemy":
                if inv[flag_idx] == 0:
                    if 'enemy_alive' in ASSETS:
                        w, h = ASSETS['enemy_alive'].get_size()
                        screen.blit(ASSETS['enemy_alive'], (ex_px - w//2, ey_px - h))
                    else:
                        pygame.draw.circle(screen, (220, 30, 30), (ex_px, ey_px - 15), 18)
                    
                    # Draw Health Bar
                    if hasattr(env, 'enemy_health'):
                        hp = env.enemy_health[0]
                        hp_pct = max(0, min(1.0, hp / 100.0))
                        bar_w = 40
                        bar_h = 6
                        bar_x = ex_px - bar_w // 2
                        bar_y = ey_px - (h if 'enemy_alive' in ASSETS else 35) - 10
                        pygame.draw.rect(screen, (255, 0, 0), (bar_x, bar_y, bar_w, bar_h))
                        pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, int(bar_w * hp_pct), bar_h))
                        
                    label = font.render("Enemy", True, COLOR_TEXT)
                    screen.blit(label, (ex_px - 20, ey_px - 45))
                    
                    # Combat FX (Slash lines if agent with sword is within 2.0)
                    for a_idx in range(2):
                        d = np.linalg.norm(pos[a_idx] - [zx, zy])
                        if d < 2.0 and inv[I_SWORD] >= 1:
                            ax_px, ay_px = int(pos[a_idx][0] * SCALE_X), int(pos[a_idx][1] * SCALE_Y)
                            pygame.draw.line(screen, (255, 50, 50), (ax_px, ay_px), (ex_px, ey_px - 15), 4)
                            
                else:
                    if 'enemy_dead' in ASSETS:
                        w, h = ASSETS['enemy_dead'].get_size()
                        screen.blit(ASSETS['enemy_dead'], (ex_px - w//2, ey_px - h//2))
                    else:
                        pygame.draw.circle(screen, (100, 30, 30), (ex_px, ey_px - 10), 12)
                continue
                
            img_key = name
            if img_key in ASSETS:
                img = ASSETS[img_key]
                if inv[flag_idx] > 0 and name != "workbench":
                    # Make it darker or semi-transparent if collected
                    img = img.copy()
                    img.set_alpha(100)
                w, h = img.get_size()
                screen.blit(img, (ex_px - w//2, ey_px - h//2))
                
                if inv[flag_idx] > 0 and name != "workbench":
                    done_label = font.render("(Done)", True, (200, 200, 200))
                    screen.blit(done_label, (ex_px - 20, ey_px + h//2))
            else:
                pygame.draw.circle(screen, (255, 255, 255), (ex_px, ey_px), 10)
                
        elif item['type'] == 'agent':
            ax, ay, a_idx = item['x'], item['y'], item['id']
            sx, sy = int(ax * SCALE_X), int(ay * SCALE_Y)
            
            img_key = f"agent_{a_idx}"
            if img_key in ASSETS:
                img = ASSETS[img_key]
                if AGENT_FACING_LEFT[a_idx]:
                    img = pygame.transform.flip(img, True, False)
                w, h = img.get_size()
                
                # Shadow
                pygame.draw.ellipse(screen, (20, 20, 20), (sx - w//2, sy - h//6, w, h//3))
                
                screen.blit(img, (sx - w//2, sy - h + h//4))
                
                if inv[I_ARMOR] > 0:
                    pygame.draw.ellipse(screen, (200, 200, 255, 100), (sx - w//2 - 2, sy - h + h//4 - 2, w + 4, h + 4), 2)
            else:
                color = COLOR_A0 if a_idx == 0 else COLOR_A1
                pygame.draw.circle(screen, color, (sx, sy - 10), 8)
                
            a_label = font.render(f"A{a_idx}", True, (255, 255, 255))
            screen.blit(a_label, (sx - 6, sy - 25))

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
    
    from core.crafting_env import F_GAME_OVER
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
    parser.add_argument("--llm-model", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models", "qlora_adapter")))
    parser.add_argument("--disable-lora", action="store_true", help="Run the base Qwen model without the LoRA adapter")
    args = parser.parse_args()

    print("Initializing Pygame...")
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("MAPPO-LLM-V3 HRL Visualization")
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()
    load_assets()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.checkpoint
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    agent_state = state_dict['model_state_dict']
    
    is_deep = "critic_mlp.4.weight" in agent_state

    # V3 uses 3 + 2 + NUM_ITEMS + NUM_OPTIONS for flag_dim
    agent = RoleConditionedMAPPOAgentV2(cnn_channels=9, goal_dim=3, flag_dim=2 + NUM_ITEMS + NUM_OPTIONS, deep=is_deep)
    agent.load_state_dict(agent_state)
    agent.eval()
    agent.to(device)

    env = HRLCraftingEnv(n_envs=1, seed=42)
    
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
    prev_pos = env.pos[0].copy()
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

        # If LLM is thinking, pause the environment so agents don't drift with old options
        if option_controller.llm_pending[0]:
            current_options = [option_controller.get_active_option(0, env_id=0), option_controller.get_active_option(1, env_id=0)]
            draw_env(screen, env, font, current_options=current_options, visual_effects=visual_effects, prev_pos=env.pos[0].copy())
            
            # Draw thinking overlay
            text = font.render("LLM Thinking...", True, (255, 255, 0))
            bg = pygame.Surface((text.get_width() + 20, text.get_height() + 20))
            bg.set_alpha(200)
            bg.fill((0, 0, 0))
            x = WINDOW_WIDTH // 2 - bg.get_width() // 2
            y = WINDOW_HEIGHT // 2 - bg.get_height() // 2
            screen.blit(bg, (x, y))
            screen.blit(text, (x + 10, y + 10))
            
            pygame.display.flip()
            clock.tick(15)
            continue

        # Update effects
        alive_effects = []
        for fx in visual_effects:
            fx['y'] -= 2
            fx['timer'] -= 1
            if fx['timer'] > 0:
                alive_effects.append(fx)
        visual_effects = alive_effects

        prev_pos = env.pos[0].copy()
        all_fov, all_gmap = env._get_obs_batch_fov()
        fov_t = torch.from_numpy(all_fov.reshape(2, 9, 7, 7)).to(device)
        
        inv = obs_raw[:, 0, 4:4+NUM_ITEMS]
        prev_inv = inv[0].copy()
        
        # Check success to trigger LLM
        a0_opt = option_controller.get_active_option(0, env_id=0)
        a1_opt = option_controller.get_active_option(1, env_id=0)
        
        a0_success = check_option_success([a0_opt], np.expand_dims(prev_inv, 0), np.expand_dims(prev_inv, 0)) # Fake previous step success to trigger correctly
        # Wait, check_option_success takes inv_prev, inv_next. We check after step!
        
        dynamic_enemy_state = np.column_stack((
            env.enemy_pos[:, 0], 
            env.enemy_pos[:, 1], 
            env.enemy_health / 100.0
        ))
        goal_emb = np.stack([dynamic_enemy_state, dynamic_enemy_state], axis=1)
        inv_repeat = np.stack([inv, inv], axis=1)
        opt_repeat = option_controller.get_option_embeddings()
        pos_repeat = env.pos.copy()
        
        vec_input = np.concatenate([pos_repeat, goal_emb, inv_repeat, opt_repeat], axis=2)
        vec_t = torch.from_numpy(vec_input.reshape(2, 2 + 3 + NUM_ITEMS + NUM_OPTIONS)).to(device)
        gmap_t = torch.zeros(2, 9, 61, 61, device=device)

        with torch.no_grad():
            action, logprob, _, value, rnn_state_out = agent.get_action_and_value(
                fov_t, gmap_t, vec_t, step_role_ids, rnn_state
            )
            rnn_state = rnn_state_out
            
        actions_np = action.cpu().numpy().reshape(1, 2)
        obs_raw, rewards, done, trunc, info = env.step(actions_np)
        
        if done[0] or trunc[0]:
            new_inv = info['terminal_flags'][0, :NUM_ITEMS]
        else:
            new_inv = obs_raw[0, 0, 4:4+NUM_ITEMS]
        
        # Check success: did the inventory change THIS frame?
        a0_success = check_option_success([a0_opt], np.expand_dims(prev_inv, 0), np.expand_dims(new_inv, 0))
        a1_success = check_option_success([a1_opt], np.expand_dims(prev_inv, 0), np.expand_dims(new_inv, 0))
        
        # Also check: is the current option ALREADY completed? (inventory already has the item)
        # This catches the case where the LLM assigned an option that was already done.
        OPTION_TO_INV = {
            "COLLECT_WOOD": 0, "COLLECT_STONE": 1, "MINE_IRON": 2,
            "CRAFT_PICKAXE": 3, "CRAFT_SWORD": 4, "CRAFT_ARMOR": 5,
            "COLLECT_GOLD": 6, "BUILD_BRIDGE": 7, "FIGHT_ENEMY": 8,
        }
        a0_already_done = a0_opt in OPTION_TO_INV and new_inv[OPTION_TO_INV[a0_opt]] > 0
        a1_already_done = a1_opt in OPTION_TO_INV and new_inv[OPTION_TO_INV[a1_opt]] > 0
        
        needs_new_options = (
            a0_success.any() or a1_success.any() or 
            a0_already_done or a1_already_done or
            "IDLE" in [a0_opt, a1_opt]
        )
        
        if option_controller.cooldown_counter[0] > 0:
            option_controller.cooldown_counter[0] -= 1
            
        if needs_new_options and not option_controller.llm_pending[0] and option_controller.cooldown_counter[0] == 0:
            print(f"Option terminated (a0={a0_opt}, a1={a1_opt}). Triggering LLM Orchestrator...")
            option_controller.set_pending([0], True)
            option_controller.cooldown_counter[0] = 30
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
            a0_done = a0_success[0] or a0_already_done
            a1_done = a1_success[0] or a1_already_done
            a0_stat = "Idle/Finished" if a0_done else f"Working on {a0_opt}"
            a1_stat = "Idle/Finished" if a1_done else f"Working on {a1_opt}"
            prompt = prompt_builder.build_hrl_prompt(inv_dict, a0_stat, a1_stat)
            print(f"  Inventory: {inv_dict}")
            print(f"  Status: A0={a0_stat}, A1={a1_stat}")
            def _cb(res):
                if res:
                    print(f"  LLM Response: {res[:200]}")
                success = option_controller.update_options_from_llm(res, env_indices=[0])
                if success:
                    new_a0 = option_controller.get_active_option(0, env_id=0)
                    new_a1 = option_controller.get_active_option(1, env_id=0)
                    print(f"  Options updated: A0={new_a0}, A1={new_a1}")
                else:
                    print(f"  WARNING: LLM response failed to parse!")
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

        current_options = [option_controller.get_active_option(0, env_id=0), option_controller.get_active_option(1, env_id=0)]

        draw_env(screen, env, font, current_options=current_options, visual_effects=visual_effects, prev_pos=prev_pos)
        pygame.display.flip()

        if done[0] or trunc[0]:
            # Note: base env already auto-reset inventory/pos/step_counts inside super().step()
            # So we just need to reset our HRL-specific state
            final_gold = new_inv[6] > 0  # Check from new_inv captured BEFORE auto-reset wipes it
            print(f"Episode finished. Gold collected: {final_gold}")
            pygame.time.delay(2000)
            
            # Reset option controller back to initial state
            option_controller.reset_options([0])
            
            # Reset RNN state for fresh episode
            rnn_state = torch.zeros(2, 256, device=device)
            visual_effects.clear()
            
            # Re-trigger initial LLM prompt for the new episode
            option_controller.set_pending([0], True)
            initial_prompt = prompt_builder.build_hrl_prompt(
                {"wood":0, "stone":0, "iron":0, "pickaxe":0, "sword":0, "armor":0, "gold":0, "bridge":0, "enemy":0},
                "Starting", "Starting"
            )
            def _reset_cb(res):
                option_controller.update_options_from_llm(res, env_indices=[0])
                option_controller.set_pending([0], False)
                print(f"  New episode options: A0={option_controller.get_active_option(0, env_id=0)}, A1={option_controller.get_active_option(1, env_id=0)}")
            bridge.query_async(initial_prompt, callback=_reset_cb)

        clock.tick(15)

    pygame.quit()
    bridge.close()

if __name__ == "__main__":
    main()
