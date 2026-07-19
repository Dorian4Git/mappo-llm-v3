import re
import os

with open("visualize.py", "r") as f:
    content = f.read()

# Add load_images and ASSETS
imports = """import pygame
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

# Drawing settings
SCALE_X = 12
SCALE_Y = 6  # Squash Y-axis for oblique perspective
WALL_HEIGHT = 20
GRID_PIXELS_X = int(GRID_LIMIT * SCALE_X)
GRID_PIXELS_Y = int(GRID_LIMIT * SCALE_Y)
PANEL_WIDTH = 350
WINDOW_WIDTH = GRID_PIXELS_X + PANEL_WIDTH
WINDOW_HEIGHT = GRID_PIXELS_Y + 80

ASSETS = {}
def load_assets():
    global ASSETS
    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    
    def load(name, scale=None, ck=(255, 0, 255)):
        path = os.path.join(asset_dir, name)
        try:
            img = pygame.image.load(path).convert()
            if ck: img.set_colorkey(ck)
            if scale: img = pygame.transform.scale(img, scale)
            return img
        except Exception as e:
            print(f"Warning: could not load {name}")
            s = pygame.Surface(scale if scale else (24, 24))
            s.fill((255, 0, 0))
            return s

    sw, sh = int(SCALE_X * 3), int(SCALE_Y * 5)
    
    # Backgrounds
    ASSETS['grass'] = load("grass_tile.png", scale=(64, 64), ck=None)
    ASSETS['water'] = load("water_tile.png", scale=(64, 64), ck=None)
    
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
                    label = font.render("Enemy", True, COLOR_TEXT)
                    screen.blit(label, (ex_px - 20, ey_px - 45))
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
"""

# Extract the rest of main() and everything below
match = re.search(r"def main\(\):.*", content, re.DOTALL)
main_content = match.group(0)

# Replace draw_env call in main_content to pass prev_pos
main_content = main_content.replace(
    "draw_env(screen, env, font, current_options=current_options, visual_effects=visual_effects)",
    "draw_env(screen, env, font, current_options=current_options, visual_effects=visual_effects, prev_pos=prev_pos)"
)

# Also need to store prev_pos in main loop
main_content = main_content.replace(
    "all_fov, all_gmap = env._get_obs_batch_fov()",
    "prev_pos = env.pos[0].copy()\n        all_fov, all_gmap = env._get_obs_batch_fov()"
)

# Need to handle the first frame where prev_pos might not exist yet
main_content = main_content.replace(
    "obs_raw, _ = env.reset()\n    \n    rnn_state",
    "obs_raw, _ = env.reset()\n    prev_pos = env.pos[0].copy()\n    rnn_state"
)

# Also call load_assets after pygame.init()
main_content = main_content.replace(
    "pygame.time.Clock()",
    "pygame.time.Clock()\n    load_assets()"
)

with open("visualize_new.py", "w") as f:
    f.write(imports + "\n" + main_content)
