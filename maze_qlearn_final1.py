"""
Controls:
    Left-click: toggle wall (click and drag supported)
    S then left-click: set Start cell
    G then left-click: set Goal cell
    T: Train Q-learning (visual training shown in UI)
    R: Run the learned policy (visual)
    C: Clear walls
    M: Random maze
    Esc / window close: Quit
"""

import pygame
import numpy as np
import random
import sys
from time import sleep

# ---------- CONFIG (modified for small/fast demo) ----------
GRID_W = 6              # number of columns (small maze)
GRID_H = 6              # number of rows    (small maze)
CELL_SIZE = 60          # pixels (bigger so 6x6 is visible)
MARGIN = 2              # pixels between cells
WINDOW_W = GRID_W * (CELL_SIZE + MARGIN) + MARGIN
WINDOW_H = GRID_H * (CELL_SIZE + MARGIN) + 120  # extra for UI text

# Q-learning hyperparams (reduced for speed)
ALPHA = 0.7             # learning rate
GAMMA = 0.95            # discount factor
EPSILON_START = 1.0     # initial exploration
EPSILON_END = 0.05
EPSILON_DECAY = 0.995   # faster decay for demo
EPISODES = 100          # fewer episodes for quicker training
MAX_STEPS_EP = 1000     # max steps per episode

# Visualization parameters
FPS = 60
RUN_DELAY = 0.05        # delay between steps when demonstrating a run (seconds)

# Rewards
STEP_PENALTY = -1.0
HIT_WALL_PENALTY = -5.0
GOAL_REWARD = 100.0

# ---------- END CONFIG ----------

# Colors
COLOR_BG = (30, 30, 30)
COLOR_EMPTY = (240, 240, 240)
COLOR_WALL = (40, 40, 40)
COLOR_START = (50, 200, 50)
COLOR_GOAL = (200, 50, 50)
COLOR_AGENT = (50, 150, 250)
COLOR_TEXT = (230, 230, 230)
COLOR_PATH = (220, 180, 20)

# Cell types
EMPTY = 0
WALL = 1
START = 2
GOAL = 3

# Actions: up, right, down, left
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
N_ACTIONS = len(ACTIONS)

def state_to_idx(x, y):
    return y * GRID_W + x

def idx_to_state(idx):
    x = idx % GRID_W
    y = idx // GRID_W
    return x, y

class MazeEnv:
    def __init__(self, grid=None):
        # grid is GRID_H x GRID_W array with values EMPTY/WALL/START/GOAL
        if grid is None:
            self.grid = np.zeros((GRID_H, GRID_W), dtype=int)
        else:
            self.grid = grid.copy()
        self.start = None
        self.goal = None
        self._find_special_cells()
        self.reset()

    def _find_special_cells(self):
        locs = np.where(self.grid == START)
        if len(locs[0]) > 0:
            self.start = (locs[1][0], locs[0][0])
        else:
            self.start = None
        locs = np.where(self.grid == GOAL)
        if len(locs[0]) > 0:
            self.goal = (locs[1][0], locs[0][0])
        else:
            self.goal = None

    def set_start(self, x, y):
        # ensure only one start
        self.grid[self.grid == START] = EMPTY
        self.grid[y, x] = START
        self.start = (x, y)

    def set_goal(self, x, y):
        self.grid[self.grid == GOAL] = EMPTY
        self.grid[y, x] = GOAL
        self.goal = (x, y)

    def reset(self):
        if self.start is None:
            # choose a default start if not set (top-left empty)
            for yy in range(GRID_H):
                for xx in range(GRID_W):
                    if self.grid[yy, xx] == EMPTY:
                        self.agent_pos = (xx, yy)
                        break
                else:
                    continue
                break
        else:
            self.agent_pos = self.start
        return state_to_idx(*self.agent_pos)

    def step(self, action):
        ax, ay = self.agent_pos
        dx, dy = ACTIONS[action]
        nx, ny = ax + dx, ay + dy

        # check bounds
        if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
            # invalid move, stay
            reward = HIT_WALL_PENALTY
            done = False
            next_state = state_to_idx(ax, ay)
            return next_state, reward, done

        # check wall
        if self.grid[ny, nx] == WALL:
            # hit wall, stay in same cell
            reward = HIT_WALL_PENALTY
            done = False
            next_state = state_to_idx(ax, ay)
            return next_state, reward, done

        # move
        self.agent_pos = (nx, ny)
        # goal?
        if self.grid[ny, nx] == GOAL:
            return state_to_idx(nx, ny), GOAL_REWARD, True

        return state_to_idx(nx, ny), STEP_PENALTY, False

    def n_states(self):
        return GRID_W * GRID_H

    def is_free(self, x, y):
        return 0 <= x < GRID_W and 0 <= y < GRID_H and self.grid[y, x] != WALL

# ---------- Q-LEARNING TRAINING (non-visual) ----------
def train_q_learning(env: MazeEnv, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
                     eps_start=EPSILON_START, eps_end=EPSILON_END, eps_decay=EPSILON_DECAY,
                     max_steps=MAX_STEPS_EP, render_callback=None, render_every=500):
    n_states = env.n_states()
    Q = np.zeros((n_states, N_ACTIONS), dtype=float)
    epsilon = eps_start

    if env.start is None or env.goal is None:
        raise ValueError("Start and Goal must be set before training.")

    for ep in range(1, episodes + 1):
        state = env.reset()
        env.agent_pos = env.start
        total_reward = 0.0
        for step in range(max_steps):
            # epsilon-greedy
            if random.random() < epsilon:
                action = random.randrange(N_ACTIONS)
            else:
                # tie-breaking random choice among best actions
                best = np.flatnonzero(Q[state] == Q[state].max())
                action = random.choice(best)

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q update
            best_next = Q[next_state].max()
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            if done:
                break

        # decay epsilon
        if epsilon > eps_end:
            epsilon *= eps_decay
            if epsilon < eps_end:
                epsilon = eps_end

        # optional rendering callback for progress (not per-step)
        if render_callback is not None and (ep % render_every == 0 or ep == 1 or ep == episodes):
            render_callback(ep, total_reward, epsilon)

    return Q

# ---------- VISUAL TRAINING (UI TRAINING) ----------
def train_q_learning_visual(env: MazeEnv, screen, font):
    """
    Visual training: runs Q-learning while updating the pygame UI each step so user can
    watch the agent explore and learn. Returns the trained Q-table.
    """
    n_states = env.n_states()
    Q = np.zeros((n_states, N_ACTIONS), dtype=float)
    epsilon = EPSILON_START

    if env.start is None or env.goal is None:
        raise ValueError("Start and Goal must be set before training.")

    # We will allow user to cancel training by closing the window.
    for ep in range(1, EPISODES + 1):
        state = env.reset()
        env.agent_pos = env.start
        total_reward = 0.0

        for step in range(MAX_STEPS_EP):
            # handle events so window stays responsive
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # allow escape to quit
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            # epsilon-greedy selection
            if random.random() < epsilon:
                action = random.randrange(N_ACTIONS)
            else:
                best = np.flatnonzero(Q[state] == Q[state].max())
                action = random.choice(best)

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q update
            best_next = Q[next_state].max()
            Q[state, action] = Q[state, action] + ALPHA * (reward + GAMMA * best_next - Q[state, action])

            state = next_state

            # Draw current training step to UI (render every step - okay for small 6x6)
            draw_grid(screen, env, agent_pos=env.agent_pos, path=None, font=font)

            # draw status text
            msg = f"Training (visual) - Episode: {ep}/{EPISODES}   Step: {step}   ε={epsilon:.3f}"
            info_y = GRID_H * (CELL_SIZE + MARGIN) + MARGIN + 60
            txt_surf = font.render(msg, True, COLOR_TEXT)
            # overlay text on top of existing draw
            screen.blit(txt_surf, (10, info_y))
            pygame.display.update()

            # small delay to visualize; adjust as needed
            pygame.time.delay(15)

            if done:
                break

        # decay epsilon
        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY
            if epsilon < EPSILON_END:
                epsilon = EPSILON_END

        # occasional console log
        if ep % 100 == 0 or ep == 1 or ep == EPISODES:
            print(f"[VISUAL TRAIN] Episode {ep}/{EPISODES} - last_total_reward={total_reward:.1f} - epsilon={epsilon:.4f}")

    return Q

# ---------- POLICY EXECUTION ----------
def run_policy(env: MazeEnv, Q, max_steps=GRID_W * GRID_H * 4, render_fn=None, delay=RUN_DELAY):
    if env.start is None or env.goal is None:
        raise ValueError("Start and Goal must be set before running the policy.")
    env.agent_pos = env.start
    state = state_to_idx(*env.agent_pos)
    path = [env.agent_pos]
    for step in range(max_steps):
        # greedy action
        best = np.flatnonzero(Q[state] == Q[state].max())
        action = random.choice(best)
        next_state, reward, done = env.step(action)
        state = next_state
        path.append(env.agent_pos)
        if render_fn:
            render_fn(env, path)
            sleep(delay)
        if done:
            return True, path
    return False, path

# ---------- PYGAME GUI ----------
def draw_grid(screen, env: MazeEnv, agent_pos=None, path=None, font=None):
    screen.fill(COLOR_BG)
    # draw grid cells
    for y in range(GRID_H):
        for x in range(GRID_W):
            rect = pygame.Rect(MARGIN + x * (CELL_SIZE + MARGIN),
                               MARGIN + y * (CELL_SIZE + MARGIN),
                               CELL_SIZE, CELL_SIZE)
            cell = env.grid[y, x]
            color = COLOR_EMPTY
            if cell == WALL:
                color = COLOR_WALL
            elif cell == START:
                color = COLOR_START
            elif cell == GOAL:
                color = COLOR_GOAL
            pygame.draw.rect(screen, color, rect)

    # draw path if provided
    if path is not None:
        for (x, y) in path:
            rect = pygame.Rect(MARGIN + x * (CELL_SIZE + MARGIN),
                               MARGIN + y * (CELL_SIZE + MARGIN),
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLOR_PATH, rect)

    # draw agent
    if agent_pos is not None:
        ax, ay = agent_pos
        rect = pygame.Rect(MARGIN + ax * (CELL_SIZE + MARGIN) + CELL_SIZE // 6,
                           MARGIN + ay * (CELL_SIZE + MARGIN) + CELL_SIZE // 6,
                           CELL_SIZE * 2 // 3, CELL_SIZE * 2 // 3)
        pygame.draw.ellipse(screen, COLOR_AGENT, rect)

    # UI box/ texts below the grid
    info_y = GRID_H * (CELL_SIZE + MARGIN) + MARGIN + 10
    if font:
        lines = [
            "Controls: Left-click: toggle wall    S + click: set Start    G + click: set Goal",
            "Keys: T = Train (visual)    R = Run    C = Clear walls    M = Random maze    Esc = Quit",
        ]
        for i, txt in enumerate(lines):
            surf = font.render(txt, True, COLOR_TEXT)
            screen.blit(surf, (10, info_y + 20 * i))
    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Maze Q-learning Demo (6x6)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    env = MazeEnv()
    q_table = None
    agent_pos = None
    drawing = False
    set_start_mode = False
    set_goal_mode = False

    message = "Set Start (S) and Goal (G), then press T to train."

    def render_callback_progress(ep, total_reward, eps):
        # prints progress to console; we don't render per training step (too slow)
        print(f"Episode {ep}/{EPISODES} - last_total_reward={total_reward:.1f} - epsilon={eps:.4f}")

    # helper: converts mouse pos to grid coords
    def mouse_to_cell(mx, my):
        if mx < MARGIN or my < MARGIN:
            return None
        x = (mx - MARGIN) // (CELL_SIZE + MARGIN)
        y = (my - MARGIN) // (CELL_SIZE + MARGIN)
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            return x, y
        return None

    # helper render function when running policy
    run_path = None
    def render_run(current_env, path):
        nonlocal run_path
        run_path = path.copy()
        draw_grid(screen, current_env, agent_pos=current_env.agent_pos, path=run_path, font=font)

    # initial draw
    draw_grid(screen, env, font=font)

    training_done = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == pygame.K_s:
                    set_start_mode = True
                    set_goal_mode = False
                    message = "Click a cell to set START."
                if event.key == pygame.K_g:
                    set_goal_mode = True
                    set_start_mode = False
                    message = "Click a cell to set GOAL."
                if event.key == pygame.K_c:
                    # clear walls
                    env.grid[env.grid == WALL] = EMPTY
                    draw_grid(screen, env, font=font)
                    message = "Walls cleared."
                if event.key == pygame.K_m:
                    # random maze (simple random walls)
                    env.grid = np.zeros((GRID_H, GRID_W), dtype=int)
                    # fill random walls but keep some path
                    for y in range(GRID_H):
                        for x in range(GRID_W):
                            if random.random() < 0.25:
                                env.grid[y, x] = WALL
                    # ensure border free
                    env.grid[0, :] = 0
                    env.grid[-1, :] = 0
                    env.grid[:, 0] = 0
                    env.grid[:, -1] = 0
                    env._find_special_cells()
                    draw_grid(screen, env, font=font)
                    message = "Random maze generated. (reset start/goal if needed)"
                if event.key == pygame.K_t:
                    # train visually
                    if env.start is None or env.goal is None:
                        message = "Set Start and Goal before training!"
                    else:
                        message = "Training visually... Please wait."
                        draw_grid(screen, env, font=font)
                        pygame.display.flip()

                        try:
                            q_table = train_q_learning_visual(env, screen, font)
                            training_done = True
                            message = "Visual Training finished! Press R to run."
                        except ValueError as e:
                            message = str(e)
                if event.key == pygame.K_r:
                    # run the learned policy
                    if q_table is None:
                        message = "Train first (press T) before running (R)."
                    else:
                        # reset path
                        run_path = None
                        success, path = run_policy(env, q_table, max_steps=GRID_W * GRID_H * 4, render_fn=render_run, delay=RUN_DELAY)
                        if success:
                            message = f"Agent reached goal in {len(path)-1} steps!"
                        else:
                            message = "Agent failed to reach goal within step limit; try training more or changing maze."

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                cell = mouse_to_cell(mx, my)
                if cell:
                    x, y = cell
                    if set_start_mode:
                        env.set_start(x, y)
                        set_start_mode = False
                        message = f"Start set at ({x}, {y})."
                    elif set_goal_mode:
                        env.set_goal(x, y)
                        set_goal_mode = False
                        message = f"Goal set at ({x}, {y})."
                    else:
                        # toggle wall on left click
                        if event.button == 1:
                            if env.grid[y, x] == WALL:
                                env.grid[y, x] = EMPTY
                            else:
                                # avoid overwriting start/goal
                                if env.grid[y, x] in (START, GOAL):
                                    pass
                                else:
                                    env.grid[y, x] = WALL
                            drawing = True
                            message = "Drawing walls..."
                        # right click could be used to erase
                        elif event.button == 3:
                            if env.grid[y, x] == WALL:
                                env.grid[y, x] = EMPTY
                            message = "Erasing walls..."

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                message = "Ready."

            if event.type == pygame.MOUSEMOTION and drawing:
                # allow click-and-drag drawing
                mx, my = pygame.mouse.get_pos()
                cell = mouse_to_cell(mx, my)
                if cell:
                    x, y = cell
                    if env.grid[y, x] != WALL:
                        if env.grid[y, x] not in (START, GOAL):
                            env.grid[y, x] = WALL

        # draw current state
        # place agent at start if trained or during run
        agent_display_pos = None
        if env.start:
            agent_display_pos = env.agent_pos if hasattr(env, 'agent_pos') and env.agent_pos is not None else env.start

        draw_grid(screen, env, agent_pos=agent_display_pos, path=run_path, font=font)

        # show message text box
        info_y = GRID_H * (CELL_SIZE + MARGIN) + MARGIN + 60
        msg_surf = font.render(message, True, COLOR_TEXT)
        screen.blit(msg_surf, (10, info_y))

        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:", e)
        pygame.quit()
        raise


