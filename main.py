import pygame
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog, TclError
from collections import deque
from UI.button import Button
from UI.slider import Slider
from UI.dropdown import Dropdown
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ----------- Constants and setup -----------
WIDTH, HEIGHT = 800, 600
PADDLE_WIDTH = 10
BALL_RADIUS = 10
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FPS = 60

INIT_BALL_SPEED = 2
INIT_PADDLE_SPEED = 3
INIT_PADDLE_HEIGHT = 120
INIT_EPSILON = 1.0
INIT_EPSILON_DECAY = 0.999
INIT_GAMMA = 0.9
INIT_SHAPING_ALPHA = 20
INIT_UPDATE_TIME = 400
INIT_LEARNING_RATE = 0.0003
INIT_REWARD_CLIP = 5.0

STABILIZE_THRESHOLD = 20
BOT_HIT_TOLERANCE = 10
INIT_TERMINAL_HIT_REWARD = 2.0
INIT_TERMINAL_MISS_PENALTY = -2.0
INIT_WAIT_CENTER_PENALTY = -0.05
INIT_STABILIZE_REWARD = 0.1
INIT_ACTION_CHANGE_PENALTY = -0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_against_human = [False]
pygame.init()
font = pygame.font.SysFont("Arial", 20)
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PingPong RL")
clock = pygame.time.Clock()

# global caches
_cached_reward_surf = None
_cached_loss_surf   = None
_last_graph_update  = 0
GRAPH_UPDATE_INTERVAL = FPS  # once per second

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "STAY"}

# ----------- Classes -----------
class Paddle:
    def __init__(self, x):
        self.paddle_height = INIT_PADDLE_HEIGHT
        self.x = x
        self.y = HEIGHT // 2 - INIT_PADDLE_HEIGHT // 2
        self.speed = 0
        self.speed_mag = INIT_PADDLE_SPEED
    def set_speed(self, speed):
        self.speed_mag = speed
    def move(self, up, down):
        if up:
            self.speed = -self.speed_mag
        elif down:
            self.speed = self.speed_mag
        else:
            self.speed = 0
        self.y += self.speed
        self.y = max(0, min(self.y, HEIGHT - self.paddle_height))
    def draw(self, win):
        pygame.draw.rect(win, WHITE, (self.x, self.y, PADDLE_WIDTH, self.paddle_height))

class Ball:
    def __init__(self):
        self.init_speed = INIT_BALL_SPEED
        self.reset()
    def set_speed(self, speed):
        self.dx = speed * (1 if self.dx > 0 else -1)
        self.dy = speed * (1 if self.dy > 0 else -1)
        self.init_speed = speed
    def reset(self):
        self.x = WIDTH // 2
        self.y = random.randrange(BALL_RADIUS, HEIGHT - BALL_RADIUS)
        sign_x = random.choice([-1, 1])
        sign_y = random.choice([-1, 1])
        self.dx = self.init_speed * sign_x
        self.dy = self.init_speed * sign_y
        self.rally_length = 0
    def move(self):
        self.x += self.dx
        self.y += self.dy
        if self.y <= 0 or self.y >= HEIGHT - BALL_RADIUS:
            self.dy *= -1
    def draw(self, win):
        pygame.draw.circle(win, WHITE, (int(self.x), int(self.y)), BALL_RADIUS)
    def check_collision(self, paddle1, paddle2):
        hit = False
        if self.dx < 0 and paddle1.x <= self.x + BALL_RADIUS <= paddle1.x + PADDLE_WIDTH:
            if paddle1.y < self.y < paddle1.y + paddle1.paddle_height:
                self.dx *= -1
                self.rally_length += 1
                hit = True
        elif self.dx > 0 and paddle2.x < self.x < paddle2.x + PADDLE_WIDTH:
            if paddle2.y - BOT_HIT_TOLERANCE < self.y < paddle2.y + paddle1.paddle_height + BOT_HIT_TOLERANCE:
                self.dx *= -1
                self.rally_length += 1
                hit = True
        return hit

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class Bot:
    def __init__(self):
        self.model = DQN(5, 3).to(device)
        self.target_model = DQN(5, 3).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=INIT_LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=20000)
        self.gamma = INIT_GAMMA
        self.epsilon = INIT_EPSILON
        self.epsilon_min = 0.05
        self.epsilon_decay = INIT_EPSILON_DECAY
        self.batch_size = 32
        self.action_changes = 0
        self.prev_action = None
        self.loss_window = deque(maxlen=100)
        self.reward_window = deque(maxlen=100)
        self.hit_count = 0
        self.total_hits = []
        self.train_step_counter = 0
        self.shaping_alpha = INIT_SHAPING_ALPHA
        self.update_time = INIT_UPDATE_TIME
        self.reward_consts = {
            "terminal_hit_reward": INIT_TERMINAL_HIT_REWARD,
            "terminal_miss_penalty": INIT_TERMINAL_MISS_PENALTY,
            "wait_center_penalty": INIT_WAIT_CENTER_PENALTY,
            "stabilize_reward": INIT_STABILIZE_REWARD,
            "reward_clip": INIT_REWARD_CLIP,
            "action_change_penalty": INIT_ACTION_CHANGE_PENALTY
        }
    def set_epsilon(self, eps, eps_decay):
        self.epsilon = eps
        self.epsilon_decay = eps_decay
    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    def get_state(self, ball, paddle_y):
        return torch.tensor([
            ball.x / WIDTH,
            ball.y / HEIGHT,
            (paddle_y + paddle1.paddle_height / 2) / HEIGHT,
            ((paddle_y + paddle1.paddle_height / 2) - ball.y) / HEIGHT,
            int(ball.dx > 0)
        ], dtype=torch.float32, device=device)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)
        q_values = self.model(states)
        next_q_online = self.model(next_states)
        next_q_target = self.target_model(next_states)
        next_actions = torch.argmax(next_q_online, dim=1)
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze().detach()
        target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.criterion(current_q, target_q)
        self.loss_window.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_step_counter += 1
        if self.train_step_counter % self.update_time == 0:
            self.update_target_network()
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

# ----------- Game Variables -----------
paddle1 = Paddle(20)
paddle2 = Paddle(WIDTH - 30)
ball = Ball()
bot = Bot()

# ---------- UI and Save/Load Feedback ----------
FEEDBACK_FONT = pygame.font.SysFont("Arial", 28, bold=True)
USER_FEEDBACK_MSG = ""
USER_FEEDBACK_TIMER = 0  # frames until feedback disappears

def show_feedback(msg, color=(255,255,0), duration=180):
    global USER_FEEDBACK_MSG, USER_FEEDBACK_TIMER
    USER_FEEDBACK_MSG = msg
    USER_FEEDBACK_TIMER = duration

def render_feedback():
    global USER_FEEDBACK_MSG, USER_FEEDBACK_TIMER
    if USER_FEEDBACK_TIMER > 0 and USER_FEEDBACK_MSG:
        text = FEEDBACK_FONT.render(USER_FEEDBACK_MSG, True, (255,255,0))
        win.blit(text, (WIDTH//2 - text.get_width()//2, 10))
        USER_FEEDBACK_TIMER -= 1

# ----- Constants -----
button_width, button_height = 200, 35
slider_width, slider_height = 350, 20
margin_top = 10
spacing = 20
start_x = (WIDTH - button_width) // 2
start_y = margin_top

# Top-right corner pause/settings button shown during gameplay
pause_btn = Button(WIDTH - button_width - 10, 10, button_width, button_height, "Settings")

# Pause menu buttons
buttons = [
    Button(start_x, start_y + i * (button_height + spacing), button_width, button_height, text)
    for i, text in enumerate(["Resume", "Save Model", "Load Model", "Exit Game"])
]

slider_labels = [
    "Ball Speed", "Paddle Speed", "Paddle Height",
    "ε‑Greedy ε₀", "ε Decay Rate", "γ (Discount Factor)",
    "Learning Rate (α)", "Shaping α", "Update time", ""
]

slider_ranges = [
    (1, 5, INIT_BALL_SPEED, 0.5),             
    (1, 5, INIT_PADDLE_SPEED, 0.5),           
    (60, 200, INIT_PADDLE_HEIGHT, 10),     
    (0.01, 1.0, INIT_EPSILON, 0.01),                   
    (0.95, 1.0, INIT_EPSILON_DECAY, 0.0001),              
    (0.90, 0.999, INIT_GAMMA, 0.001),               
    (0.0001, 0.001, INIT_LEARNING_RATE, 0.0001),          
    (1, 50, INIT_SHAPING_ALPHA, 1),
    (50, 2000, INIT_UPDATE_TIME, 50),
    (0,0,0,0)
]

sliders = []
slider_start_y = start_y + len(buttons) * (button_height + spacing) + 30

left_x = WIDTH // 4 - slider_width // 2     
right_x = 3 * WIDTH // 4 - slider_width // 2

for i, (label, (minv, maxv, startv, step)) in enumerate(zip(slider_labels, slider_ranges)):
    if i < 5:
        x = left_x
        y = slider_start_y + i * (slider_height + 35)
    else:
        x = right_x
        y = slider_start_y + (i - 5) * (slider_height + 35)
    if i == 9:
        buttons.append(Button(x + button_width // 4, y - button_height // 2, button_width, button_height, "Advanced Settings"))
    else:
        sliders.append(Slider(x, y, slider_width, minv, maxv, startv, label, step=step))

dropdown_width, dropdown_height = 200, 30
dropdown_y = HEIGHT - dropdown_height * 3
dropdown_x = (WIDTH - dropdown_width) // 2
dropdown = Dropdown(
    dropdown_x, dropdown_y, dropdown_width, dropdown_height,
    font=pygame.font.SysFont("Arial", 18),
    main_color=(200, 200, 200),
    hover_color=(240, 240, 240),
    options=["Perfect Bot", "Human"], starting_index=0
)

advanced_sliders = []
advanced_slider_labels = ["Terminal Hit Reward", "Terminal Miss Penalty", "Wait Center Penalty", "Stabilize Reward", "Reward Clip", "Action Change Penalty", ""]
advanced_slider_ranges = [
    (1, 10, INIT_TERMINAL_HIT_REWARD, 1),
    (-10, 0, INIT_TERMINAL_MISS_PENALTY, 1),
    (-0.5, 0, INIT_WAIT_CENTER_PENALTY, 0.01),
    (0, 0.5, INIT_STABILIZE_REWARD, 0.01),
    (1, 5, INIT_REWARD_CLIP, 0.5),
    (-0.01, 0, INIT_ACTION_CHANGE_PENALTY, 0.0001),
    (0,0,0,0)
]

back_btn = None
slider_start_y = 100
mid_x = WIDTH // 2 - slider_width // 2
for i, (label, (minv, maxv, startv, step)) in enumerate(zip(advanced_slider_labels, advanced_slider_ranges)):
    if i == 6:
        back_btn = Button(mid_x, slider_start_y + i * (slider_height + 35), button_width, button_height, "Back")
    else:
        x = mid_x
        y = slider_start_y + i * (slider_height + 35)
        advanced_sliders.append(Slider(x, y, slider_width, minv, maxv, startv, label, step=step))

def draw(paused=False, advanced=False):
    if paused and advanced:
        win.fill(BLACK)
        for slider in advanced_sliders:
            slider.draw(win)
        back_btn.draw(win)

    elif not paused:
        win.fill(BLACK)
        paddle1.draw(win)
        paddle2.draw(win)
        ball.draw(win)
        pause_btn.draw(win)
        for x, y in target_marks:
            pygame.draw.circle(win, (255, 0, 0), (x, int(y)), 6, 2)
        
        # --- Display Stats on Gameplay Screen ---
        stats_font = pygame.font.SysFont("Arial", 18)
        stats = []

        # Basic Stats
        stats.append(f"Episode: {episode}")
        stats.append(f"Hits (ep): {hits_in_episode}")
        stats.append(f"Hits (avg): {np.mean(bot.total_hits[-10:]):.2f}" if bot.total_hits else "Hits (avg): N/A")
        stats.append(f"Avg Reward: {np.mean(bot.reward_window):.3f}" if bot.reward_window else "Avg Reward: N/A")
        stats.append(f"Avg Loss: {np.mean(bot.loss_window):.5f}" if bot.loss_window else "Avg Loss: N/A")

        # Agent Stats
        stats.append(f"ε: {bot.epsilon:.4f}")
        stats.append(f"Last Action: {last_action_name}")
        stats.append(f"Action Changes: {bot.action_changes}")

        # Optional Deep Debug Info
        stats.append(f"Q-values: [{', '.join(f'{q:.2f}' for q in last_qvals)}]")

        if target_y is not None:
            stats.append(f"Target Y: {int(target_y)}")
        if curr_distance_to_target is not None:
            stats.append(f"Distance to Target: {curr_distance_to_target:.2f}")

        # Draw all stats on the screen
        for i, line in enumerate(stats):
            text_surface = stats_font.render(line, True, (255, 255, 0))
            win.blit(text_surface, (10, 10 + i * 22))

        render_feedback()
        pygame.display.update()

    else:
        win.fill(BLACK)
        for button in buttons:
            button.draw(win)
        for slider in sliders:
            slider.draw(win)
        dropdown.draw(win)
        render_feedback()
        pygame.display.update()

def simple_enemy_AI(paddle, ball):
    if paddle.y + paddle1.paddle_height / 2 < ball.y - 10:
        paddle.move(False, True)
    elif paddle.y + paddle1.paddle_height / 2 > ball.y + 10:
        paddle.move(True, False)
    else:
        paddle.move(False, False)

def calculate_target_y(ball):
    x, y, dx, dy = ball.x, ball.y, ball.dx, ball.dy
    while True:
        if dx < 0:
            return None
        x += dx
        y += dy
        if y <= 0 or y >= HEIGHT - BALL_RADIUS:
            dy *= -1
            y = min(max(y, 0), HEIGHT - BALL_RADIUS)
        if x >= paddle2.x:
            return y

def apply_settings_from_ui():
    ball_speed = sliders[0].value
    paddle_speed = sliders[1].value
    paddle1.paddle_height = sliders[2].value
    paddle2.paddle_height = sliders[2].value
    eps = sliders[3].value
    eps_decay = sliders[4].value
    bot.gamma = sliders[5].value
    lr = sliders[6].value
    bot.shaping_alpha = sliders[7].value
    bot.update_time = sliders[8].value
    ball.set_speed(ball_speed)
    paddle1.set_speed(paddle_speed)
    paddle2.set_speed(paddle_speed)
    bot.set_epsilon(eps, eps_decay)
    bot.set_learning_rate(lr)
    train_against_human[0] = False if dropdown.selected_index == 0 else True

def apply_advanced_settings_from_ui():
    bot.reward_consts["terminal_hit_reward"] = advanced_sliders[0].value
    bot.reward_consts["terminal_miss_penalty"] = advanced_sliders[1].value
    bot.reward_consts["wait_center_penalty"] = advanced_sliders[2].value
    bot.reward_consts["stabilize_reward"] = advanced_sliders[3].value
    bot.reward_consts["reward_clip"] = advanced_sliders[4].value
    bot.reward_consts["action_change_penalty"] = advanced_sliders[5].value

def save_model_dialog():
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch Models", "*.pth")], title="Save Model")
        root.destroy()
        if file_path:
            torch.save({
                'model_state_dict': bot.model.state_dict(),
                'target_state_dict': bot.target_model.state_dict(),
                'optimizer_state_dict': bot.optimizer.state_dict(),
                'episode': episode,
                'reward_window': list(bot.reward_window),
                'loss_window': list(bot.loss_window),
                'total_hits': list(bot.total_hits),
                'hits_in_episode': hits_in_episode,
            }, file_path)
            show_feedback(f"Model & stats saved to {file_path.split('/')[-1]}", (90,220,90))
    except (Exception, TclError) as e:
        show_feedback(f"Save failed: {e}", (255,120,90), 240)

def load_model_dialog():
    global episode, hits_in_episode
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Models", "*.pth")], title="Load Model")
        root.destroy()
        if file_path:
            data = torch.load(file_path, map_location=device)
            bot.model.load_state_dict(data['model_state_dict'])
            bot.target_model.load_state_dict(data.get('target_state_dict', data['model_state_dict']))
            bot.optimizer.load_state_dict(data['optimizer_state_dict'])
            # Restore statistics/metrics if present
            episode = data.get('episode', episode)
            hits_in_episode = data.get('hits_in_episode', 0)
            # restore deques with correct maxlen
            bot.reward_window = deque(data.get('reward_window', []), maxlen=100)
            bot.loss_window = deque(data.get('loss_window', []), maxlen=100)
            bot.total_hits = list(data.get('total_hits', []))
            show_feedback(f"Loaded {file_path.split('/')[-1]}", (90,220,90))
    except (Exception, TclError) as e:
        show_feedback(f"Load failed: {e}", (255,120,90), 240)

def render_graph(data, label, color):
    fig, ax = plt.subplots(figsize=(3, 1.5), dpi=100)
    ax.plot(data, color=color)
    ax.set_title(label)
    ax.set_facecolor("lightgray")
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    raw_data = canvas.get_renderer().buffer_rgba()
    surf = pygame.image.frombuffer(raw_data, canvas.get_width_height(), "RGBA")
    plt.close(fig)
    return surf

def draw_stats_overlay(win):
        if _cached_reward_surf and _cached_loss_surf:
            win.blit(_cached_reward_surf, (WIDTH // 2 - GRAPH_WIDTH // 2, 20))
            win.blit(_cached_loss_surf,   (WIDTH // 2 - GRAPH_WIDTH // 2, 40 + GRAPH_HEIGHT))
        mode_text = "Step Mode: ON" if step_mode else "Step Mode: OFF"
        win.blit(font.render(mode_text, True, WHITE), (20, HEIGHT - 40))
        if step_mode:
            win.blit(font.render("Press SPACE to step", True, WHITE), (20, HEIGHT - 20))

def handle_step_mode(events):
    global step_pending
    if not step_mode:
        return True
    for event in events:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            step_pending = False
            return True
    return not step_pending

def toggle_step_mode():
    global step_mode, step_pending
    step_mode = not step_mode
    step_pending = step_mode

# ----------- Gameplay Variables -----------
frame_step = 0
running = True
paused = True

episode = 0
episode_reward = 0
hits_in_episode = 0

target_y = None
tracking_to_target = False
prev_bot_center = None
prev_distance_to_target = None
target_marks = []

last_shaping_reward = 0.0
curr_distance_to_target = None
terminal_reward = 0.0
wait_penalty = 0.0
last_action_name = "?"
last_qvals = [0.0, 0.0, 0.0]
qvals = [0.0, 0.0, 0.0]

reward_history = []
loss_history = []
step_mode = False
step_pending = False
advanced = False

GRAPH_WIDTH, GRAPH_HEIGHT = 300, 150

font = pygame.font.SysFont("Arial", 18)

if ball.dx > 0:
    target_y = calculate_target_y(ball)
    tracking_to_target = True
    prev_bot_center = paddle2.y + paddle1.paddle_height / 2
    prev_distance_to_target = abs(prev_bot_center - target_y) if target_y is not None else None
    if target_y is not None:
        target_marks.append((paddle2.x, int(target_y)))

while running:
    clock.tick(FPS)
    frame_step += 1
    request_pause = False
    request_unpause = False

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                toggle_step_mode()
            elif event.key == pygame.K_SPACE and step_mode:
                step_pending = False
        if pause_btn.handle_event(event):
            if not paused:
                sliders[3].value = bot.epsilon
                request_pause = True
        if advanced and back_btn.handle_event(event):
            apply_advanced_settings_from_ui()
            advanced = False
            break
        if paused:
            for btn in buttons:
                if btn.handle_event(event):
                    if btn.text == "Resume":
                        apply_settings_from_ui()
                        request_unpause = True
                        break
                    elif btn.text == "Save Model":
                        save_model_dialog()
                        break
                    elif btn.text == "Load Model":
                        load_model_dialog()
                        break
                    elif btn.text == "Exit Game":
                        running = False
                    elif btn.text == "Advanced Settings":
                        advanced = not advanced
                        break
            if request_unpause:
                break
            for slider in sliders:
                slider.handle_event(event)
            if advanced:
                for slider in advanced_sliders:
                    slider.handle_event(event)
            dropdown.handle_event(event)

    if request_unpause:
        paused = False
        continue

    if request_pause:
        paused = True

    if paused:
        draw(paused, advanced)
        pygame.display.update()
        continue

    if step_mode:
        if not handle_step_mode(events):
            continue

    # --------- MAIN GAME ----------
    keys = pygame.key.get_pressed()
    up1 = keys[pygame.K_w]
    down1 = keys[pygame.K_s]
    if train_against_human[0]:
        paddle1.move(up1, down1)
    else:
        simple_enemy_AI(paddle1, ball)

    state = bot.get_state(ball, paddle2.y)
    distance_x = abs(ball.x - paddle2.x)
    effective_epsilon = bot.epsilon
    if ball.dx > 0 and distance_x < 40:
        effective_epsilon = 0.0
    if random.random() < effective_epsilon:
        action = random.randint(0, 2)
    else:
        with torch.no_grad():
            q_values = bot.model(state)
            action = torch.argmax(q_values).item()
            qvals = [float(x) for x in q_values.cpu().numpy()]
    last_action_name = ACTION_NAMES.get(action, str(action))
    last_qvals = qvals

    action_change_reward = 0
    if bot.prev_action is not None and action != bot.prev_action:
        bot.action_changes += 1
        action_change_reward = bot.reward_consts["action_change_penalty"]
    bot.prev_action = action
    if action == 0:
        paddle2.move(True, False)
    elif action == 1:
        paddle2.move(False, True)
    else:
        paddle2.move(False, False)

    ball.move()
    hit = ball.check_collision(paddle1, paddle2)

    anticipation_start = False
    if (target_y is None) and ball.dx > 0:
        anticipation_start = True
    elif ball.dx > 0 and ((ball.x <= WIDTH // 2 and ball.x + ball.dx > WIDTH // 2)):
        anticipation_start = True
    elif hit and ball.dx > 0:
        anticipation_start = True

    if anticipation_start:
        target_y = calculate_target_y(ball)
        tracking_to_target = True
        prev_bot_center = paddle2.y + paddle1.paddle_height / 2
        prev_distance_to_target = abs(prev_bot_center - target_y) if target_y is not None else None
        if target_y is not None:
            target_marks.append((paddle2.x, int(target_y)))

    if (hit and ball.dx < 0) or ball.x > WIDTH:
        tracking_to_target = False
        target_y = None
        prev_bot_center = None
        prev_distance_to_target = None
        curr_distance_to_target = None

    # ---- Reward logic ----
    reward = 0.0
    done = False
    last_shaping_reward = 0.0
    curr_distance_to_target = None
    terminal_reward = 0.0
    wait_penalty = 0.0

    reward += action_change_reward

    if tracking_to_target and target_y is not None and prev_distance_to_target is not None:
        bot_center = paddle2.y + paddle1.paddle_height / 2
        curr_distance_to_target = abs(bot_center - target_y)
        shaping_delta = prev_distance_to_target - curr_distance_to_target
        reward += bot.shaping_alpha * shaping_delta / HEIGHT
        last_shaping_reward = bot.shaping_alpha * shaping_delta / HEIGHT
        prev_distance_to_target = curr_distance_to_target
        prev_bot_center = bot_center
        if abs(bot_center - target_y) < STABILIZE_THRESHOLD and action == 2:
            reward += bot.reward_consts["stabilize_reward"]

    elif ball.dx < 0:
        paddle_center = paddle2.y + paddle1.paddle_height / 2
        center_dist = abs(paddle_center - HEIGHT / 2) / (HEIGHT / 2)
        wait_penalty = bot.reward_consts["wait_center_penalty"] * center_dist
        reward += wait_penalty

    if ball.x > WIDTH:
        reward += bot.reward_consts["terminal_miss_penalty"]
        terminal_reward = bot.reward_consts["terminal_miss_penalty"]
        done = True
    elif hit and ball.dx < 0:
        reward += bot.reward_consts["terminal_hit_reward"]
        terminal_reward = bot.reward_consts["terminal_hit_reward"]
        hits_in_episode += 1
    elif ball.x < 0:
        reward += bot.reward_consts["terminal_hit_reward"]
        terminal_reward = bot.reward_consts["terminal_hit_reward"]
        done = True

    clipped_reward = float(np.clip(reward, -bot.reward_consts["reward_clip"], bot.reward_consts["reward_clip"]))
    episode_reward += clipped_reward

    next_state = bot.get_state(ball, paddle2.y)
    bot.remember(state, action, clipped_reward, next_state, done)
    bot.train()

    # Track stats
    if bot.loss_window:
        loss_history.append(bot.loss_window[-1])
    reward_history.append(clipped_reward)

    if done:
        episode += 1
        bot.reward_window.append(episode_reward)
        bot.total_hits.append(hits_in_episode)
        episode_reward = 0
        ball.reset()
        paddle1.y = HEIGHT // 2 - paddle1.paddle_height // 2
        paddle2.y = HEIGHT // 2 - paddle1.paddle_height // 2
        bot.action_changes = 0
        bot.prev_action = None
        hits_in_episode = 0
        prev_bot_center = None
        prev_distance_to_target = None
        target_marks.clear()
        if ball.dx > 0:
            target_y = calculate_target_y(ball)
            tracking_to_target = True
            prev_bot_center = paddle2.y + paddle1.paddle_height / 2
            prev_distance_to_target = abs(prev_bot_center - target_y) if target_y is not None else None
            if target_y is not None:
                target_marks.append((paddle2.x, int(target_y)))
        else:
            target_y = None
            tracking_to_target = False

        if episode % 10 == 0:
            mean_r = np.mean(bot.reward_window)
            mean_l = np.mean(bot.loss_window)
            mean_h = np.mean(bot.total_hits[-10:]) if len(bot.total_hits) >= 10 else 0
            print(f"Ep {episode}, eps={bot.epsilon:.3f}, avg_reward={mean_r:.3f}, avg_loss={mean_l:.3f}, hits/ep(last10)={mean_h:.2f}")

    mean_r = np.mean(bot.reward_window) if bot.reward_window else 0
    mean_l = np.mean(bot.loss_window) if bot.loss_window else 0
    mean_h = np.mean(bot.total_hits[-10:]) if len(bot.total_hits) >= 10 else 0

    if step_mode and not paused:
        if frame_step - _last_graph_update >= GRAPH_UPDATE_INTERVAL:
            _cached_reward_surf = render_graph(reward_history[-100:], "Rewards", "green")
            _cached_loss_surf   = render_graph(loss_history[-100:],   "Losses",  "red")
            _last_graph_update = frame_step
        draw(paused, advanced)
        draw_stats_overlay(win)
    else:
        draw(paused, advanced)
    pygame.display.update()

    if step_mode:
        step_pending = True

pygame.quit()
sys.exit()