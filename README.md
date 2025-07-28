# 🏓 PongRL: Interactive Reinforcement Learning Trainer for Pong

### _Learn Reinforcement Learning by Playing!_

---

## 🌟 Motivation

Reinforcement Learning (RL) is one of the most powerful paradigms in modern AI, enabling agents to learn by interacting with their environment. However, understanding *how* it works — why agents make decisions, how hyperparameters affect learning, and what makes training unstable — can often feel like a black box.

**PongRL** was built to address this gap by allowing users to **observe, play, and modify** every aspect of a reinforcement learning agent, in real-time. Whether you're a student, educator, or curious enthusiast, PongRL helps you explore RL by turning learning into a game itself.

---

## 🎮 What is PongRL?

PongRL is a complete Pong environment powered by **Deep Q-Learning**, where:

- A trainable RL **Bot** learns to play Pong.
- The bot trains either against a **Perfect AI** or a **Human Player**.
- **Full control** is given to the user to tweak game mechanics, model behavior, and learning dynamics.

It's more than a game — it's a **sandbox** to **visualize**, **experiment**, and **interact** with the RL training loop, step-by-step.

---

## ✨ Key Features

### 🧠 1. Interactive RL Agent Training
- Agent is initialized with random weights.
- Trains live against either:
  - A **Perfect Bot** (always returns the ball correctly)
  - A **Human Player** via keyboard
- Training is observable frame-by-frame in **Step Mode**.

### 🎛 2. Fully Adjustable Parameters (Live UI!)
Adjust settings **before or during** training via an intuitive Pygame UI:
- **Game Mechanics:**
  - Ball Speed
  - Paddle Speed
  - Paddle Size
- **RL Hyperparameters:**
  - `ε` (Exploration rate), `γ` (Discount factor), Learning Rate
  - `ε` Decay Rate, Replay Memory Size

### ⏸ 3. Pause / Resume / Step Modes
- **Normal Mode**: Watch training unfold in real time.
- **Step Mode**: Manually step through each frame — perfect for debugging rewards and decisions.
- Switch modes instantly with on-screen buttons.

### 📈 4. Real-time Training Stats
During training, monitor:
- **Reward Graph**: Cumulative rewards per episode.
- **Loss Graph**: Q-value prediction loss over time.
- Live feedback on current reward, action taken, Q-values, and model status.

### 💾 5. Save & Load Agent Snapshots
At any point:
- Save current model weights + training stats.
- Load from saved snapshots to resume training or testing.

### 🧪 6. Perfect AI Testing Mode
Once you're happy with training:
- Enter **Testing Mode** to play the trained bot against the **Perfect AI**.
- Epsilon is set to 0 — the bot now acts purely on its learned policy.

---

## 🖥️ Screenshots

| Settings UI | Step Mode + Graphs | Training Stats |
|-------------|--------------------|----------------|
| *(Insert screenshots here if hosted)* |

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.10+
- `pygame`
- `matplotlib`
- `torch`

Install dependencies:

```bash
pip install -r requirements.txt