# Frozen Lake: Q-Learning 

A complete Reinforcement Learning solution for the Frozen Lake problem, built entirely from scratch in Python. 
---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Environment Design](#2-environment-design)
3. [Q-Learning Algorithm](#3-q-learning-algorithm)
4. [Training Procedure](#4-training-procedure)
5. [Results](#5-results)
6. [Execution Instructions](#7-execution-instructions)

---

## 1. Introduction

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent is not told what to do it discovers what works through trial and error.

After every action the agent receives a **reward** signal. Over time it learns to take actions that maximise the total reward it collects. The key idea is that the agent has no prior knowledge of the environment. It only learns from experience.

The three core components are:
- **Agent** — the learner and decision maker
- **Environment** — the world the agent interacts with
- **Reward** — the feedback signal the agent uses to learn

A single interaction looks like this:

```
Agent observes state S
Agent takes action A
Environment returns reward R and new state S'
Agent learns from (S, A, R, S')
```

This cycle repeats thousands of times until the agent learns an effective policy.

### What is Frozen Lake?

Frozen Lake is a grid-world navigation problem. An agent starts at position S on an 8×8 frozen grid and must reach the goal G without falling into any holes H. The surface is frozen, meaning the agent has no grip — it must carefully navigate a path around the holes.

```
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F H F F F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

The agent has no map and no knowledge of where the holes are. It learns purely by falling in holes and occasionally reaching the goal — building up knowledge one episode at a time.

---

## 2. Environment Design

### State Representation

The grid contains 64 cells arranged in 8 rows and 8 columns. Each cell is a **state**. States are represented as single integers from 0 to 63, numbered left to right, top to bottom:

```
col:  0   1   2   3   4   5   6   7
     ┌───┬───┬───┬───┬───┬───┬───┬───┐
row 0│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
row 1│ 8 │ 9 │10 │11 │12 │13 │14 │15 │
...
row 7│56 │57 │58 │59 │60 │61 │62 │63 │
     └───┴───┴───┴───┴───┴───┴───┴───┘
```

The conversion between integer state and grid coordinates is:

```
state → (row, col):    row = state // 8,   col = state % 8
(row, col) → state:    state = row * 8 + col
```

Integer states are used throughout because the Q-table lookup is simpler with a single integer key. The conversion to `(row, col)` only happens temporarily inside `step()` for boundary-safe movement.

- **Start state S** = state 0 (row 0, col 0)
- **Goal state G** = state 63 (row 7, col 7)

### Action Representation

The agent has 4 possible actions at every state:

| Number | Action | Effect |
|--------|--------|--------|
| 0 | Left | col − 1 |
| 1 | Down | row + 1 |
| 2 | Right | col + 1 |
| 3 | Up | row − 1 |

Movement is bounded  if the agent tries to move off the grid edge it stays in place. For example, attempting to go Right from column 7 results in no movement.

### Reward Structure

| Event | Reward | Done |
|-------|--------|------|
| Reach Goal (G) | +1.0 | True |
| Fall in Hole (H) | 0.0 | True |
| Any other step | 0.0 | False |

Reward only arrives at the goal. This makes the problem sparse — the agent must navigate the entire grid correctly before receiving any positive signal. All learning comes from occasionally reaching the goal and having that +1 reward ripple backwards through the Q-table over thousands of episodes.

---

## 3. Q-Learning Algorithm

### Description of Q-Learning

Q-Learning is a model-free, off-policy RL algorithm. It maintains a **Q-table** — a 64×4 matrix where each cell `Q(s, a)` holds a score answering: *"how good is it to take action a from state s?"*

The Q-table starts at all zeros because the agent knows nothing. After every single step it updates one cell based on what just happened. Over thousands of episodes the values converge to reflect the true long-term value of each state-action pair.

The agent's **policy** is implicit in the Q-table — at any state, simply pick the action with the highest Q-value.

### Explanation of the Update Equation

After every step the following update is applied:

```
Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') − Q(s, a) ]
```

| Symbol | Name | Meaning |
|--------|------|---------|
| `Q(s, a)` | Current estimate | Score for this state-action pair before update |
| `α` | Learning rate | How much to shift the estimate each update (0–1) |
| `r` | Reward | Immediate reward received after taking action a |
| `γ` | Discount factor | How much future rewards are worth relative to now |
| `max Q(s', a')` | Best future value | Highest Q-value available in the next state |
| `r + γ max Q(s', a')` | Target | What the Q-value should be, given what we now know |
| `target − Q(s, a)` | Error | The gap between target and current estimate |

The equation says: * push the current estimate a little bit towards the target*. If the agent did better than expected, the Q-value goes up. If it did worse, it goes down.

When the episode ends in a terminal state (hole or goal), there is no next state, so the target simplifies to just the reward:

```
done=True:   target = r
done=False:  target = r + γ × max Q(s', a')
```

### Exploration Strategy

**exploration-exploitation tradeoff**. If the agent always picks the highest Q-value action, it never discovers better routes it hasn't tried yet. But if it always acts randomly, it never uses what it has learned.

The solution used here is **epsilon-greedy**:

```
With probability ε    → take a random action  (explore)
With probability 1−ε  → take the best action  (exploit)
```

Epsilon starts at 1.0 (fully random) and decays after each episode using **multiplicative decay**:

```
ε ← max(ε_min,  ε × (1 − decay_rate))
```

Multiplicative decay was chosen over additive decay after experimentation showed that additive decay reduced ε to its minimum after only ~990 episodes — far too early for this maze where random exploration has only a 0.49% chance of finding the goal per episode.

---

## 4. Training Procedure

### Training Loop

Each episode follows this structure:

```
1. Reset the environment → agent returns to state 0
2. For each step (up to 1000):
      a. Choose action via epsilon-greedy
      b. Execute action → environment returns (S', R, done)
      c. Update Q-table using the update equation
      d. Move to S'
      e. If done: end episode
3. Decay epsilon
4. Record reward, success, epsilon
```

The Q-table is updated after every individual step — not at the end of the episode. This means a 15-step episode produces 15 Q-table updates.

### Hyperparameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Episodes | 50,000 | Needed due to maze difficulty — random walks only find the goal 0.49% of the time |
| Max steps per episode | 1,000 | Gives random exploration enough time to find the goal |
| Learning rate α | 0.8 | Aggressive but stable — converges quickly on this deterministic problem |
| Discount factor γ | 0.95 | Values rewards ~20 steps into the future (1/(1−0.95) = 20) |
| Epsilon start | 1.0 | Fully random at the start — knows nothing |
| Epsilon minimum | 0.01 | Always retains 1% exploration |
| Epsilon decay | 0.0001 (multiplicative) | Reaches minimum at ~46,000 episodes — ensures long exploration |

---

## 5. Results

### Final Success Rate

| Metric | Value |
|--------|-------|
| Training success rate (last 1,000 episodes) | 97.5% |
| Evaluation success rate (100 episodes, ε=0) | 100% |
| Average reward (evaluation) | 1.000 |
| Successful runs | 100 / 100 |
| Failures | 0 / 100 |

### Learned Policy

The policy extracted from the Q-table after training:

```
 ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
 →  →  →  →  ↓  ↓  ↓  ↓
 →  →  ↑  H  ↓  ↓  ↓  ↓
 →  →  ↑  H  ↓  ↓  ↓  ↓
 →  →  ↑  H  ↓  ↓  →  ↓
 ↑  H  H  →  →  ↓  H  ↓
 ↑  H  →  ↑  H  ↓  H  ↓
 ↑  ←  ←  H  →  →  →  G
```

### Discussion of Performance

The agent learned a complete and optimal policy with a 100% success rate on evaluation. Several observations:

**Hole avoidance:** The column of holes at col 3 (rows 2–4) is navigated by the agent going Up at col 2 reversing course rather than walking into the hole. This was not programmed; it emerged from the reward signal alone.

**Convergence speed:** The agent crossed 50% success rate at episode ~12,800 and 95% at episode ~35,860. This gradual improvement reflects the reward rippling backwards from the goal state-by-state through the Q-table over thousands of episodes.

**Hyperparameter sensitivity:** The choice of epsilon decay strategy had a dramatic effect. Additive decay with `ε_decay=0.001` produced 0% success because the agent stopped exploring before building any meaningful Q-table signal. Multiplicative decay with `ε_decay=0.0001` gave the agent enough time to find the goal repeatedly and learn from it.

**Sparse reward challenge:** Because reward is only given at the goal, the agent must complete the entire path correctly before receiving any signal. This is why 50,000 episodes were needed — the learning signal is weak early on and takes many repetitions to propagate.

---


---

##  Execution Instructions

### Requirements

- Python 3.8 or higher
- numpy >= 1.21.0

### Installation

```bash
# Clone the repository
git clone <your-repo-url>

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

Run the scripts in order:

**Step 1 — Train the agent**
```bash
python train.py
```
Trains for 50,000 episodes. Prints progress every 1,000 episodes. Saves the Q-table and training stats to `results/`.

**Step 2 — Evaluate and display the policy**
```bash
python evaluate.py
```
Loads the trained Q-table. Prints the learned policy as an arrow grid, shows all Q-values, and runs 100 evaluation episodes with no exploration.

**Step 3 — Generate training plots**
```bash
python plot_results.py
```
Loads the training stats and saves `results/training_curves.png` — three plots showing success rate, average reward, and epsilon decay over training.
