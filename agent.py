# agent.py — The Q-Learning Agent
# The agent is the learner. It holds the Q-table which is the best course of action and updates its qtable after every step.


import numpy as np
import random

from environment import NUM_STATES, NUM_ACTIONS


class QLearningAgent:
    """
    A Q-Learning agent that learns to navigate the Frozen Lake.

    The Q-table is a 64 x 4 array.
    Rows = states (0-63), Columns = actions (0=Left, 1=Down, 2=Right, 3=Up)
    Each cell holds a score: "how good is it to take this action from this state?"
    """

    def __init__(
        self,
        alpha=0.8,       # learning rate  — how much to update Q-values each step
        gamma=0.95,      # discount factor — how much to value future rewards
        epsilon=1.0,     # starting exploration rate (1.0 = fully random)
        epsilon_min=0.01,# lowest epsilon will ever decay to
        epsilon_decay=0.001  # how much to subtract from epsilon each episode
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # The Q-table — starts at all zeros because the agent knows nothing
        # Shape: (64, 4) — one row per state, one column per action
        self.q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
        
        
        
    # Action selection
    def choose_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon  - pick a RANDOM action (explore)
        With probability 1-epsilon - pick the BEST known action (exploit)

        Early in training epsilon is high so the agent explores widely.
        As epsilon decays, the agent increasingly trusts what it has learned.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore — random action
            return random.randint(0, NUM_ACTIONS - 1)
        else:
            # Exploit — pick action with highest Q-value for this state
            return np.argmax(self.q_table[state])

    # Learning
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-learning equation:

            Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]

        Parameters
        ----------
        state      : int   — where the agent was
        action     : int   — what it did
        reward     : float — what it got
        next_state : int   — where it ended up
        done       : bool  — whether the episode ended
        """
        current_q = self.q_table[state, action]

        if done:
            # Episode is over — there is no next state to consider
            # The target is just the reward received
            target = reward
        else:
            # The target is: reward + discounted best future Q-value
            best_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * best_next_q

        # The error is the gap between target and current estimate
        error = target - current_q

        # push the Q-value a little bit towards the target
        self.q_table[state, action] = current_q + self.alpha * error

    def decay_epsilon(self):
        """
        Reduce epsilon after each episode.
        Ensures it never drops below epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)


    # Policy extraction
    def get_best_action(self, state):
        """
        Return the best action for a state according to the learned Q-table.
        Used after training — no exploration, pure exploitation.
        """
        return np.argmax(self.q_table[state])
