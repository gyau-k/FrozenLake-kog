# evaluate.py — Policy Display and Evaluation
# Loads the trained Q-table the agent.py did, displays the learned policy as a grid,
# then runs 100 clean episodes to measure real performance.

import numpy as np

from environment import FrozenLakeEnv, MAP, GRID_SIZE, NUM_ACTIONS
from agent import QLearningAgent

# Symbols used when printing the policy grid
ACTION_SYMBOLS = {0: "←", 1: "↓", 2: "→", 3: "↑"}
EVAL_EPISODES  = 100
RESULTS_DIR    = "results"


# Part D — Policy Extraction and Display

def extract_policy(q_table):
    """
    For every state, pick the action with the highest Q-value.
    That is the learned policy.

    Returns a list of 64 integers — one best action per state.
    """
    return [np.argmax(q_table[s]) for s in range(GRID_SIZE * GRID_SIZE)]


def display_policy(policy):
    """
    Print the policy as an 8x8 grid of arrows.

    Terminal states (H, G) are shown as-is — no action needed there.
    """
    print("\n" + "=" * 40)
    print("  Learned Policy")
    print("=" * 40)
    print("  (← Left  ↓ Down  → Right  ↑ Up)\n")

    for row in range(GRID_SIZE):
        row_str = "  "
        for col in range(GRID_SIZE):
            state = row * GRID_SIZE + col
            cell  = MAP[row][col]

            if cell == 'H':
                row_str += " H "    # hole — no action
            elif cell == 'G':
                row_str += " G "    # goal — no action
            else:
                row_str += f" {ACTION_SYMBOLS[policy[state]]} "

        print(row_str)
    print()


def display_q_values(q_table):
    """
    Print the raw Q-values for every state useful for seeing
    how confident the agent is about each direction.
    """
    print("=" * 40)
    print("  Q-Table (best action per state)")
    print("=" * 40)
    print(f"  {'State':>5}  {'Cell':>4}  {'←':>6}  {'↓':>6}  {'→':>6}  {'↑':>6}  {'Best':>6}")
    print("  " + "-" * 50)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            state = row * GRID_SIZE + col
            cell  = MAP[row][col]
            q     = q_table[state]
            best  = ACTION_SYMBOLS[np.argmax(q)]
            print(f"  {state:>5}  {cell:>4}  "
                  f"{q[0]:>6.3f}  {q[1]:>6.3f}  {q[2]:>6.3f}  {q[3]:>6.3f}  {best:>6}")
    print()



# Part E — Evaluation
def evaluate(agent, episodes=EVAL_EPISODES):
    """
    Run the agent for a fixed number of episodes with NO exploration.
    Epsilon is set to 0 — the agent always picks its best known action.

    Reports:
      - Success rate
      - Average reward
      - Number of successes
      - Number of failures
    """
    env = FrozenLakeEnv()

    # Turn off exploration — pure exploitation of learned policy
    agent.epsilon = 0.0

    successes = 0
    failures  = 0
    rewards   = []

    for episode in range(episodes):
        state = env.reset()
        done  = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)           # always picks best
            state, reward, done, info = env.step(action)
            total_reward += reward

            # Safety cap — shouldn't loop forever with a trained agent
            # but protects against edge cases
            if total_reward > 1:
                break

        rewards.append(total_reward)
        if total_reward > 0:
            successes += 1
        else:
            failures += 1

    success_rate   = successes / episodes * 100
    average_reward = sum(rewards) / episodes

    print("=" * 40)
    print(f"  Evaluation over {episodes} episodes")
    print("=" * 40)
    print(f"  Success Rate    : {success_rate:.1f}%")
    print(f"  Average Reward  : {average_reward:.3f}")
    print(f"  Successes       : {successes}")
    print(f"  Failures        : {failures}")
    print("=" * 40 + "\n")

    return {
        "success_rate"  : success_rate,
        "average_reward": average_reward,
        "successes"     : successes,
        "failures"      : failures
    }


# Main
if __name__ == "__main__":

    #Load the trained Q-table 
    print("\nLoading trained Q-table...")
    q_table = np.load(f"{RESULTS_DIR}/q_table.npy")
    print(f"Q-table shape: {q_table.shape}\n")

    # Rebuild agent with trained Q-table 
    agent = QLearningAgent()
    agent.q_table = q_table

    #show the policy 
    policy = extract_policy(q_table)
    display_policy(policy)

    #show Q-values 
    display_q_values(q_table)

    #evaluate 
    evaluate(agent, episodes=EVAL_EPISODES)
