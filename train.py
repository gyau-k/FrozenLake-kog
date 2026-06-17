# train.py — Training Loop
# This is where the agent explores the environment 
# The agent plays thousands of episodes, updating the Q-table after every step.

import numpy as np
import pickle
import os

from environment import FrozenLakeEnv
from agent import QLearningAgent

#  Hyperparameters 
#chnage these to see how they affect learning
EPISODES       = 50000   # total number of games to play
MAX_STEPS      = 1000    # max steps per episode (longer to give random walks a chance)
ALPHA          = 0.8     # learning rate
GAMMA          = 0.95    # discount factor
EPSILON        = 1.0     # starting exploration rate
EPSILON_MIN    = 0.01    # lowest epsilon will decay to
EPSILON_DECAY  = 0.0001  
                        

# Where to save results
RESULTS_DIR    = "results"


def train():
    #setup 
    env   = FrozenLakeEnv()
    agent = QLearningAgent(
        alpha         = ALPHA,
        gamma         = GAMMA,
        epsilon       = EPSILON,
        epsilon_min   = EPSILON_MIN,
        epsilon_decay = EPSILON_DECAY
    )

    #  Stats trackers 
    rewards_per_episode  = []   # reward at end of each episode (0 or 1)
    success_per_episode  = []   # True/False for each episode
    epsilon_per_episode  = []   # epsilon value at start of each episode

    print(f"Training for {EPISODES} episodes...\n")


    # Main training loop
    for episode in range(EPISODES):

        state = env.reset() #put agent back at start
        total_reward = 0
        done = False

        #  Record epsilon before this episode starts 
        epsilon_per_episode.append(agent.epsilon)

    
        # Step loop — one episode
        for step in range(MAX_STEPS):

            # agent decides what to do
            action = agent.choose_action(state)

            #Environment responds
            next_state, reward, done, info = env.step(action)

            #agent learns from what just happened
            agent.update(state, action, reward, next_state, done)

            #Move to the next state
            state = next_state
            total_reward += reward

            #End episode if terminal state reached
            if done:
                break

        #after each episode 
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        success_per_episode.append(total_reward > 0)

        #Print progress every 1000 episodes 
        if (episode + 1) % 1000 == 0:
            recent       = success_per_episode[-1000:]
            success_rate = sum(recent) / len(recent) * 100
            print(f"  Episode {episode+1:>6} | "
                  f"Success rate (last 1000): {success_rate:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")


    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save the trained Q-table so evaluate.py can load it
    np.save(f"{RESULTS_DIR}/q_table.npy", agent.q_table)

    # Save stats for plotting
    stats = {
        "rewards"  : rewards_per_episode,
        "successes": success_per_episode,
        "epsilons" : epsilon_per_episode,
        "episodes" : EPISODES
    }
    with open(f"{RESULTS_DIR}/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    #  Final summary 
    total_successes = sum(success_per_episode)
    overall_rate    = total_successes / EPISODES * 100
    last_1000_rate  = sum(success_per_episode[-1000:]) / 1000 * 100

    print(f"\n{'='*55}")
    print(f"  Training complete")
    print(f"{'='*55}")
    print(f"  Total episodes   : {EPISODES}")
    print(f"  Total successes  : {total_successes}")
    print(f"  Overall rate     : {overall_rate:.1f}%")
    print(f"  Last 1000 rate   : {last_1000_rate:.1f}%")
    print(f"  Final epsilon    : {agent.epsilon:.4f}")
    print(f"  Q-table saved to : {RESULTS_DIR}/q_table.npy")
    print(f"{'='*55}\n")

    return agent, stats


if __name__ == "__main__":
    train()
