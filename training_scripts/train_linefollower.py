#!/usr/bin/env python3
import random
import gym_gazebo
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.8
    epsilon_decay = 0.99

    q_table = np.zeros((observation_space.n, action_space.n))

    highest_reward = 0

    try:
        state, _ = env.reset()
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print("New highest reward: {total_reward}.")

                saveQ(q_table, "best_q_table.npy") # Will save the new best Q-Table
                print("Q-table saved!")

            env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table):
    alpha = 0.1
    gamma = 0.9

    current_q = q_table[state, action]
    max_future_q = np.max(q_table[next_state]) # Robot reached next state after taking the action, so look for the highest Q value in that new state (the row)

    q_table[state, action] = current_q + alpha * (reward + gamma * max_future_q - current_q) # Bellman equation to renew the Q value using the new state to estimate future rewards

    return q_table


def choose_action(state, q_table, epsilon, action_space):
    if random.random() > epsilon: # Choose a random number so that it will explore sometimes and exploit other times. This is the code for exploiting
        
        max_q = np.max(q_table[state]) # Chooses highest Q value

        best_actions = np.where(q_table[state] == max_q)[0] # Finds where the max Q value(s) is in the Q matrix and returns the column indice(s) in an array, indicating which action to take

        action = random.choice(best_actions.tolist()) # Choose a random index from the array

    else:
        action = action_space.sample() # Choose a random action, either 0,1,2
    
    return action


def saveQ(q_table, save_path):

    np.save(save_path, q_table)

    return

def loadQ(load_path):

    np.load(load_path)

    return

if __name__ == "__main__":
    main()