import pickle
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def run(config, is_train=True, render=True):
    render_mode = "human" if render else None
    env = gym.make(
        "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode=render_mode
    )

    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    epsilon_decay = config["epsilon_decay"]
    min_epsilon = config["min_epsilon"]
    num_episodes = config["num_episodes"]

    # init Q-table
    if is_train:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open("q_table_4x4.pickle", "rb") as f:
            q_table = pickle.load(f)

    rewards_per_episode = np.zeros(num_episodes)
    for episode_idx in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if is_train and np.random.rand() < epsilon:
                action = env.action_space.sample()  # exploration
            else:
                action = np.argmax(q_table[state, :])  # exploitation

            next_state, reward, terminated, truncated, _ = env.step(action)
            # print(reward)

            # update
            if is_train:
                q_table[state, action] += alpha * (
                    reward
                    + gamma * np.max(q_table[next_state, :])
                    - q_table[state, action]
                )

            state = next_state

        epsilon = max(epsilon - epsilon_decay, min_epsilon)

        if epsilon == min_epsilon:
            alpha *= 0.1

        if reward == 1:
            rewards_per_episode[episode_idx] = 1

        if episode_idx % 1000 == 0:
            print(f"Эпизод {episode_idx}, Epsilon: {epsilon:.4f}")

    env.close()
    print("Обучение завершено для FrozenLake-v1 4x4!")

    sum_rewards = np.zeros(num_episodes)
    for i in range(num_episodes):
        sum_rewards[i] = np.sum(rewards_per_episode[max(0, i - 100) : (i + 1)])

    plt.plot(sum_rewards)
    plt.xlabel("Number of episodes")
    plt.ylabel("Sum of rewards per 100 episodes")
    plt.savefig("frozen_lake4x4.png")

    if is_train:
        with open("q_table_4x4.pickle", "wb") as f:
            pickle.dump(q_table, f)
    return q_table, sum_rewards


def grid_search():
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    scores = []
    for alpha in alphas:
        config = {
            "alpha": alpha,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.0001,
            "min_epsilon": 0.001,
            "num_episodes": 15000,
        }
        q_table, sum_rewards = run(config, is_train=True, render=False)
        scores.append(sum_rewards[1000:].mean())

    for alpha, score in zip(alphas, scores):
        print(f"alpha: {alpha}, mean reward: {score}")


if __name__ == "__main__":
    config = {
        "alpha": 0.8,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.0001,
        "min_epsilon": 0.001,
        "num_episodes": 15000,
    }
    q_table, sum_rewards = run(config, is_train=True, render=False)
    # grid_search()
