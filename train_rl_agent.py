import numpy as np
from pathlib import Path

from src.rl_env import RLAttentionWrapper


def train_q_learning(
    episodes=200,
    alpha=0.1,       # learning rate
    gamma=0.95,      # discount
    epsilon_start=1.0,
    epsilon_end=0.05,
):
    env = RLAttentionWrapper(episode_length=300)

    n_states = env.n_states
    n_actions = env.n_actions

    # initialize Q-table
    Q = np.zeros((n_states, n_actions))

    epsilon_decay = (epsilon_start - epsilon_end) / max(1, episodes)

    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        epsilon = max(epsilon_end, epsilon_start - ep * epsilon_decay)

        total_reward = 0.0

        while True:
            # epsilon-greedy (exploration vs exploitation)
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _ = env.step(action)

            # Q-learning update rule
            best_next = np.max(Q[next_state])
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * best_next - Q[state, action]
            )

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)

        if (ep + 1) % 20 == 0:
            avg_r = np.mean(rewards_per_episode[-20:])
            print(f"Episode {ep+1}/{episodes}, avg reward last 20: {avg_r:.3f}, epsilon={epsilon:.2f}")

    return Q, rewards_per_episode


def main():
    Q, rewards = train_q_learning()

    # save the learned Q-table
    out_path = Path("models")
    out_path.mkdir(exist_ok=True)
    np.save(out_path / "q_table.npy", Q)
    print("Q-table saved to models/q_table.npy")

    # TODO: could add plotting of learning curve here


if __name__ == "__main__":
    main()
