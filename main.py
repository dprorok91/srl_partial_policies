import argparse
from typing import Any

import gymnasium as gym
import numpy as np
import tensorflow as tf

from models import action_values_array, build_policy_network

ACTION_VALUES = action_values_array()


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_env(seed: int = 0, render_mode: str | None = None):
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    env.reset(seed=seed)
    return env


def random_baseline(
    episodes: int = 20,
    max_steps: int = 200,
    seed: int = 0,
) -> list[float]:
    rewards: list[float] = []
    for episode_idx in range(episodes):
        env = make_env(seed=seed + episode_idx)
        total_reward = 0.0
        terminated = False
        truncated = False
        observation, _ = env.reset(seed=seed + episode_idx)

        for _ in range(max_steps):
            action = ACTION_VALUES[np.random.randint(len(ACTION_VALUES))]
            observation, reward, terminated, truncated, _ = env.step(np.array(action, dtype=np.float32))
            total_reward += float(reward)
            if terminated or truncated:
                break

        rewards.append(total_reward)
        env.close()

    return rewards


def sample_action_from_logits(logits: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    logits = tf.cast(logits, tf.float32)
    logits_2d = tf.expand_dims(logits, axis=0)
    action_index = tf.random.categorical(logits_2d, num_samples=1)[0, 0]
    chosen_log_prob = tf.gather(tf.nn.log_softmax(logits), action_index)
    action_value = tf.gather(tf.constant(ACTION_VALUES, dtype=tf.float32), action_index)
    return action_index, action_value, chosen_log_prob


def greedy_action_from_logits(logits: tf.Tensor) -> tf.Tensor:
    action_index = tf.argmax(logits, axis=-1)
    return tf.gather(tf.constant(ACTION_VALUES, dtype=tf.float32), action_index)


def collect_episode(
    policy: tf.keras.Model,
    env,
    max_steps: int = 200,
    training: bool = True,
    render: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    if seed is None:
        observation, _ = env.reset()
    else:
        observation, _ = env.reset(seed=seed)

    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        state = tf.convert_to_tensor(observation, dtype=tf.float32)
        logits = policy(tf.expand_dims(state, axis=0), training=training)[0]

        if training:
            action_index, action_value, _ = sample_action_from_logits(logits)
        else:
            action_index = tf.argmax(logits, axis=-1)
            action_value = tf.gather(tf.constant(ACTION_VALUES, dtype=tf.float32), action_index)

        next_observation, reward, terminated, truncated, _ = env.step(np.array(action_value, dtype=np.float32))

        states.append(state.numpy().copy())
        actions.append(int(action_index.numpy()))
        rewards.append(float(reward))

        if render:
            env.render()

        observation = next_observation
        if terminated or truncated:
            break

    return {"states": states, "actions": actions, "rewards": rewards}


def discounted_returns(rewards: list[float], gamma: float = 0.99) -> np.ndarray:
    running_total = 0.0
    returns: list[float] = []
    for reward in reversed(rewards):
        running_total = reward + gamma * running_total
        returns.append(running_total)
    returns.reverse()
    return np.asarray(returns, dtype=np.float32)


def policy_log_probs(policy: tf.keras.Model, states: np.ndarray, actions: np.ndarray) -> tf.Tensor:
    log_probs: list[tf.Tensor] = []
    for state, action in zip(states, actions):
        logits = policy(tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0), training=True)[0]
        log_prob = tf.gather(tf.nn.log_softmax(logits), tf.cast(action, tf.int32))
        log_probs.append(log_prob)
    return tf.stack(log_probs, axis=0)


def train_reinforce(
    policy: tf.keras.Model,
    episodes: int = 200,
    max_steps: int = 200,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    seed: int = 0,
) -> dict[str, Any]:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    reward_history: list[float] = []
    loss_history: list[float] = []

    for episode_idx in range(episodes):
        env = make_env(seed=seed + episode_idx)
        episode = collect_episode(policy, env, max_steps=max_steps, training=True, seed=seed + episode_idx)
        env.close()

        returns = discounted_returns(episode["rewards"], gamma=gamma)
        states = tf.convert_to_tensor(np.asarray(episode["states"], dtype=np.float32), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.asarray(episode["actions"], dtype=np.int32), dtype=tf.int32)

        with tf.GradientTape() as tape:
            log_probs = policy_log_probs(policy, states, actions)
            discounted_returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
            loss = -tf.reduce_sum(log_probs * discounted_returns_tensor) / tf.cast(tf.shape(log_probs)[0], tf.float32)

        grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))

        episode_reward = float(np.sum(episode["rewards"]))
        reward_history.append(episode_reward)
        loss_history.append(float(loss.numpy()))

    return {
        "policy": policy,
        "reward_history": reward_history,
        "loss_history": loss_history,
    }


def evaluate_policy(
    policy: tf.keras.Model,
    episodes: int = 10,
    max_steps: int = 200,
    seed: int = 0,
    render: bool = False,
) -> tuple[list[float], float]:
    rewards: list[float] = []
    for episode_idx in range(episodes):
        env = make_env(seed=seed + episode_idx, render_mode="human" if render else None)
        total_reward = 0.0
        observation, _ = env.reset(seed=seed + episode_idx)

        for _ in range(max_steps):
            state = tf.convert_to_tensor(observation, dtype=tf.float32)
            logits = policy(tf.expand_dims(state, axis=0), training=False)[0]
            action_value = greedy_action_from_logits(logits)
            observation, reward, terminated, truncated, _ = env.step(np.array(action_value, dtype=np.float32))
            total_reward += float(reward)
            if render:
                env.render()
            if terminated or truncated:
                break

        rewards.append(total_reward)
        env.close()

    return rewards, float(np.mean(rewards))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple REINFORCE agent on Pendulum-v1.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes.")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    set_seed(args.seed)

    policy = build_policy_network(input_dim=3, num_actions=len(ACTION_VALUES), hidden_layers=[])

    random_rewards = random_baseline(episodes=20, max_steps=args.max_steps, seed=args.seed)
    training_summary = train_reinforce(
        policy=policy,
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
    )
    eval_rewards, mean_eval_reward = evaluate_policy(
        policy=training_summary["policy"],
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed + 1000,
        render=args.render,
    )

    print(f"Random baseline mean reward: {np.mean(random_rewards):.2f}")
    print(f"Training episodes: {args.episodes}")
    print(f"Final training reward: {training_summary['reward_history'][-1]:.2f}")
    print(f"Evaluation mean reward: {mean_eval_reward:.2f}")

    return {
        "policy": training_summary["policy"],
        "random_baseline": random_rewards,
        "reward_history": training_summary["reward_history"],
        "loss_history": training_summary["loss_history"],
        "eval_rewards": eval_rewards,
        "mean_eval_reward": mean_eval_reward,
    }


if __name__ == "__main__":
    main()
