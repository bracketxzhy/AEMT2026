"""PPO training loop for the ecological regulation policy."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from config import ExperimentConfig, get_experiment_config
from envs.mclsse_env import MCLSSEEnv
from models.ppo_agent import PPOActorCritic
from utils.common import ensure_directories, set_global_seed, write_csv, write_json


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rewards_array = np.asarray(rewards, dtype=np.float32)
    values_array = np.asarray(values + [0.0], dtype=np.float32)
    advantages = np.zeros_like(rewards_array)
    gae = 0.0
    for idx in range(len(rewards_array) - 1, -1, -1):
        delta = rewards_array[idx] + gamma * values_array[idx + 1] - values_array[idx]
        gae = delta + gamma * gae_lambda * gae
        advantages[idx] = gae
    returns = advantages + values_array[:-1]
    return advantages, returns


def collect_rollouts(
    agent: PPOActorCritic,
    config: ExperimentConfig,
    device: torch.device,
    iteration: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, float]]]:
    batch: Dict[str, List[np.ndarray | float]] = {
        "states": [],
        "actions": [],
        "log_probs": [],
        "returns": [],
        "advantages": [],
    }
    summaries: List[Dict[str, float]] = []

    for episode_idx in range(config.ppo.rollout_episodes_per_iteration):
        seed = config.ppo.seed + iteration * 1000 + episode_idx * 37
        env = MCLSSEEnv(config=config, seed=seed)
        state = env.reset(seed=seed)

        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        log_probs: List[float] = []
        rewards: List[float] = []
        values: List[float] = []

        episode_return = 0.0
        done = False
        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = agent.sample_action(state_tensor)
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            next_state, reward, done, info = env.step(action)

            states.append(state.astype(np.float32))
            actions.append(action)
            log_probs.append(float(log_prob_tensor.squeeze(0).cpu().item()))
            rewards.append(float(reward))
            values.append(float(value_tensor.squeeze(0).cpu().item()))

            episode_return += float(reward)
            state = next_state

        advantages, returns = compute_gae(rewards, values, config.ppo.gamma, config.ppo.gae_lambda)
        batch["states"].extend(states)
        batch["actions"].extend(actions)
        batch["log_probs"].extend(log_probs)
        batch["returns"].extend(returns.tolist())
        batch["advantages"].extend(advantages.tolist())

        summaries.append(
            {
                "episode_return": episode_return,
                "episode_length": float(len(rewards)),
                "terminated_unsafe": float(info["done_reason"] != "horizon"),
            }
        )

    arrays = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in batch.items()
    }
    return arrays, summaries


def update_agent(
    agent: PPOActorCritic,
    optimizer: optim.Optimizer,
    rollout: Dict[str, np.ndarray],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    states = torch.as_tensor(rollout["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []

    num_samples = states.shape[0]
    minibatch = min(config.ppo.batch_size, num_samples)

    for _ in range(config.ppo.update_epochs):
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, minibatch):
            batch_indices = indices[start : start + minibatch]
            mb_states = states[batch_indices]
            mb_actions = actions[batch_indices]
            mb_old_log_probs = old_log_probs[batch_indices]
            mb_returns = returns[batch_indices]
            mb_advantages = advantages[batch_indices]

            new_log_probs, entropy, values = agent.evaluate_actions(mb_states, mb_actions)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surrogate_1 = ratio * mb_advantages
            surrogate_2 = torch.clamp(
                ratio,
                1.0 - config.ppo.clip_epsilon,
                1.0 + config.ppo.clip_epsilon,
            ) * mb_advantages
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            value_loss = F.mse_loss(values, mb_returns)
            entropy_bonus = entropy.mean()
            loss = policy_loss + config.ppo.value_coef * value_loss - config.ppo.entropy_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.ppo.grad_clip)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            entropies.append(float(entropy_bonus.detach().cpu().item()))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }


def train_ppo(
    config: ExperimentConfig | None = None,
    device: torch.device | None = None,
) -> Tuple[PPOActorCritic, List[Dict[str, float]]]:
    config = config or get_experiment_config()
    ensure_directories([config.output_root, config.figure_dir, config.metric_dir, config.model_dir])
    write_json(config.config_snapshot_path, config.to_dict())
    set_global_seed(config.ppo.seed)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOActorCritic(
        state_dim=config.ppo.state_dim,
        action_dim=config.ppo.action_dim,
        hidden_dim=config.ppo.hidden_dim,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.ppo.learning_rate)

    history: List[Dict[str, float]] = []
    best_return = -float("inf")

    for iteration in range(1, config.ppo.training_iterations + 1):
        rollout, summaries = collect_rollouts(agent, config, device, iteration)
        losses = update_agent(agent, optimizer, rollout, config, device)

        mean_return = float(np.mean([item["episode_return"] for item in summaries]))
        mean_length = float(np.mean([item["episode_length"] for item in summaries]))
        unsafe_rate = float(np.mean([item["terminated_unsafe"] for item in summaries]))
        history.append(
            {
                "iteration": float(iteration),
                "mean_return": mean_return,
                "mean_episode_length": mean_length,
                "unsafe_rate": unsafe_rate,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
            }
        )

        if mean_return > best_return:
            best_return = mean_return
            torch.save(
                {
                    "state_dict": agent.state_dict(),
                    "config": config.to_dict(),
                    "best_mean_return": best_return,
                },
                config.checkpoint_path,
            )

    write_csv(config.training_history_path, history)
    return agent, history


def main() -> None:
    config = get_experiment_config()
    agent, history = train_ppo(config=config)
    print(
        f"Training completed with {len(history)} iterations. "
        f"Best checkpoint saved to {config.checkpoint_path}."
    )


if __name__ == "__main__":
    main()

