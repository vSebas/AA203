from .Agent import Agent
from typing import List
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch, datetime

class ReinforceAgent(Agent):
    def __init__(self, state_dim : int, action_dim : int, hidden_dim : int=24) -> None:
        super().__init__(state_dim, action_dim, hidden_dim)
        self.policy_network = self.build_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters())
        self.agent_name = "reinforce"
        self.gradient_variant = "baseline"

    def build_network(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.action_dim),
            torch.nn.Softmax(dim=-1)
        )

    def policy(self, state: np.ndarray, train : bool=False) -> int:
        state = torch.tensor(state, dtype=torch.float)
        action_probs = self.policy_network(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        if train:
            log_prob = action_distribution.log_prob(action)
            return action, log_prob
        else:
            return action

    def train(self, env: gym.wrappers, num_episodes: int=500) -> None:
        reward_history = []
        for episode in range(num_episodes):
            obs, info = env.reset(seed=1738)
            terminated, truncated = False, False

            log_probs = []
            rewards = []

            while not terminated and not truncated:
                action, log_prob = self.policy(obs, train=True)
                obs, reward, terminated, truncated, info = env.step(action.item())
                log_probs.append(log_prob)
                rewards.append(reward)

            self.learn(rewards, log_probs)
            total_reward = sum(rewards)
            reward_history.append(total_reward)
            print(f"Episode {episode+1}: Total Reward = {total_reward}")

        self.plot_rewards(reward_history)

    def learn(self, rewards: list, log_probs: list) -> None:
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) Implement the naive REINFORCE, REINFORCE with causality trick and REINFORCE with causality trick + a baseline to 'center' the returns.
        ###     2) After you've finished your implementation, please comment out all sections but the section you wish to evaluate for training.
        ###
        ### Please see the following docs for support:
        ###     torch.stack: https://docs.pytorch.org/docs/stable/generated/torch.stack.html

        # 1) Naive REINFORCE
        discounted_return = torch.tensor(
            sum((self.gamma ** t) * rewards[t] for t in range(len(rewards))),
            dtype=torch.float,
        )
        naive_loss = -torch.stack(
            [log_prob * discounted_return for log_prob in log_probs]
        ).sum()

        # 2) REINFORCE with causality trick
        returns = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)
        returns = torch.tensor(returns, dtype=torch.float)
        causal_loss = -torch.stack(
            [log_prob * ret for log_prob, ret in zip(log_probs, returns)]
        ).sum()

        # 3) REINFORCE with causality trick and baseline to "center" the returns
        baseline = returns.mean()
        baseline_loss = -torch.stack(
            [log_prob * (ret - baseline) for log_prob, ret in zip(log_probs, returns)]
        ).sum()

        if self.gradient_variant == "naive":
            loss = naive_loss
        elif self.gradient_variant == "causal":
            loss = causal_loss
        else:
            loss = baseline_loss
        ###########################################################################

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    @staticmethod
    def plot_rewards(reward_history: List[int]) -> None:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.figure()
        plt.plot(reward_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Reward Curve')
        filename = f"reward_curve_{current_time}.png"
        plt.savefig(filename)
        plt.show()
        print(f"Saved reward curve as {filename}")
