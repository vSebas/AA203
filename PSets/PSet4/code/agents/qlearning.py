from .Agent import Agent, Transition
from collections import deque
from typing import Union
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import math, torch, random

class QLearning(Agent):
    def __init__(self, state_dim : int, action_dim : int, hidden_dim : int=24, use_gpu : bool=False) -> None:
        super().__init__(state_dim, action_dim, hidden_dim, use_gpu)
        self.policy_network = self.build_network().to(self.device)
        self.buffer = deque([], maxlen=1000) # empty replay buffer with the 1000 most recent transitions
        self.agent_name = "qlearning"

        self.iteration = 0

    def eps_threshold(self) -> float: # epsilon threshold for e-greedy exploration
        eps_start, eps_end, eps_decay = 0.9, 0.05, 1000
        self.iteration += 1
        return eps_end + (eps_start - eps_end) * math.exp(-1 * self.iteration / eps_decay)

    def build_network(self) -> torch.nn.Module:
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) Construct and return a Multi-Layer Perceptron (MLP) representing the Q function. Recall that a Q function accepts
        ###     two arguments i.e., a state and action pair. For this implementation, your Q function will process an observation
        ###     of the state and produce an estimate of the expected, discounted reward for each available action as an output -- allowing you to 
        ###     select the prediction assosciated with either action.
        ###     2) Use a hidden layer dimension as specified by 'self.hidden_dim'.
        ###     3) Our solution implements a three layer MLP with ReLU activations on the first two hidden units.
        ###     But you are welcome to experiment with your own network definition!
        ###
        ### Please see the following docs for support:
        ###     nn.Sequential: https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        ###     nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        ###     nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.action_dim),
        )
        ###########################################################################
    
    def policy(self, state : Union[np.ndarray, torch.tensor], train : bool=False) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) If train == True, sample from the policy with e-greedy exploration with a decaying epsilon threshold. We've already
        ###     implemented a function you can use to call the exploration threshold at any instance in the iteration i.e., 'self.eps_treshold()'.
        ###     2) If train == False, sample the action with the highest Q value as predicted by your network.
        ###     HINT: An exemplar implementation will need to use torch.no_grad() in the solution.
        ###
        ### Please see the following docs for support:
        ###     random.random: https://docs.python.org/3/library/random.html#random.random
        ###     torch.randint: https://docs.pytorch.org/docs/stable/generated/torch.randint.html
        ###     torch.argmax: https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
        ###     torch.no_grad(): https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
        if train and random.random() < self.eps_threshold():
            return torch.randint(self.action_dim, (1,), device=self.device)

        with torch.no_grad():
            q_values = self.policy_network(state)
            return torch.argmax(q_values, dim=-1)
        ###########################################################################
    
    def sample_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(self.buffer) > self.batch_size
        samples = random.sample(self.buffer, self.batch_size)

        states, actions, targets = [], [], []
        for i in range(self.batch_size):
            s, a, r, sp = samples[i].state.to(self.device), samples[i].action.item(), samples[i].reward.to(self.device), samples[i].next_state if samples[i].next_state is None else samples[i].next_state.to(self.device)
            states.append(s)
            actions.append(a)
            with torch.no_grad():
                targets.append(r if sp is None else r + self.gamma*torch.max(self.policy_network(sp)))
        
        return torch.cat(states), torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1), torch.cat(targets).unsqueeze(1)

    def train(self, env : gym.wrappers, num_episodes : int=100) -> None:
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) Implementing the training algorithm according to Algorithm 1 on page 5 in "Playing Atari with Deep Reinforcement Learning".
        ###     2) Importantly, only take a gradient step on your memory buffer if the buffer size exceeds the batch size hyperparameter. 
        ###     HINT: In our implementation, we used the AdamW optimizer.
        ###     HINT: Use the custom 'Transition' data structure to push observed (s, a, r, s') transitions to the memory buffer. Then, 
        ###     you can sample from the buffer simply by calling 'self.sample_buffer()'.
        ###     HINT: In our implementation, we clip the value of gradients to 100, which is optional.
        ###
        ### Please see the following docs for support:
        ###     torch.optim.AdamW: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        ###     torch.nn.MSELoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        ###     torch.nn.utils.clip_grad_value_: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
        optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()

        for _ in tqdm(range(num_episodes)):
            obs, info = env.reset()
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            terminated, truncated = False, False

            while not terminated and not truncated:
                action = self.policy(state, train=True)
                obs, reward, terminated, truncated, info = env.step(action.item())

                reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
                next_state = None
                if not terminated and not truncated:
                    next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                self.buffer.append(Transition(state, action, reward, next_state))
                state = next_state

                if len(self.buffer) > self.batch_size:
                    states, actions, targets = self.sample_buffer()
                    q_values = self.policy_network(states).gather(1, actions)
                    loss = loss_fn(q_values, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
                    optimizer.step()
        ###########################################################################
