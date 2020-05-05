from collections import deque

from DQN import *
from ReplayMemory import *
import gym
import torch
import numpy as np
from torch.nn.modules.loss import MSELoss
import math
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from os import path


class Agent:

    def __init__(self, gamma, epsilon_start, epsilon_end, epsilon_decay, alpha, target_update, max_iter, tau, batch_size=16, dropout_ratio=0.25):
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.target_update = target_update
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.dropout_ratio = dropout_ratio
        self.tau = tau
        self.tag = "g" + str(self.gamma) + "e" + str(self.epsilon_decay) + "lr" + str(self.alpha) + "t" \
                   + str(self.target_update) + "b" + str(self.batch_size) + "d" + str(self.dropout_ratio) + "tau" + str(self.tau)
        self.memory = ReplayMemory(5000, self.batch_size)
        self.env = gym.make("LunarLander-v2")
        self.n_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.dropout_ratio)
        self.target_net = DQN(self.dropout_ratio)
        self.policy_net = self.policy_net.float()
        self.target_net = self.target_net.float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.loss = MSELoss()

    def epsilon_greedy(self, state, epsilon):
        temp = np.random.random()
        if temp < epsilon:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.policy_net(state.float()).max(1)[1].view(1, 1)
        return action

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        return

    def model_update(self):
        samples = self.memory.create_batch()
        if not samples:
            return
        batch = Sample(*zip(*samples))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).float()
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch.float()).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss(expected_state_action_values.unsqueeze(1), state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def get_state_vec(self, state):
        state = list(state)
        # state.append(math.sqrt(state[0] ** 2 + state[1] ** 2))
        # state.append(math.sqrt(state[2] ** 2 + state[3] ** 2))
        # state.append(state[6] * state[7])
        state = np.array(state)
        return torch.from_numpy(state).unsqueeze(0)

    def load_model(self):
        self.policy_net.load_state_dict(torch.load("parameter/" + self.tag))
        return self

    def load_rewards(self):
        with open("parameter/" + self.tag + '_training_rewards.pkl', "rb") as input_file:
            rewards = pickle.load(input_file)
        return rewards

    def reward_exist(self):
        return path.exists("parameter/" + self.tag + '_training_rewards.pkl')

    def plot_rewards(self, rewards, suffix):
        filename = "results/" + self.tag
        epochs = [(i + 1) for i in range(len(rewards))]
        fig = plt.figure(figsize=(6, 4), tight_layout=True)
        plt.plot(epochs, rewards)
        plt.xlabel('epoch')
        plt.ylabel('rewards')
        fig.savefig(filename + "_" + suffix + '.png', dpi=fig.dpi)
        return

    def train(self):
        self.target_net.eval()
        self.policy_net.train()
        rewards = []
        epoch_rewards = deque(maxlen=100)
        total_loss = 0
        epsilon = self.epsilon_start
        iter_str = "not_converge"
        step = 0
        for iter in range(self.max_iter):
            state = self.env.reset()
            state_vec = self.get_state_vec(state)
            cum_rewards = 0
            epoch_loss = 0
            while True:
                step += 1
                action = self.epsilon_greedy(state_vec, epsilon)
                state_new, reward, done, _ = self.env.step(action.item())
                cum_rewards += reward
                state_new_vec = self.get_state_vec(state_new)
                reward_torch = torch.tensor([reward], device=self.device).float()
                if done:
                    if abs(reward) >= 100:
                        if reward >= 100:
                            print("succeed!")
                        state_new_vec = None
                self.memory.push(state_vec, action, reward_torch, state_new_vec)
                state_vec = state_new_vec
                loss = self.model_update()
                if loss:
                    total_loss += loss
                    epoch_loss += loss
                if step % self.target_update == 0:
                    self.soft_update(self.policy_net, self.target_net, self.tau)
                    # self.target_net.load_state_dict(self.policy_net.state_dict())
                if done:
                    break
            rewards.append(cum_rewards)
            epoch_rewards.append(cum_rewards)
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            if iter % 50 == 0:
                print("{} iteration with total loss {}, epoch loss {}, average reward {}".format(iter, total_loss, epoch_loss, np.mean(epoch_rewards)))
                print("Epsilon {}".format(epsilon))
            if np.mean(epoch_rewards) >= 200.0:
                print("Agent Learnt after {} iter".format(iter))
                iter_str = str(iter)
                break
        torch.save(self.policy_net.state_dict(), "parameter/" + self.tag)
        with open("parameter/" + self.tag + '_training_rewards.pkl', 'wb') as f:
            pickle.dump(rewards, f)
        print("Training complete")
        self.plot_rewards(rewards, "training_" + iter_str)
        return rewards

    def test(self):
        self.target_net.eval()
        self.policy_net.eval()
        rewards = []
        for iter in range(100):
            state = self.env.reset()
            state_vec = self.get_state_vec(state)
            cum_rewards = 0
            epsilon = 0.
            while True:
                action = self.epsilon_greedy(state_vec, epsilon)
                state_new, reward, done, _ = self.env.step(action.item())
                cum_rewards += reward
                state_new_vec = self.get_state_vec(state_new)
                if done:
                    break
                state_vec = state_new_vec
            rewards.append(cum_rewards)
        with open("parameter/" + self.tag + '_testing_rewards.pkl', 'wb') as f:
            pickle.dump(rewards, f)
        print("Test complete")
        print(np.mean(rewards))
        self.plot_rewards(rewards, "test")
        return

