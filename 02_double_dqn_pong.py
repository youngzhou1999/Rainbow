#!/usr/bin/env python3
from lib import wrappers
from lib.models import DQN 
from lib.buffers import CommonExperienceBuffer

import argparse
import time
import numpy as np
import collections
import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


SEED = 123
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19
dir = '/ddqn/'
start_datetime = datetime.datetime.now().strftime("%m-%d_%H-%M")

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    q_eval = net(states_v).gather(
        1, actions_v.unsqueeze(-1).long()).squeeze(-1)
    q_eval_ = net(next_states_v)
    _, max_acts_ = torch.max(q_eval_, dim=1)
    max_actions_ = torch.tensor([int(max_act_.item()) \
                            for max_act_ in max_acts_]).to(device)
    with torch.no_grad():
        q_target = tgt_net(next_states_v).gather(
            1,max_actions_.unsqueeze(-1).long()).squeeze(-1)
        q_target[done_mask] = 0.0
        q_target = q_target.detach()

    td_target = rewards_v + GAMMA * q_target
                                
    return nn.MSELoss()(q_eval,
                        td_target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    else:
        torch.manual_seed(SEED)

    env = wrappers.make_env(args.env)

    net = DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape,
                            env.action_space.n).to(device)
    writer = SummaryWriter(logdir=dir, comment="-" + args.env)

    buffer = CommonExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    mean_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            mean_rewards.append(m_reward)
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), m_reward, epsilon,
                speed
            ))
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            np.savetxt(dir + "ddqn_mean_reward_episode_{}.csv".format(start_datetime), np.array(mean_rewards), delimiter=",")
            np.savetxt(dir + "ddqn_reward_{}.csv".format(start_datetime), np.array(total_rewards), delimiter=",")
            if best_m_reward is None or (best_m_reward < m_reward and m_reward > 14.0):
                torch.save(net.state_dict(), dir + args.env +
                           "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
