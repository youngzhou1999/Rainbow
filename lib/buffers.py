import collections
import numpy as np

class CommonExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class Common_NSteps_ExperienceBuffer:
    def __init__(self, capacity, DEFAULT_N_STEPS=4, GAMMA=0.99):
        self.buffer = collections.deque(maxlen=capacity)
        self.n_steps = DEFAULT_N_STEPS
        self.gamma = GAMMA

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        states, actions, rewards, states_, dones = [], [], [], [], []
        # QUES? HOW TO AVOID SAME VALUE IN BATCH
        for i in range(batch_size):
            hi = np.random.randint(self.n_steps, len(self.buffer))
            lo = hi - self.n_steps
            R = 0
            state, action, reward, done, state_ = self.buffer[lo]
            states.append(state)
            actions.append(action)
            for j in range(lo, hi):
                state, action, reward, done, state_ = self.buffer[j]
                R += (self.gamma** (j - lo)) * reward
                if done:
                    break
            rewards.append(R)
            dones.append(done)
            states_.append(state_)
        
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(states_)


class PrioReplayBuffer:
    def __init__(self, buf_size, \
                prob_alpha=0.6, BETA_START=0.4, BETA_FRAMES=100000):
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros(
            (buf_size, ), dtype=np.float32)
        self.beta = BETA_START
        self.BETA_START = BETA_START
        self.BETA_FRAMES = BETA_FRAMES

    def update_beta(self, idx):
        v = self.BETA_START + idx * (1.0 - self.BETA_START) / \
            self.BETA_FRAMES
        self.beta = min(1.0, v)
        return self.beta

    def __len__(self):
        return len(self.buffer)

    def populate(self, sample):
        max_prio = self.priorities.max() if \
            self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer),
                                   batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, \
               np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices,
                          batch_priorities):
        for idx, prio in zip(batch_indices,
                             batch_priorities):
            self.priorities[idx] = prio