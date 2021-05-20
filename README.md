# Deep Q Network(DQN), its tricks and Rainbow Algorithm

## 1. Instruction

Hi. This is my codes for Rainbow Algorithm. Including:

- DQN as baseline
- Double-DQN
- Dueling-DQN
- n-steps-DQN
- DQN with PER
- Noisy-net-DQN
- Distributional DQN

And, using above tricks to code for Rainbow.

## 2. Results

I trained these codes in pong so far. The basic hyper parameters are as below:

```python
SEED = 123
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

# n-steps hyperparams
DEFAULT_N_STEPS = 4

# PER hyperparams
ALPHA = 0.6
BETA = 0.4
BETA_FRAMES = 100000

```

