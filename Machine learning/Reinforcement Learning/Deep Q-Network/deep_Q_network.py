import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np
from collections import deque
import random


rows = 4
cols = 4

start = (3, 0)
goal = (0, 3)
trap = (1, 2)

#          w, s, a, d
actions = [0, 1, 2, 3]

moves = {0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)}

epsilon = 1
min_epsilon = 0.16
epsilon_decay = 0.995

discount_factor = 0.95

lr = 0.001

BATCH_SIZE = 32

EPISODES = 1000
MAX_STEPS = 100

# These consecutive experiences are highly correlated. Training directly on them can make neural-network learning unstable.
# Experience Replay allows us to randomly sample old experiences, creating more varied training batches.
replay_buffer = deque(maxlen = 10000)

class QNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features = 2, out_features = 64)
        self.layer_2 = nn.Linear(in_features = 64, out_features = 4)

        self.relu = nn.ReLU()


    def forward(self, x:torch.tensor):

        x = self.layer_1(x)

        x = self.relu(x)

        x = self.layer_2(x)

        return x


def choose_action(state:tuple, epsilon:float, model) -> int:

    r = np.random.random()

    if r < epsilon:
        action = int(np.random.choice(actions))

    else:
        with torch.no_grad():

            output = model(torch.tensor([state], dtype= torch.float32))

            if not torch.isfinite(output).all():
                print("Invalid Q-values:", output)
                print("State:", state)
                raise ValueError("Q-network produced NaN or infinity")
            
            action = torch.argmax(output, dim = 1).item()
    
    return int(action)

def step(state, action):

    reward = 0
    done = False

    if action not in actions:
        raise ValueError("Invalid action")

    new_state = (state[0] + moves[action][0], state[1] + moves[action][1])

    if new_state == goal:
        done = True
        reward += 100

    elif new_state == trap:
        done = True
        reward -= 100

    elif 0 <= new_state[0] < rows and 0 <= new_state[1] < cols:
        done = False
        reward -= 1

    else:
        done = False
        reward -= 1
        new_state = state

    return new_state, reward, done


def remember(state, action, reward, new_state, done):

    replay_buffer.append((state, action, reward, new_state, done))


def sample_batch():

    if len(replay_buffer) < BATCH_SIZE:
        return None

    return random.sample(replay_buffer, BATCH_SIZE)


def prepare_batch(batch):

    states, actions, rewards, new_states, dones = zip(*batch)

    states = torch.tensor(states, dtype = torch.float32)
    actions = torch.tensor(actions, dtype = torch.long)
    rewards = torch.tensor(rewards, dtype = torch.float32)
    new_states = torch.tensor(new_states, dtype = torch.float32)
    dones = torch.tensor(dones, dtype = torch.bool)

    return states, actions, rewards, new_states, dones

def calc_target_q(rewards, new_states, dones, model):

    with torch.no_grad():

        next_q_val = model(new_states)

        max_next_q = torch.max(next_q_val, dim = 1).values

        target_q = rewards + (~dones).float() * discount_factor * max_next_q

    return target_q

def predict_q_values(states, actions, model):

    output = model(states)

    actions = torch.unsqueeze(actions, 1)

    selected_q = torch.gather(output, dim = 1, index = actions)

    selected_q = selected_q.squeeze(1)

    return selected_q

def calc_loss(pred_q, target_q, loss_fn):
    
    return loss_fn(pred_q, target_q)

def train_step(states, actions, rewards, new_states, dones, model, optimizer, loss_fn):

    target_q = calc_target_q(rewards, new_states, dones, model)

    pred_q = predict_q_values(states, actions, model)

    loss = calc_loss(pred_q, target_q, loss_fn)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()

def evaluate(model):
    state = start
    path = [state]
    total_reward = 0

    for _ in range(MAX_STEPS):
        action = choose_action(state, epsilon=0, model=model)

        new_state, reward, done = step(state, action)

        path.append(new_state)
        total_reward += reward
        state = new_state

        if done:
            break

    print("Path:", path)
    print("Total reward:", total_reward)
    print("Reached goal:", state == goal)

model = QNetwork()

optimizer = optim.Adam(model.parameters(), lr)

loss_fn = nn.SmoothL1Loss()


for episode in range(EPISODES):

    state = start

    total_reward = 0

    for step_count in range(MAX_STEPS):

        action = choose_action(state, epsilon, model)

        if action not in actions:
            print("Action:", action, type(action))
            print("State:", state)
            print("Epsilon:", epsilon)
            raise ValueError("Invalid action generated")

        new_state, reward, done = step(state, action)

        remember(state, action, reward, new_state, done)

        batch = sample_batch()
        
        if batch is not None:
            states, batch_actions, rewards, new_states, dones = prepare_batch(batch)

            loss = train_step(states, batch_actions, rewards, new_states, dones, model, optimizer, loss_fn)

        state = new_state
        total_reward += reward
            
        if done:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(
            f"Episode: {episode + 1}, "
            f"Reward: {total_reward}, "
            f"Epsilon: {epsilon:.3f}"
        )

# testing
evaluate(model)