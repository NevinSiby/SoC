import snake
import torch
import numpy as np
from collections import deque as dq
import matplotlib.pyplot as plt

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPLAY_BUFFER = dq(maxlen = 50_000)
hiddensize1 = 512
hiddensize2 = 128
learning_rate = 1e-4
GAMMA = 0.995
BATCH_SIZE = 256
UPDATION_TIME = 200
LEARN_EVERY = 5


EPISODES = 100_000
EPSILON = 1
EPSILON_DECAY_ENDS = EPISODES//4 * 3
EPSILON_DECAY = EPSILON/EPSILON_DECAY_ENDS
in_sz = snake.cell_no ** 2
ac_spc = 4
SHOW_EVERY = 5_000



REWARD_BATCHES = 100
AVG_REWARD = []
MAX_REWARD = []
MIN_REWARD = []



REWARDS = []

class Model(torch.nn.Module):
    def __init__(self, input_size, Action):
        super(Model, self).__init__()
        self.inp_lay = torch.nn.Linear(input_size, hiddensize1)
        self.relu = torch.nn.ReLU()
        self.hidden_layer = torch.nn.Linear(hiddensize1, hiddensize2)
        self.output = torch.nn.Linear(hiddensize2, Action)
    def forward(self, x):
        out = self.relu(self.inp_lay(x))
        out = self.relu(self.hidden_layer(out))
        return self.output(out)

class DQNAgent:
    def __init__(self, input, action_space, learning_rate):
        self.Q = Model(input, action_space).to(dev)
        self.target = Model(input, action_space).to(dev)
        self.target.load_state_dict(self.Q.state_dict())

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr = learning_rate)


    def update(self, pred, actual, ep):
        l = self.loss(pred, actual)
        l.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if ep % UPDATION_TIME == 0:
            self.target.load_state_dict(self.Q.state_dict())


def grid_from_state(snake_obj, fruit_obj):
    state = np.zeros((snake.cell_no,snake.cell_no))
    if 0<=snake_obj.Snake_body[0].x<=29 and 0<=snake_obj.Snake_body[0].y<=29: state[int(snake_obj.Snake_body[0].x)][int(snake_obj.Snake_body[0].y)] = 10
    decrement = 8 / len(snake_obj.Snake_body)
    l = len(snake_obj.Snake_body)
    for i in range(1,l):
        if 0<=snake_obj.Snake_body[i].x<=29 and 0<=snake_obj.Snake_body[i].y<=29: state[int(snake_obj.Snake_body[i].x)][int(snake_obj.Snake_body[i].y)] = 10 - i*decrement
    state[int(fruit_obj.pos.x)][int(fruit_obj.pos.y)] = 1
    return state

DQNmodel = DQNAgent(in_sz, ac_spc, learning_rate)


DQNmodel.Q.train()
for episode in range(EPISODES):
    my_snake = snake.Snake()
    my_fruit = snake.Fruit()
    rew = 0
    state_grid = grid_from_state(my_snake, my_fruit)
    state = torch.tensor(state_grid, dtype=torch.float32, device=dev).flatten()
    done = False
    steps = 0
    score = 0
    step_count = 0
    while not done:
        if(snake.random.random() < EPSILON):
            N = snake.random.randint(0,3)
        else:
            with torch.no_grad(): N = torch.argmax(DQNmodel.Q(state)).item()
        my_snake.action(N)
        prev_dist, current_dist = my_snake.Movement(my_fruit)
        my_fruit, _ , reward, done, score, steps = my_snake.RETURNING_ALL(my_fruit, score, steps)
        step_count += 1
        if(current_dist < prev_dist):
            reward+=5
        else:
            reward-=1
        rew += reward
        if episode%SHOW_EVERY == 0:
            snake.Draw(my_fruit, my_snake, None)
        state_grid = grid_from_state(my_snake, my_fruit)
        new_state = torch.tensor(state_grid, dtype=torch.float32, device=dev).flatten()
        REPLAY_BUFFER.append((state, N, new_state, done, reward))
        state = new_state
        EPSILON = max(0.01, EPSILON - (EPSILON_DECAY / (EPISODES / 100)))
        if episode % SHOW_EVERY == 0:
            snake.Draw(my_fruit, my_snake, None)
        if step_count % LEARN_EVERY == 0 and len(REPLAY_BUFFER) > BATCH_SIZE:
            batch = snake.random.sample(REPLAY_BUFFER, BATCH_SIZE)
            state, action, new_state, done, reward = zip(*batch)
            states = torch.stack(state).to(dev)
            new_states = torch.stack(new_state).to(dev)
            actions = torch.tensor(action, dtype = torch.int64).to(dev)
            rewards = torch.tensor(reward, dtype = torch.float32).to(dev)
            dones = torch.tensor(done, dtype=torch.bool).to(dev)
            q_values = DQNmodel.Q(states)
            req_q_values = q_values.gather(1, actions.unsqueeze(1)).to(dev)
            with torch.no_grad():
                target_q_values = DQNmodel.target(new_states)
                future_q = torch.max(target_q_values, dim=1)[0].unsqueeze(1)
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                target = rewards + GAMMA*future_q*(~dones).to(dev)
            DQNmodel.update(req_q_values, target, episode)
    del my_fruit
    del my_snake
    REWARDS.append(rew)
    if len(REWARDS) % REWARD_BATCHES == 0:
        AVG_REWARD.append(sum(REWARDS[-REWARD_BATCHES:])/ REWARD_BATCHES)
        MAX_REWARD.append(max(REWARDS[-REWARD_BATCHES:]))
        MIN_REWARD.append(min(REWARDS[-REWARD_BATCHES:]))

X = [i*REWARD_BATCHES for i in range(len(AVG_REWARD))]
plt.plot(X, AVG_REWARD, label = "Average Reward", color = "blue")
plt.plot(X, MIN_REWARD, label = "Minimum Reward", color = "red")
plt.plot(X, MAX_REWARD, label = "Maximum Reward", color = "green")
plt.xlabel("Batches")
plt.ylabel("Rewards")
plt.show()

print("done")
