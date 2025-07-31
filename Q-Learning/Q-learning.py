import snake
import numpy as np
import matplotlib.pyplot as plt
from math import exp



EPISODES = 50_000
Q_TABLE_SIZE = [2, 2, 2, 3, 3, 4]
ALPHA = 0.1
GAMMA = 0.95

EPSILON = 1

EPSILON_DECAY_ENDS =  (EPISODES//6) * 5
EPSILON_DECAY = EPSILON / EPSILON_DECAY_ENDS

SHOW_EVERY = 2_000     

q_table = np.random.uniform(low= -1000, high=-900, size=Q_TABLE_SIZE)

def reset(f, s):
    
    vec = f - s
    if vec.x == 0: a = 0
    else: a = vec.x / abs(vec.x)
    if vec.y == 0: b=0
    else: b = vec.y / abs(vec.y)
    S, R, L = my_snake.Snake_body[0] + my_snake.movevector, my_snake.Snake_body[0] + snake.v2(my_snake.movevector.y, my_snake.movevector.x), my_snake.Snake_body[0] + snake.v2(-my_snake.movevector.y, my_snake.movevector.x)
    Dang_S, Dang_L, Dang_R = 0, 0, 0
    if S.x > snake.cell_no-1 or S.x<0 or S.y>snake.cell_no-1 or S.y<0:
        Dang_S = 1
    if R.x > snake.cell_no-1 or R.x<0 or R.y>snake.cell_no-1 or R.y<0:
        Dang_R = 1
    if L.x > snake.cell_no-1 or L.x<0 or L.y>snake.cell_no-1 or L.y<0:
        Dang_L = 1
    
    state = [int(Dang_S), int(Dang_R), int(Dang_L), int(a + 1) , int(b + 1)]
    return(state)


BATCHES = 100
AVG_REWARD = []
MAX_REWARD = []
MIN_REWARD = []



SCORES = []




for episode in range(EPISODES):
    done = False
    steps = 0
    rew = 0
    my_snake = snake.Snake()
    my_fruit = snake.Fruit()
    state = reset(my_fruit, my_snake)
    score = 0


    while not done:
        decider = snake.random.random()
        if decider > EPSILON:
            N = np.argmax(q_table[tuple(state)])
        else:
            N = snake.random.randint(0, 3)

        my_snake.action(N)
        steps+=1
        prev_distance, current_distance = my_snake.Movement(my_fruit)
        if current_distance < prev_distance:
            reward = 5
        else:
            reward = -1

        my_fruit, new_state, r, done, score, steps = my_snake.RETURNING_ALL(my_fruit, score, steps)
        reward += r
        rew+=reward

        if episode % SHOW_EVERY == 0:
            snake.Draw(my_fruit, my_snake, None)
        
        if done:
            new_curr_q = reward
        else:
            max_future_q = np.max(q_table[tuple(new_state)])
            curr_q = q_table[tuple(state + [N])]
            new_curr_q = curr_q + ALPHA*(reward + GAMMA*max_future_q - curr_q)

        q_table[tuple(state + [N])] = new_curr_q
        state = new_state
        



    REWARDS.append(rew)
    
    SCORES.append(score)
    if len(REWARDS) % BATCHES == 0:
        AVG_REWARD.append(sum(REWARDS[-BATCHES:])/ BATCHES)
        MAX_REWARD.append(max(REWARDS[-BATCHES:]))
        MIN_REWARD.append(min(REWARDS[-BATCHES:]))





    EPSILON = max(0, EPSILON - EPSILON_DECAY)

    del my_fruit
    del my_snake



x = [i*BATCHES for i in range(len(AVG_REWARD))]
plt.plot(x, AVG_REWARD, label = "Avg Reward", color = "Blue")
plt.plot(x, MAX_REWARD, label = "Max Reward", color = "Green")
plt.plot(x, MIN_REWARD, label = "Min Reward", color = "Red")
plt.xlabel("EPISODES")
plt.ylabel("REWARDS")

plt.legend()
plt.show()


np.save(f"q_table_1_table_size_{2 * 2 * 2 * 3 *3  * 4}_divby2_new_2.npy", q_table)
