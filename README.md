# SoC
# Topic: RL: Race and Learn (ID:90)
## Week 1:
### Python
Python is a high-level, interpreted programming language known for its readability and simplicity. It's widely used for scripting, automation, data science, web development, and more.
#### Variables and datatypes
Variables store data. Python supports datatypes like int, float, str, list, dict, set, bool
#### Control flow
Python uses if, elif, else for conditionals and for, while loops for iteration.
#### Functions
Functions are defined using the *def* keyword to organize reusable blocks of code.
#### Classes and Objects
Python is object-oriented. Classes define custom types with attributes and methods.
#### Lambda and Map
Lambda defines anonymous functions. map, filter, reduce apply functions over sequences.
```python
make_multiplier = lambda n: lambda x: x * n
double = make_multiplier(2)
tripler = make_multiplier(3)
print(double(5))  # Output: 10
print(tripler(5))  # Output 15
```
#### Built-in Functions
Common ones: len(), sum(), range(), sorted(), enumerate().
 
#### Libraries
Popular libraries include numpy, pandas, matplotlib, torch, pygame.

### Pygame
It is a library in python. It provides functions for sound rendering, adding graphics, handling inputs. So this could be used for making a game.
#### Game Window
*pygame.display.set_mode(H, W)* creates a window where the game is rendered. The height and width of this window will be *W* and *H* respectively.

#### Input Handling
Handles user inputs (keyboard, mouse) using pygame.event.get(). Use pygame.key.get_pressed() and pygame.mouse.get_pos() for input polling.

#### Drawing
Use pygame.draw functions to draw shapes like circles, rectangles, and lines.

#### Surfaces
Images and drawings are Surface objects, which can be blitted (copied) onto the screen. Screenname.blit() is used to do this blitting to the Game window.

#### Images
pygame.image.load() loads images; use blit() to display them.

#### Sounds
pygame.mixer module plays sound effects and background music. We need to write pygame.mixer.init() in the beginning of the program to use this.

#### Sprites
pygame.sprite.Sprite is a built-in class for managing game objects (Like we can use this for an aircraft in a shooting game).

#### Collision Detection
pygame.sprite.colliderect() and related functions detect overlaps between sprites (This is used in my program to find if snake aquired fruit)


#### FPS and Clock
pygame.time.Clock() is used to control frame rate and game speed. 

#### Game Loop
Core of any game: continuously processes input, updates state, and renders output. This s usually run in a fixed fps(frames per second)

As an assignment, I have made a snake game using pygame. The code for the same is included in the repository under file name [snake.py](./snake.py)

## Weeks 2:
### Neural Networks
A Neural Network (NN) is a computational model inspired by the human brain, used to approximate complex functions and patterns from data. It consists of layers of interconnected nodes (neurons) that process inputs and learn to make predictions through training. 

Neurons basically takes input from the previous layer and spits out as a scalar.
#### Linear Neural Network
In a linear neural network, each neuron performs a weighted sum of its inputs and adds a bias:
```math
a = w \cdot x + b
```
- a: input vector (previous layer)
- w: weight matrix
- x: output vector (this is the current layer)
- b: bias vector
By this each layer is connected to its previous layer.

If no activation functions are used between layers, then composing multiple layers still results in a linear function. So we impose activation functions.

**Activation function** : An activation function is applied to the output of each neuron to introduce non-linearity into the network. Without it, even deep networks would behave like a linear model. It "squishes" the neuron's output (a scalar) to a desired range and enables the model to learn complex, non-linear patterns.

```math
\text{Sigmoid:}\: \sigma(x) = \frac{1}{1 + e^{−x}}
```
```math
\text{ReLU (Rectified Linear Unit):}\: ReLU(x) = \max(0, x)
```
```math
\text{Tanh (Hyperbolic Tangent):} \: tanh(x) = \frac{e^x − e^{−x}}{e^x + e^{−x}}
```
#### Cost Function
Cost is sum of square of difference between required output and output our neural network give. This is used in learning. Cost function should be minimised for an accurate neural network.
#### Backtracking
We find the gradient of Codt function with respect to w and b and subtract w and b by learning rate ∙ (- grad). By this, the cost function decreases over and over while we inputting test dataset. This is called **backtracking** because we're moving backwards as we find the output first and then readjust the weights and biases.

```math
w \leftarrow w − \alpha \cdot \frac{\partial C}{\partial w}
```
```math
b \leftarrow b − \alpha \cdot \frac{\partial C}{\partial b}
```
```math
\alpha \: \text{ is learning rate and C is the cost function}
```
#### Convolutional Neural Network
This is used when we need to recognise images and neurons are here pixels. So it's 2 dimensional
```math
a = w * x + b
```
- x: input image or feature map (2D)
- w: filter/kernel (usually 3×3 or 5×5 window)
- *: convolution operation (not dot product!)
- b: bias term
- a: output feature map (activation map

CNNs typically use ReLU activations, pooling layers, and finally fully connected layers for classification. Like standard NNs, CNNs also use backpropagation to train the filters and weights.

> Also I learnt to implement this in Pytorch

As part of the assignment, also prepared a brief report documenting the architecture, training results, and key observations from the implementation. The full code and report are included in this repository in the folder [MNIST Classification](./MNIST%20Classification/).

## Week 3
### Reinforcement Learning (RL)
Reinforcement Learning (RL) is a framework where agents learn to make decisions by interacting with an environment and receiving feedback in the form of rewards. The foundation of RL is built on Markov Decision Processes (MDPs).
###  Markov Decision Processes (MDPs) and  Markov Reward Processes (MRPs).
A **Markov Decision Process (MDP)** is a framework for modeling environments where an agent makes decisions to maximize cumulative reward through interaction.
**Tuple form:**
```python
MDP = (S, A, P, R, γ)
```
- **S**: Set of states  
- **A**: Set of actions  
- **P(s'|s, a)**: Transition probability — probability of reaching `s'` from `s` after action `a`  
- **R(s, a)**: Reward function — expected reward for taking action `a` in state `s`  
- **γ**: Discount factor — measures the importance of future rewards  
> In MDPs, the agent learns a **policy** π(s) that chooses actions to maximize expected future reward.
A **Markov Reward Process (MRP)** is a simplified environment model where no decisions are made. It represents a sequence of states and rewards, with probabilistic transitions and passive reward collection.
**Tuple form:**
```python
MRP = (S, P, R, γ)
```
- **S**: Set of possible states  
- **P(s'|s)**: Transition probability — probability of moving from state `s` to `s'`  
- **R(s)**: Reward function — expected reward upon entering state `s`  
- **γ**: Discount factor (0 ≤ γ ≤ 1) — determines the present value of future rewards  
MRPs help evaluate how valuable each state is, without involving any actions or control by the agent.

### Policy
A policy is a rule used by an agent to decide what actions to take. It can be deterministic or stochastic. Determiinistic policy is denoted by μ, aₜ = μ(sₜ) *(which means aₜ is performed when we are in sₜ.)*. Stochastic policy is denoted by π, aₜ ~ π(· | sₜ) *(This denotes the probability that we perform aₜ, whe we are in sₜ)*.

### Value Functions
In reinforcement learning, a **value function** estimates how good it is for an agent to be in a particular state (or to take a specific action in that state). It captures the **expected return** (i.e., total future reward) the agent can achieve from that point onward, under a given policy.
There are two main types of value functions:
- **State Value Function** `V(s)`  
  The expected return when starting from state `s` and following a policy `π` thereafter.  
  Mathematically:  
  ```math
  V(s) = \mathbb{E}_\pi[ G_t | s_t = s ] = R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + \dots = \mathbb{E}_\pi \left[ \sum_k \gamma^k · r_{t+k} | s_t = s \right]
  ```

- **Action Value Function** `Q(s, a)`  
  The expected return when starting from state `s`, taking action `a`, and then following policy `π`.  
  Mathematically:
  ```math
   Q(s,a) = \mathbb{E}_\pi[ G_t | s_t = s, a_t = a ] = R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + \dots = \mathbb{E}_\pi \left[ \sum_k \gamma^k · r_{t+k}  | s_t = s, a_t = a \right]
  ```

Where:
- `E[·]` is the expectation (average over possible outcomes)
- `γ` is the discount factor (0 ≤ γ ≤ 1)
- `Gₜ` is the return from time step `t`
- `rₜ₊ₖ₊₁` is the reward at future time step `t + k + 1`
Value functions are central to many reinforcement learning algorithms — they guide the agent toward higher-reward behaviors by predicting the long-term benefit of states and actions.
These are computed using the **Bellman equations**, which break down future rewards recursively.
### Bellmann equation
For State value function
```math
\begin{align*}
V(s) &= E[ G_t | s_t = s ] \\
     &= R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + \dots \\
     &= R_t + \gamma \cdot (R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + \dots) \\
     &= R_t + \gamma \cdot V(s_{t+1}) \\
     &= \mathbb{E}[ R_t + \gamma \cdot V(s_{t+1}) | s_t = s ]
\end{align*}
```
For Action value function, 
```math
\begin{align*}
Q(s, a) &= E[ G_t | s_t = s , a_t = a] \\
     &= R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + \dots \\
     &= R_t + \gamma \cdot (R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + \dots) \\
     &= R_t + \gamma \cdot Q(s_{t+1}, a_{t+1}) \\
     &= \mathbb{E}[ R_t + \gamma \cdot Q(s_{t+1}) | s_t = s, a_t = a ]
\end{align*}
```
### Dynamic Programming (DP)
Dynamic Programming is a class of algorithms used to compute value functions and derive optimal policies in reinforcement learning, assuming full knowledge of the environment's dynamics (i.e., the transition probabilities and reward function).
DP solves problems recursively using the **Bellman equations** and relies on two main strategies:

#### Policy Evaluation
Estimate the value function `V(s)` for a fixed policy π:
```math
V(s) = \mathbb{E}_\pi [ R_t + \gamma \cdot V(s_{t+1}) | s_t = s ]
```

#### Policy Improvement
Improve the policy by acting greedily with respect to `Q(s, a)`:

```math
\pi '(s) = \arg \max_a Q(s, a)
```

If you alternate between policy evaluation and policy improvement, you get **Policy Iteration**. If you combine both steps into one, you get **Value Iteration**.
### On Policy Learning
On-policy learning refers to reinforcement learning methods where the agent **learns about the policy it is currently following**.
### Monte Carlo Learning
Monte Carlo (MC) methods are **model-free reinforcement learning algorithms** that learn from **complete episodes of experience**. They do **not use bootstrapping or the Bellman equation**. Instead, they update value estimates using the **actual returns** observed after visiting a state or state-action pair.

####  Policy Evaluation
To estimate the action-value function `Q(s, a)`, the algorithm maintains:
- A total count `N(s, a)`.
- A cumulative return for each `(s, a)`
- The estimate is updated as:

```math
Q(s, a) \leftarrow Q(s, a) + \frac{1}{N(s, a)} \cdot [ G_t - Q(s, a) ]
```
In Monte-Carlo,
```math
 G_t = R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} + \gamma^3 \cdot R_{t+3} + \dots
```
#### Policy Improvement

Once `Q(s, a)` estimates are updated from experience, the policy can be improved using an **ε-greedy strategy** to balance **exploration** and **exploitation**:

```math
\pi(s) = 
\begin{cases}
\arg\max_a Q(s, a) & \rightarrow \text{with probability } 1 - \epsilon  \quad \text{(greedy action)} \\
\text{random action from action space}  & \rightarrow \text{with probability } \epsilon 
\end{cases}
```

- `ε ∈ [0, 1]` controls the exploration rate.
- A small `ε` (e.g., 0.1) encourages mostly greedy actions with occasional exploration.
- This prevents the agent from getting stuck in suboptimal policies.

> ε-greedy is often used in on-policy methods like **SARSA**.

#### Variants

- **First-visit Monte Carlo**: Updates `N(s, a)` only on the first time `(s, a)` is visited in an episode
- **Every-visit Monte Carlo**: Updates `N(s, a)` on **every** occurrence of `(s, a)` in the episode

> These methods require complete episodes and are suitable for **episodic tasks**.
### Temporal Difference Learning (SARSA)
#### Temporal Difference (0) – TD(0)
TD(0) methods are **model-free reinforcement learning algorithms** that learn from **each step** in an episode — they do **not need to wait for the episode to end**. Unlike Monte Carlo methods, TD(0) **does use bootstrapping and the Bellman equation** for learning.
#### Policy Evaluation
To estimate the **action-value function** `Q(s, a)`, the TD(0) update rule is:
```math
Q(s, a) \leftarrow Q(s, a) + \alpha \cdot [ r + \gamma \cdot Q(s', a') - Q(s, a) ]
```

Where:
- `α` is the **learning rate** (0 < α ≤ 1)
- `r` is the reward received after taking action `a` in state `s`
- `s'` is the next state
- `a'` is the next action (in SARSA, the on-policy version)
- `γ` is the discount factor

>  This is a **one-step update**, using the current reward and the estimate of the next state's value.
#### Policy Improvement
Once `Q(s, a)` estimates are updated from experience, the policy can be improved using an **ε-greedy strategy** to balance **exploration** and **exploitation**:
```math
\pi(s) = 
\begin{cases}
\arg\max_a Q(s, a) & \rightarrow \text{with probability } 1 - \epsilon  \quad \text{(greedy action)} \\
\text{random action from action space}  & \rightarrow \text{with probability } \epsilon 
\end{cases}
```

- `ε ∈ [0, 1]` controls the exploration rate.
- A small `ε` (e.g., 0.1) encourages mostly greedy actions with occasional exploration.
- This prevents the agent from getting stuck in suboptimal policies.


#### Temporal Difference (λ) – TD(λ)
TD(λ) is a generalization of TD(0) that blends **bootstrapping** with **Monte Carlo** using a parameter `λ ∈ [0, 1]`.
- `λ = 0` → behaves like TD(0)
- `λ = 1` → behaves like Monte Carlo
- Intermediate values allow for a trade-off between bias and variance
TD(λ) uses **eligibility traces** to assign credit to recently visited state-action pairs over time.
####  TD(λ): Forward View

The forward view is a **theoretical formulation** that defines the return as a **weighted average of n-step returns**:

```math
G^1 = r_t + \gamma \cdot Q(s_{t+1}, a_{t+1})
```
```math
G^2 = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot Q(s_{t+2}, a_{t+2})
```
```math
G^3 = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \gamma^3 \cdot Q(s_{t+3}, a_{t+3})
```
The λ-return is then given by:

```math
G_\lambda = (1 - \lambda) \cdot G^1 + \lambda(1 - \lambda) \cdot G^2 + \lambda^2(1 - \lambda) \cdot G^3 + \dots + \lambda_{n-1}(1 - \lambda) \cdot G^n + \dots
     = (1 - \lambda) \sum_{ₙ₌₁}^\infty \lambda^{n-1} \cdot G^n
```
```math
Q(s, a) \leftarrow Q(s, a) + \alpha \cdot [ G_\lambda - Q(s, a) ]
```
Where `Gⁿ` is the n-step return.


####  TD(λ): Backward View

The backward view is the **practical implementation** used in online algorithms. It uses **eligibility traces** to give credit to previously visited (s, a) pairs.

Update rule:

```math
Q(s, a) \leftarrow Q(s, a) + \alpha · \delta_t · e(s, a)
```

Where:
- `δₜ = r + γ · Q(s', a') - Q(s, a)` is the TD error
- `e(s, a)` is the **eligibility trace**, updated as:
  ```math
  e(s, a) \leftarrow \gamma \cdot \lambda \cdot e(s, a)
  ```

This approach allows the algorithm to assign partial credit to all past actions, decaying over time based on λ and γ.

---
### Off Policy Learning
Off-policy learning refers to reinforcement learning methods where the agent **learns about the policy it is not currently following**. It maybe following an arbitrary initial policy.
#### Q-Learning
Q-learning is a classic off-policy algorithm that learns the **optimal action-value function** `Q*(s, a)` regardless of the agent's behavior. It uses the following update rule:
```text
Q(s, a) ← Q(s, a) + α · [ r + γ · maxₐ' Q(s', a') - Q(s, a) ]
```

Where:
- `α` is the learning rate
- `r` is the reward received
- `s'` is the next state
- `maxₐ' Q(s', a')` is the **greedy action** in the next state
- `γ` is the discount factor

Unlike SARSA, Q-learning does **not use the action actually taken** in the next state — it uses the **greedy action**, making it **off-policy**.

>  Q-learning converges to the optimal policy even if the agent behaves randomly (as long as all state-action pairs are explored sufficiently).

### Policy Gradient Methods
Earlier we found optima policy by optimising value function. This is the method when we directly parametrize policy and optimise it. It has better convergence properties, can learn stochastic policies. But it has a disadvantage that it has high varience when evaluating a policy.
#### REINFORCE Algorithm
REINFORCE is a fundamental **policy gradient algorithm** that optimizes a parameterized policy using sampled returns from complete episodes. The objective is to **maximize the expected return**:
```math
J(\theta) = \mathbb{E}_{\pi_\theta} [ G_t ]
```
J(θ) is the policy objective function. θ is the parameter of the policy.
Using the **likelihood ratio trick** ```(multiply and divide by the policy)``` , the gradient of the objective becomes:

```math
\nabla \pi_\theta(a_t \mid s_t) = \left( \frac{\nabla \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)} \right) \cdot \pi_\theta(a_t \mid s_t) = \nabla_\theta \log \left(\pi_\theta(a_t \mid s_t) \right) \cdot \pi_\theta(a_t \mid s_t)
```
```math
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \left(\pi_\theta(a_t \mid s_t)\right) \cdot G_t \right]
```
Then we adjust the parameter θ to improve the policy
Where:
- `θ` are the parameters of the policy network
- `π_θ(aₜ | sₜ)` is the probability of taking action `aₜ` in state `sₜ`
- `Gₜ` is the return (sum of discounted future rewards from time `t`)

> This formula says: **increase the probability of actions that led to higher returns**.
### Actor Critic
It is a method for policy optimisation. It has two parts:
- **Critic**: Updates action value parameters w
- **Actor**: Updates policy parameters θ, as per the direction suggested by critic
### Model Based RL
Model is a representation of MDP using a parameter ɳ. We will assume State space and Action space as already known and representtion state is M = <P_ɳ,R_ɳ>.

```math
S_{t+1} \approx P_\eta(S_{t+1}|S_t,A_t)
```
```math
R_{t+1} = R_\eta(R_{t+1}|S_t,A_t)
```
One of the Model-based algorithm is **Dyna-Q** algorithm
#### Dyna Q algorithm
Pseudocode:
```text
Initialise Q(s,a) and Model(s,a) for all s,a ∈ S,A
Do Forever:
  S ← current state
  A ← ∈-greedy(S,Q)
  Execute Action A; Observe resultant reward R ans state s'
  Q(s, a) ← Q(s, a) + α · [ R + γ · maxₐ' Q(s', a) - Q(s, a) ]
  Model(S,A) ← R,s' (Assuming deterministic policy)
  repeat n times:
    S ← random previously observed state
    A ← random action previously taken in S
    R,s' ← Model(S,A) ←
    Q(s, a) ← Q(s, a) + α · [ R + γ · maxₐ' Q(s', a) - Q(s, a) ]
    
```
### Multi-Armed Bandit
The **Multi-Armed Bandit (MAB)** problem is a foundational setting in reinforcement learning that models the trade-off between **exploration** and **exploitation**.

Imagine a slot machine (a "bandit") with multiple arms, each providing a different (unknown) reward distribution.  
The agent’s goal is to **maximize cumulative reward** over time by choosing which arms to pull. 

**Regret** : Opportunity loss for one step. 
```math
I_t = \mathbb{E}[v^* − q(A_t)]

```
> **So, Maximise cumulative value = Minimise cumulative regret**

There is:
- **No state transition**
- Only one state and repeated actions
- The challenge is to **learn the best arm** through trial and error


### Upper Confidence Bounds
 - Estimate an upper confidence bound Uₜ(a*) for each actions.
 - Such that, q(a) ≤ Qₜ(a) + Uₜ(a) at higher probability.
*In simple words, they tried the actions which has more probability to be optimal action.*

### UCB 1 algorithm
The UCB1 algorithm selects the action 'Aₜ' a that maximizes the sum of its estimated value and an exploration bonus:
```math
A_t = arg max_a Q(a) + \sqrt{-log p/ 2N_t(a)}
\text{  where  } Uₜ = \sqrt{-log p/ 2Nₜ(a)}
```
> This is obtaned by Hoeffding's inequality

## Week 4
### Implementing Q-Learning using Pytorch / Tensorflow
Pytorch and Tensorflow are two python modules which is widely used to implement deep-learning framework. These modules provide robust tools for learning methods for implementing learning methods. Using PyTorch, we gain full control over the learning process due to its dynamic computation graph and intuitive debugging capabilities. In contrast, TensorFlow—particularly in its high-level APIs like Keras—offers a more abstracted interface, which simplifies implementation but provides comparatively less flexibility for low-level customization. So pytorch is suitable if you want full control over the learnng framework and tensorflow if you want basic learning models. 
So the basic algorithm for Q-table learning :
 - Discretize the observation space.
 - Initialize a Q-table with size [Discretized observation space size, action space cardinality]
 - Training loop:
    * Play the action corresponds to argmax(Q_table[state]) with a probability (1 - ε), or a random action with a probability, ε (ε decreases over episodes)
    * Using the reward obtained, update Q_table[state], with ```r + γ*Q_table[state]```
    * And repeat the process for many episodes
As part of the assignment created a Q-table learning framework using pytorch for the snake game developed in first week assignment. The file [Q-Learning.py](./Q-Learning/Q-learning.py) is in the folder [Q-Learning](./Q-Learning/)
## Week 5
### Deep Q-Learning (DQN)
Here, we use neural networks to determine the acion tha has to be done. This was developed because fr complex environments, the cardinality of observation space shoots up essentially making q-table to be of huge size. So to reduce computatonal costs, we could use a neural network, where input layer is observation space and output layer is Q_values for the action space. We can either use a covolutional neural network for the input layer and after 1-2 layers, flatten the layer and  use linear neural network for subsequent layers. And for exploration, we follow ε-greedy strategy, where ε decreases over episodes. Here the updation per step is too small compared to Q-table learning so we are adjusting based on multiple timesteps at the same time. For this we will create a replay buffer, ```R = [s, a, s', r]```, where
 - ```s is current state.```
 - ```a is action we took at state s.```
 - ```s' is state we reached after playing this action.```
 - ```r is reward obtained.```
So after a point when cardinality of Replay buffer > n, we sample a subset from this buffer which exactly has n random elements from this buffer, and we train the neural network using this random subset of buffer. This is so much better than training the neural network with just one element of this buffer.
As part of the assignment created a Deep Q learning framework using pytorch for the snake game developed in first week assignment. The file [DQN.py](./Q-Learning/DQN.py) is in the folder [Q-Learning](./Q-Learning/). Also created a report [QandDQN.pdf](./Q-Learning/QandDQN.pdf), which compares the classical Q-table learning approach from the previous assignment with the Deep Q-Network framework introduced in this one.

## Week 6
### Policy Gradient Methods
#### 1. Trust Region Policy Optimization (TRPO)
TRPO is a policy gradient RL method. Here we uses neural network to determine the policy.
The algorithm is:
 - We will define a neural network whose input layer is the observation space and there is 2 output layer (which we make parallel), one of them gives the value functions and other gives the mean of policies at that state. (We must have already defined standard deviation as a hyperparameter of this neural network, so it will update during training)
>Assumption: Policy is a normal distribution with this mean and variance
>π(a|s) ~ N(μ(s), σ²)
 - We would take a sample from this distribution and play that action. The outputa are stored in a buffer and a set of this goes to neural network to train(like DQN)
 - The loss function here is
```math
  L = r_t \times A_t \quad \text{where,  } r_t = \frac{\pi_{new}(a_t|s_t)}{\pi_{old}(a_t|s_t)} \quad  \text{ Subject to the constraint   } \mathbb{E}_t [ KL[ π_{θ_{old}}(·|s_t) || π_{θ_{new}}(·|s_t) ] ] ≤ \delta
```
> KL divergence of two normal distributions is defined as:
```math
D_{\text{KL}}\left( \mathcal{N}_{\text{old}} \,||\, \mathcal{N}_{\text{new}} \right) = \sum_{i=1}^d \left[\log \left( \frac{\sigma_{\text{new},i}}{\sigma_{\text{old},i}} \right) + \frac{\sigma_{\text{old},i}^2 + \left( \mu_{\text{old},i} - \mu_{\text{new},i} \right)^2}{2 \sigma_{\text{new},i}^2}- \frac{1}{2} \right]
```
#### 2. Proximal Policy Optimization (PPO)
PPO is also a policy gradient RL method like TRPO. It was made to simplify the policy gradient RL method. In TRPO, we're maximizing based on a constraint and in PPO we could clip the loss and reduce computation. The algorithm is similar except the loss function. We will define ε and clip r_t between 1-ε and 1+ε. So now new policy does not deviate too much from old policy. Mathematically
```math
L = r_t \times A_t \quad \text{where,  } r_t = clip(1 - \epsilon, \frac{\pi_{new}(a_t|s_t)}{\pi_{old}(a_t|s_t)}, 1+ \epsilon)
```

```math
A_t \text{ in both PPO and TRPO is the advantage function it tells about how much advantage it is to perform that particualr action in this state.}
```
```math
\text{In practice, especially in PPO and TRPO, it is often estimated using Generalized Advantage Estimation (GAE), where }
```
```math
A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots,
```
```math
 \text{with } \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```
## Week 7 and Week 8
### Final project
Trained a PPO  agent to solve the CarRacing-v3 environment using visual input. The goal is to develop an agent that can complete laps efficiently using image-based reinforcement learning. 

CarRacing-v3 is an environment developed by OpenAI as part of its Gym toolkit, designed to benchmark and standardize reinforcement learning (RL) methods (like Q-learning, DQN, PPO, etc). Prior to such environments, there was no consistent framework for evaluating RL methods. OpenAI’s standardized environments like CarRacing-v3, CartPole,etc... have played a crucial role in making RL research reproducible, comparable, and scalable.

I used stable baselines module mainly to create this agent. The code for the same, a brief report and some training videos at steps 2000, 60000, 80000 and 100000 are in the [PPO](./PPO/) directory.


