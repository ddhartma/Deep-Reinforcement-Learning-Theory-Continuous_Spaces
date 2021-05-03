[image1]: assets/gridworld.png "image1"
[image2]: assets/rl_overview.png "image2"
[image3]: assets/discrete_spaces.png "image3"
[image4]: assets/continuous_spaces.png "image4"
[image5]: assets/non_uniform_discretization.png "image5"
[image6]: assets/scores_plt.png "image6"
[image7]: assets/scores_plt_test.png "image7"


# Deep Reinforcement Learning Theory - RL in Continuous Spaces

## Content
- [Introduction](#intro)
- [Problem analysis](#problem_analysis)
- [Discrete vs. Continuous Spaces](#discrete_cont)
- [Discretization](#discretization)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)


## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

## Problem analysis <a name="Setup_Instructions"></a>
- So far all reinforcement learning environments were implemented where the number of states and actions is limited. 
- With small, **finite Markov Decision Processes (MDPs)**, it is possible to represent the **action-value function** with a **table**, **dictionary**, or other finite structure.

- Gridworld: Say the world has 
    - four possible states, 
    - and the agent has four possible actions at its disposal (up, down, left, right).  
    - The estimated optimal action-value function in a table, with a row for each state and a column for each action. --> Q-table.

    ![image1]

    ![image2]

## Discrete vs. Continuous Spaces <a name="discrete_cont"></a>
### Discrete Spaces
- States and actions as a **dictionary** or **look-up table**.
- Examples: 
    - **value function V** as a **mapping** from the **set of states** to a **real number**.
    -  **the action value function Q** as s **mapping** of a **state action pair** to a **real number**.
    - **value iteration**, the internal for loop goes over each state as one by one, and updates the corresponding value estimate V of s. 
    - **Model-free methods** like Q-learning assume discrete spaces as well.

    ![image3]

### Continuous Spaces <a name="continuous_spaces"></a>
- Discrete visualization of states: real number, bar chart
- Continuous visualization of states: vector, density plot  
- Dealing with contnuous spaces: **Discretization** and **Function Approximation**

    ![image4]

## Discretization <a name="discretization"></a> 
- Discretization converts a **continuous space into a discrete one**.
- States and actions can be discretized
- See standard gridworld with obstacles below. Agent could think,
there is no path across these obstacles.
- **Non-Uniform** Discretization: Vary the grid according to these obstacles, then a feasible path for the agent is possible.
- An alternate approach would be to divide up the grid into **smaller cells where required**.
    ![image5]

### Implementation
- Open Jupyter Notebook ```discretization.ipynb```
    ```
    import sys
    import gym
    import numpy as np

    import pandas as pd
    import matplotlib.pyplot as plt

    # Set plotting options
    %matplotlib inline
    plt.style.use('ggplot')
    np.set_printoptions(precision=3, linewidth=120)

    !python -m pip install pyvirtualdisplay
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    ```
    ### Create an environment
    ```
    # Create an environment and set random seed
    env = gym.make('MountainCar-v0')
    env.seed(505); 

    # Explore state (observation) space
    # [Position Velocity]
    print("State space:", env.observation_space)
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)

    RESULT:
    State space: Box(2,)
    - low: [-1.2  -0.07]
    - high: [ 0.6   0.07]
    ```
    ### Discretize the State Space with a Uniform Grid
    ```
    def create_uniform_grid(low, high, bins=(10, 10)):
        """ Define a uniformly-spaced grid that can be used to discretize a space.
        
        INPUTS:
        ------------
            low - (array_like) lower bounds for each dimension of the continuous space.
            high - (array_like) upper bounds for each dimension of the continuous space.
            bins - (tuple) number of bins along each corresponding dimension.
        
        OUTPUTS:
        ------------
            grid - (list of array_like) list of arrays containing split points for each dimension.
        """
        grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
        print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
        for l, h, b, splits in zip(low, high, bins, grid):
            print("    [{}, {}] / {} => {}".format(l, h, b, splits))
        return grid
        

    low = [-1.0, -5.0]
    high = [1.0, 5.0]
    create_uniform_grid(low, high)  # [test]
    
    RESULT:
    Uniform grid: [<low>, <high>] / <bins> => <splits>
        [-1.0, 1.0] / 10 => [-0.8 -0.6 -0.4 -0.2  0.   0.2  0.4  0.6  0.8]
        [-5.0, 5.0] / 10 => [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
    ```
    ### Discretize samples 
    ```
    def discretize(sample, grid):
        """ Discretize a sample as per given grid.
        
        INPUTS:
        ------------
            sample - (array_like) a single sample from the (original) continuous space.
            grid - (list of array_like) list of arrays containing split points for each dimension.
        
        OUTPUTS:
        ------------
            discretized_sample - (array_like) sequence of integers with the same number of dimensions as sample.
        """
        
        # numpy.digitize returns the indices of g to which each value in s belongs.
        return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


    # Test with a simple grid and some samples
    grid = create_uniform_grid([-1.0, -5.0], [1.0, 5.0])
    samples = np.array(
        [[-1.0 , -5.0],
        [-0.81, -4.1],
        [-0.8 , -4.0],
        [-0.5 ,  0.0],
        [ 0.2 , -1.9],
        [ 0.8 ,  4.0],
        [ 0.81,  4.1],
        [ 1.0 ,  5.0]])
    discretized_samples = np.array([discretize(sample, grid) for sample in samples])
    print("\nSamples:", repr(samples), sep="\n")
    print("\nDiscretized samples:", repr(discretized_samples), sep="\n")

    RESULT:
    Uniform grid: [<low>, <high>] / <bins> => <splits>
        [-1.0, 1.0] / 10 => [-0.8 -0.6 -0.4 -0.2  0.   0.2  0.4  0.6  0.8]
        [-5.0, 5.0] / 10 => [-4. -3. -2. -1.  0.  1.  2.  3.  4.]

    Samples:
    array([[-1.  , -5.  ],
        [-0.81, -4.1 ],
        [-0.8 , -4.  ],
        [-0.5 ,  0.  ],
        [ 0.2 , -1.9 ],
        [ 0.8 ,  4.  ],
        [ 0.81,  4.1 ],
        [ 1.  ,  5.  ]])

    Discretized samples:
    array([[0, 0],
        [0, 0],
        [1, 1],
        [2, 5],
        [5, 3],
        [9, 9],
        [9, 9],
        [9, 9]])
    ```
    ### Create am Q-Learning Agent
    ```
    class QLearningAgent:
        """ Q-Learning agent that can act on a continuous state space by discretizing it.
        """

        def __init__(self, env, state_grid, alpha=0.02, gamma=0.99,
                    epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
            """ Initialize variables, create grid for discretization.
            
                INPUTS:
                ------------
                    env - (OpenAI gym instance) instance of an OpenAI Gym environment
                    state_grid - (list of two numpy arrays) array1: position, array2: velocity, 
                                discretized the State Space with a Uniform Grid
                    alpha- (float) step-size parameter for the update step (constant alpha concept), default=0.02
                    gamma - (float) discount rate. It must be a value between 0 and 1, inclusive, default=0.99
                    epsilon - (float) probability with which the agent selects an action uniformly at random
                    epsilon_decay_rate - (float) decay rate for epsilon, default=0.9995
                    min_epsilon - (float) min for epsilon, default=.01
                    seed - (int) seed for random, default=505
                
                OUTPUTS:
                ------------
                None
            
            """
            # Environment info
            self.env = env
            self.state_grid = state_grid
            self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
            self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
            self.seed = np.random.seed(seed)
            print("Environment:", self.env)
            print("State space size:", self.state_size)
            print("Action space size:", self.action_size)
            
            # Learning parameters
            self.alpha = alpha  # learning rate
            self.gamma = gamma  # discount factor
            self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
            self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
            self.min_epsilon = min_epsilon
            
            # Create Q-table
            self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
            print("Q table size:", self.q_table.shape)

        def preprocess_state(self, state):
            """ Map a continuous state to its discretized representation.
            
                INPUTS:
                ------------
                    state - (1D numpy array) state[0] - position, state[1] - velocity, continuous entries
                
                OUTPUTS:
                ------------
                    discretized_state - (tuple) discretized version of state using np.digitize
            """
            print(state)
            discretized_state = tuple(discretize(state, self.state_grid))
            return discretized_state

        def reset_episode(self, state):
            """ Reset variables for a new episode.
            
                INPUTS:
                ------------
                    state - (1D numpy array) state[0] - position, state[1] - velocity, continuous entries
                
                OUTPUTS:
                ------------
                    self.last_action - (int) number for certain action
            
            """
            # Gradually decrease exploration rate
            self.epsilon *= self.epsilon_decay_rate
            self.epsilon = max(self.epsilon, self.min_epsilon)

            # Decide initial action
            self.last_state = self.preprocess_state(state)
            self.last_action = np.argmax(self.q_table[self.last_state])
            return self.last_action
        
        def reset_exploration(self, epsilon=None):
            """ Reset exploration rate used when training.
            
                INPUTS:
                ------------
                    epsilon - (float) probability with which the agent selects an action uniformly at random
                
                OUTPUTS:
                ------------
                    no direct
                    self.epsilon - (float) reset epsilon if epsilon is not None 
                
            """
            self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

        def act(self, state, reward=None, done=None, mode='train'):
            """ Pick next action and update internal Q table (when mode != 'test').
            
                INPUTS:
                ------------
                    state - (1D numpy array) state[0] - position, state[1] - velocity
                    reward - (float) rewrd for next step to update Q-table
                    done - (bool) if True episode is over, default=None 
                    mode - (string) 'train' or 'test'
                
                OUTPUTS:
                ------------
                    action - (int) based on Sarsamax return corresponding action
                
            """
            state = self.preprocess_state(state)
            if mode == 'test':
                # Test mode: Simply produce an action
                action = np.argmax(self.q_table[state])
            else:
                # Train mode (default): Update Q table, pick next action
                # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
                self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                    (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])

                # Exploration vs. exploitation
                do_exploration = np.random.uniform(0, 1) < self.epsilon
                if do_exploration:
                    # Pick a random action
                    action = np.random.randint(0, self.action_size)
                else:
                    # Pick the best action from Q table
                    action = np.argmax(self.q_table[state])

            # Roll over current state, action for next step
            self.last_state = state
            self.last_action = action
            return action

        
    q_agent = QLearningAgent(env, state_grid)

    RESULT:
    Environment: <TimeLimit<MountainCarEnv<MountainCar-v0>>>
    State space size: (10, 10)
    Action space size: 3
    Q table size: (10, 10, 3)
    ```
    ### Start the Learning Procedure
    ```
    def run(agent, env, num_episodes=20000, mode='train'):
        """ Run agent in given reinforcement learning environment and return scores.
            
            INPUTS:
            ------------
                agent - (instance of class QLearningAgent) 
                env - (OpenAI gym instance) instance of an OpenAI Gym environment
                num_episodes - (int) number of episodes
                mode - (string) - mode train or test
            
            OUTPUTS:
            ------------
                scores - (list) list of total reward for each episode
        
        """
        scores = []
        max_avg_score = -np.inf
        for i_episode in range(1, num_episodes+1):
            # Initialize episode
            state = env.reset()
            action = agent.reset_episode(state)
            total_reward = 0
            done = False

            # Roll out steps until done
            while not done:
                state, reward, done, info = env.step(action)
                total_reward += reward
                action = agent.act(state, reward, done, mode)

            # Save final score
            scores.append(total_reward)
            
            # Print episode stats
            if mode == 'train':
                if len(scores) > 100:
                    avg_score = np.mean(scores[-100:])
                    if avg_score > max_avg_score:
                        max_avg_score = avg_score

                if i_episode % 100 == 0:
                    print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                    sys.stdout.flush()

        return scores

    scores = run(q_agent, env)

    RESULT:
    Episode 20000/20000 | Max Average Score: -131.87
    ```
    ### Visualize data 
    ```
    def plot_scores(scores, rolling_window=100):
        """ Plot scores and optional rolling mean using specified window.
            
            INPUTS:
            ------------
                scores - (list) list of total reward for each episode
                rolling_window - (int)
            
            OUTPUTS:
            ------------
                rolling_mean - (pandas Series) rolling mean of scores
        """
        plt.plot(scores); plt.title("Scores");
        rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
        plt.plot(rolling_mean);
        return rolling_mean

    rolling_mean = plot_scores(scores)
    ```
    ![image6]
    ### Run in test mode and analyze scores obtained
    ```
    # Run in test mode and analyze scores obtained
    test_scores = run(q_agent, env, num_episodes=100, mode='test')
    print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
    _ = plot_scores(test_scores, rolling_window=10)
    ```
    ![image7]





## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)