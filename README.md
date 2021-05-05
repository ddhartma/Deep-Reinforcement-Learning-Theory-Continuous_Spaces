[image1]: assets/gridworld.png "image1"
[image2]: assets/rl_overview.png "image2"
[image3]: assets/discrete_spaces.png "image3"
[image4]: assets/continuous_spaces.png "image4"
[image5]: assets/non_uniform_discretization.png "image5"
[image6]: assets/scores_plt.png "image6"
[image7]: assets/scores_plt_test.png "image7"
[image8]: assets/tile_coding.png "image8"
[image9]: assets/coarse_coding.png "image9"
[image10]: assets/function_approximation.png "image10"
[image11]: assets/tile_coding_plot.png "image11"
[image12]: assets/gradient_descent.png "image12"
[image13]: assets/action_vec_approx.png "image13"
[image14]: assets/kernel_func.png "image14"
[image15]:  assets/non_lin_func_approx.png "image15"

# Deep Reinforcement Learning Theory - RL in Continuous Spaces

## Content
- [Introduction](#intro)
- [Problem analysis](#problem_analysis)
- [Discrete vs. Continuous Spaces](#discrete_cont)
- [Discretization](#discretization)
- [Tile Coding](#tile_coding)
- [Coarse Coding](#coarse_coding)
- [Function Approximation](#function_approximation)
- [Linear Function Approximation](#lin_func_approx)
- [Kernel Functions](#kernel_functions)
- [Nonlinear Function approximation](#nonlin_func_approx)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

- Real world problems are normally continuous
- In order to handle those spaces to mechanisms are usefuls
    - **Discretize** continuous spaces
    - Directly try to approximate desired value functions (**Function approximation** of state-value and action-value functions) 
        - feature transformation
        - non-linear feature transforms like radial basis functions
        - non-linear combinations of features, apply an activation function (neural networks)

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


## Tile Coding <a name="tile_coding"></a>
- overlay multiple grids or tilings on top of the space,
each slightly offset from each other.
- Now, any position S in the state space can be
coarsely identified by the tiles that it activates.
- The tile coding algorithm in turn updates these weights iteratively.
- This ensures nearby locations that share tiles also share some component of state value, effectively smoothing the learned value function.
- Better: Adaptive tile coding, which starts with fairly large tiles, and divides each tile into two whenever appropriate.
- Split tile when agent does no longer learn much with the current representation (value function isn't changing).
- Stop when some upper limit on the number of splits or some max iterations are reached
- Tile to split: the one with the greatest effect on the value function. For this, we need to keep track of subtiles and their projected weights. Then, we can pick the tile with the greatest difference between subtile weights.

    ![image8]
- Open Jupyter notebook ```tile_coding.ipynb```
    ```
    # Import common libraries
    import sys
    import gym
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Set plotting options
    %matplotlib inline
    plt.style.use('ggplot')
    np.set_printoptions(precision=3, linewidth=120)
    ```
    ### Create an environment
    ```
    # Create an environment
    env = gym.make('Acrobot-v1')
    env.seed(505);

    # Explore state (observation) space
    print("State space:", env.observation_space)
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)

    # Explore action space
    print("Action space:", env.action_space)
    ```
    ### Tiling
    ```
    def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
        """Define a uniformly-spaced grid that can be used for tile-coding a space.
        
        Inputs:
        ------------
            low - (array_like) lower bounds for each dimension of the continuous space.
            high - (array_like) upper bounds for each dimension of the continuous space.
            bins - (tuple) number of bins along each corresponding dimension.
            offsets - (tuple) split points for each dimension should be offset by these values.
        
        Outputs:
        ------------
            grid - (list of array_like) list of arrays containing split points for each dimension.
        """
        
        grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(bins))]
        print("Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>")
        for l, h, b, o, splits in zip(low, high, bins, offsets, grid):
            print("    [{}, {}] / {} + ({}) => {}".format(l, h, b, o, splits))
        return grid


    low = [-1.0, -5.0]
    high = [1.0, 5.0]
    create_tiling_grid(low, high, bins=(10, 10), offsets=(-0.1, 0.5))  # [test]

    RESULT:
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 10 + (-0.1) => [-0.9 -0.7 -0.5 -0.3 -0.1  0.1  0.3  0.5  0.7]
        [-5.0, 5.0] / 10 + (0.5) => [-3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5]

    [array([-0.9, -0.7, -0.5, -0.3, -0.1,  0.1,  0.3,  0.5,  0.7]),
    array([-3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5,  4.5])]
    ```
    ```
    def create_tilings(low, high, tiling_specs):
        """Define multiple tilings using the provided specifications.

        INPUTS:
        ------------
            low - (array_like) lower bounds for each dimension of the continuous space.
            high - (array_like) upper bounds for each dimension of the continuous space.
            
        OUTPUTS:
        ------------
            tilings - (list) list of tilings (grids), each produced by create_tiling_grid().
        """
        
        return [create_tiling_grid(low, high, bins, offsets) for bins, offsets in tiling_specs]


    # Tiling specs: [(<bins>, <offsets>), ...]
    tiling_specs = [((10, 10), (-0.066, -0.33)),
                    ((10, 10), (0.0, 0.0)),
                    ((10, 10), (0.066, 0.33))]
    tilings = create_tilings(low, high, tiling_specs)

    RESULT:
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 10 + (-0.066) => [-0.866 -0.666 -0.466 -0.266 -0.066  0.134  0.334  0.534  0.734]
        [-5.0, 5.0] / 10 + (-0.33) => [-4.33 -3.33 -2.33 -1.33 -0.33  0.67  1.67  2.67  3.67]
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 10 + (0.0) => [-0.8 -0.6 -0.4 -0.2  0.   0.2  0.4  0.6  0.8]
        [-5.0, 5.0] / 10 + (0.0) => [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 10 + (0.066) => [-0.734 -0.534 -0.334 -0.134  0.066  0.266  0.466  0.666  0.866]
        [-5.0, 5.0] / 10 + (0.33) => [-3.67 -2.67 -1.67 -0.67  0.33  1.33  2.33  3.33  4.33]
    ```
    ### Discretize
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
        
        return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


    def tile_encode(sample, tilings, flatten=False):
        """Encode given sample using tile-coding.
        
        INPUTS:
        ------------
            sample - (array_like) a single sample from the (original) continuous space.
            tilings - (list) list of tilings (grids), each produced by create_tiling_grid().
            flatten - (bool) If true, flatten the resulting binary arrays into a single long vector.

        OUTPUTS:
        ------------
            encoded_sample -(list or array_like) list of binary vectors, one for each tiling, or flattened into one.
        """
        
        encoded_sample = [discretize(sample, grid) for grid in tilings]
        return np.concatenate(encoded_sample) if flatten else encoded_sample


    # Test with some sample values
    samples = [(-1.2 , -5.1 ),
            (-0.75,  3.25),
            (-0.5 ,  0.0 ),
            ( 0.25, -1.9 ),
            ( 0.15, -1.75),
            ( 0.75,  2.5 ),
            ( 0.7 , -3.7 ),
            ( 1.0 ,  5.0 )]
    encoded_samples = [tile_encode(sample, tilings) for sample in samples]
    print("\nSamples:", repr(samples), sep="\n")
    print("\nEncoded samples:", repr(encoded_samples), sep="\n")

    RESULT:
    Samples:
    [(-1.2, -5.1), (-0.75, 3.25), (-0.5, 0.0), (0.25, -1.9), (0.15, -1.75), (0.75, 2.5), (0.7, -3.7), (1.0, 5.0)]

    Encoded samples:
    [[(0, 0), (0, 0), (0, 0)], [(1, 8), (1, 8), (0, 7)], [(2, 5), (2, 5), (2, 4)], [(6, 3), (6, 3), (5, 2)], [(6, 3), (5, 3), (5, 2)], [(9, 7), (8, 7), (8, 7)], [(8, 1), (8, 1), (8, 0)], [(9, 9), (9, 9), (9, 9)]]
    ```
    ### Q-Table
    ```
    class QTable:
        """ Simple Q-table
        """

        def __init__(self, state_size, action_size):
            """ Initialize Q-table.
            
            INPUTS:
            ----------
                state_size - (tuple) Number of discrete values along each dimension of state space.
                action_size - (int) Number of discrete actions in action space.
            """
            self.state_size = state_size
            self.action_size = action_size

            # Create Q-table, initialize all Q-values to zero
            # Note: If state_size = (9, 9), action_size = 2, q_table.shape should be (9, 9, 2)
            self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
            print("QTable(): size =", self.q_table.shape)


    class TiledQTable:
        """ Composite Q-table with an internal tile coding scheme
        """
        
        def __init__(self, low, high, tiling_specs, action_size):
            """ Create tilings and initialize internal Q-table(s).
            
            INPUTS:
            ------------
                low - (array_like) lower bounds for each dimension of the continuous space.
                high - (array_like) upper bounds for each dimension of the continuous space.
                tiling_specs - (list of tuples) sequence of (bins, offsets) to be passed to create_tilings() along with low, high.
                action_size - (int) Number of discrete actions in action space.
                
            OUTPUTS:
            ------------
                None
            """
            self.tilings = create_tilings(low, high, tiling_specs)
            self.state_sizes = [tuple(len(splits)+1 for splits in tiling_grid) for tiling_grid in self.tilings]
            self.action_size = action_size
            self.q_tables = [QTable(state_size, self.action_size) for state_size in self.state_sizes]
            print("TiledQTable(): no. of internal tables = ", len(self.q_tables))
        
        def get(self, state, action):
            """ Get Q-value for given <state, action> pair.
            
            INPUTS:
            ----------
                state - (array_like) Vector representing the state in the original continuous space.
                action - (int) Index of desired action.
            
            OUTPUTS:
            -------
                value - (float) Q-value of given <state, action> pair, averaged from all internal Q-tables.
            """
            # Encode state to get tile indices
            encoded_state = tile_encode(state, self.tilings)
            
            # Retrieve q-value for each tiling, and return their average
            value = 0.0
            for idx, q_table in zip(encoded_state, self.q_tables):
                value += q_table.q_table[tuple(idx + (action,))]
            value /= len(self.q_tables)
            return value
        
        def update(self, state, action, value, alpha=0.1):
            """ Soft-update Q-value for given <state, action> pair to value.
            
                Instead of overwriting Q(state, action) with value, perform soft-update:
                    Q(state, action) = alpha * value + (1.0 - alpha) * Q(state, action)
            
            INPUTS:
            ----------
                state - (array_like) Vector representing the state in the original continuous space.
                action - (int) Index of desired action.
                value - (float) desired Q-value for <state, action> pair.
                alpha  - (float) Update factor to perform soft-update, in [0.0, 1.0] range.
            """
            # Encode state to get tile indices
            encoded_state = tile_encode(state, self.tilings)
            
            # Update q-value for each tiling by update factor alpha
            for idx, q_table in zip(encoded_state, self.q_tables):
                value_ = q_table.q_table[tuple(idx + (action,))]  # current value
                q_table.q_table[tuple(idx + (action,))] = alpha * value + (1.0 - alpha) * value_


    # Test with a sample Q-table
    tq = TiledQTable(low, high, tiling_specs, 2)
    s1 = 3; s2 = 4; a = 0; q = 1.0
    print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value at sample = s1, action = a
    print("[UPDATE] Q({}, {}) = {}".format(samples[s2], a, q)); tq.update(samples[s2], a, q)  # update value for sample with some common tile(s)
    print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value again, should be slightly updated

    RESULT:
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 10 + (-0.066) => [-0.866 -0.666 -0.466 -0.266 -0.066  0.134  0.334  0.534  0.734]
        [-5.0, 5.0] / 10 + (-0.33) => [-4.33 -3.33 -2.33 -1.33 -0.33  0.67  1.67  2.67  3.67]
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 10 + (0.0) => [-0.8 -0.6 -0.4 -0.2  0.   0.2  0.4  0.6  0.8]
        [-5.0, 5.0] / 10 + (0.0) => [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 10 + (0.066) => [-0.734 -0.534 -0.334 -0.134  0.066  0.266  0.466  0.666  0.866]
        [-5.0, 5.0] / 10 + (0.33) => [-3.67 -2.67 -1.67 -0.67  0.33  1.33  2.33  3.33  4.33]
    QTable(): size = (10, 10, 2)
    QTable(): size = (10, 10, 2)
    QTable(): size = (10, 10, 2)
    TiledQTable(): no. of internal tables =  3
    [GET]    Q((0.25, -1.9), 0) = 0.0
    [UPDATE] Q((0.15, -1.75), 0) = 1.0
    [GET]    Q((0.25, -1.9), 0) = 0.06666666666666667
    ```
    ### Q-Learning 
    ```
    class QLearningAgent:
        """ Q-Learning agent that can act on a continuous state space by discretizing it.
        """

        def __init__(self, env, tq, alpha=0.02, gamma=0.99,
                    epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=0):
            """ Initialize variables, create grid for discretization.
            
                INPUTS:
                ------------
                    env - (OpenAI gym instance) instance of an OpenAI Gym environment
                    tq
                    alpha - (float) step-size parameter for the update step (constant alpha concept), default=0.02
                    gamma - (float) discount rate. It must be a value between 0 and 1, inclusive, default=0.99
                    epsilon - (float) probability with which the agent selects an action uniformly at random
                    epsilon_decay_rate - (float) decay rate for epsilon, default=0.9995
                    min_epsilon - (float) min for epsilon, default=.01
                    seed - (int) seed for random, default=0
                
                OUTPUTS:
                ------------
                    None
            """
            # Environment info
            self.env = env
            self.tq = tq 
            self.state_sizes = tq.state_sizes           # list of state sizes for each tiling
            self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
            self.seed = np.random.seed(seed)
            print("Environment:", self.env)
            print("State space sizes:", self.state_sizes)
            print("Action space size:", self.action_size)
            
            # Learning parameters
            self.alpha = alpha  # learning rate
            self.gamma = gamma  # discount factor
            self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
            self.epsilon_decay_rate = epsilon_decay_rate   # how quickly should we decrease epsilon
            self.min_epsilon = min_epsilon

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
            
            self.last_state = state
            Q_s = [self.tq.get(state, action) for action in range(self.action_size)]
            self.last_action = np.argmax(Q_s)
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
            Q_s = [self.tq.get(state, action) for action in range(self.action_size)]
            # Pick the best action from Q table
            greedy_action = np.argmax(Q_s)
            if mode == 'test':
                # Test mode: Simply produce an action
                action = greedy_action
            else:
                # Train mode (default): Update Q table, pick next action
                # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
                value = reward + self.gamma * max(Q_s)
                self.tq.update(self.last_state, self.last_action, value, self.alpha)

                # Exploration vs. exploitation
                do_exploration = np.random.uniform(0, 1) < self.epsilon
                if do_exploration:
                    # Pick a random action
                    action = np.random.randint(0, self.action_size)
                else:
                    # Pick the greedy action
                    action = greedy_action

            # Roll over current state, action for next step
            self.last_state = state
            self.last_action = action
            return action
    ```
    ```
    n_bins = 5
    bins = tuple([n_bins]*env.observation_space.shape[0])
    offset_pos = (env.observation_space.high - env.observation_space.low)/(3*n_bins)

    tiling_specs = [(bins, -offset_pos),
                    (bins, tuple([0.0]*env.observation_space.shape[0])),
                    (bins, offset_pos)]

    tq = TiledQTable(env.observation_space.low, 
                    env.observation_space.high, 
                    tiling_specs, 
                    env.action_space.n)
    agent = QLearningAgent(env, tq)

    RESULT:
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 5 + (-0.13333334028720856) => [-0.733 -0.333  0.067  0.467]
        [-1.0, 1.0] / 5 + (-0.13333334028720856) => [-0.733 -0.333  0.067  0.467]
        [-1.0, 1.0] / 5 + (-0.13333334028720856) => [-0.733 -0.333  0.067  0.467]
        [-1.0, 1.0] / 5 + (-0.13333334028720856) => [-0.733 -0.333  0.067  0.467]
        [-12.566370964050293, 12.566370964050293] / 5 + (-1.675516128540039) => [-9.215 -4.189  0.838  5.864]
        [-28.274333953857422, 28.274333953857422] / 5 + (-3.769911289215088) => [-20.735  -9.425   1.885  13.195]
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 5 + (0.0) => [-0.6 -0.2  0.2  0.6]
        [-1.0, 1.0] / 5 + (0.0) => [-0.6 -0.2  0.2  0.6]
        [-1.0, 1.0] / 5 + (0.0) => [-0.6 -0.2  0.2  0.6]
        [-1.0, 1.0] / 5 + (0.0) => [-0.6 -0.2  0.2  0.6]
        [-12.566370964050293, 12.566370964050293] / 5 + (0.0) => [-7.54  -2.513  2.513  7.54 ]
        [-28.274333953857422, 28.274333953857422] / 5 + (0.0) => [-16.965  -5.655   5.655  16.965]
    Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>
        [-1.0, 1.0] / 5 + (0.13333334028720856) => [-0.467 -0.067  0.333  0.733]
        [-1.0, 1.0] / 5 + (0.13333334028720856) => [-0.467 -0.067  0.333  0.733]
        [-1.0, 1.0] / 5 + (0.13333334028720856) => [-0.467 -0.067  0.333  0.733]
        [-1.0, 1.0] / 5 + (0.13333334028720856) => [-0.467 -0.067  0.333  0.733]
        [-12.566370964050293, 12.566370964050293] / 5 + (1.675516128540039) => [-5.864 -0.838  4.189  9.215]
        [-28.274333953857422, 28.274333953857422] / 5 + (3.769911289215088) => [-13.195  -1.885   9.425  20.735]
    QTable(): size = (5, 5, 5, 5, 5, 5, 3)
    QTable(): size = (5, 5, 5, 5, 5, 5, 3)
    QTable(): size = (5, 5, 5, 5, 5, 5, 3)
    TiledQTable(): no. of internal tables =  3
    Environment: <TimeLimit<AcrobotEnv<Acrobot-v1>>>
    State space sizes: [(5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5)]
    Action space size: 3
    ```
    ### Start Training 
    ```
    def run(agent, env, num_episodes=10000, mode='train'):
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

    scores = run(agent, env)

    RESULT:
    Episode 10000/10000 | Max Average Score: -240.44
    ```
    ### Plot
    ```
    def plot_scores(scores, rolling_window=100):
        """Plot scores and optional rolling mean using specified window."""
        plt.plot(scores); plt.title("Scores");
        rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
        plt.plot(rolling_mean);
        return rolling_mean

    rolling_mean = plot_scores(scores)
    ```
    ![image11]


## Coarse Coding <a name="coarse_coding"></a>
- Like Tile coding, but uses a sparser set of features to encode the state space.
- Take a state S, mark all the circles that it belongs to.
- Bit vector with a one for those circles and 0 for the rest.
- = sparse coding representation of the state
- Smaller circles results in 
    - less generalization across the space
    - learning takes longer
    - greater effective resolution

- Larger circles 
    - more generalization,
    - smoother value function.
    - fewer large circles to cover the space,
    - lower resolution.

    ![image9]


## Function Approximation <a name="function_approximation"></a>
- True state value function **v<sub>π</sub>(s)**, or action value function **q<sub>π</sub>(s,a)** is typically smooth and continuous over the entire space.
- Capturing this completely is practically infeasible except for some very simple problems.
- Best approach is function approximation: 
    - Introduce a parameter vector W that shapes the function.
    - Reduce to tweaki this parameter vector to get the desired approximation.
    - The approximating function can either map a state to its value, or a state action pair to the corresponding q value.
    - Other approach: map from one state to a number of different q values, one for each action all at once. Useful for q learning.

- In general, define a transformation that converts any given state **s**
into a feature vector **x(s)**. 
- Dot Product. Multiply each feature with the corresponding weight, and sum it up.
- = linear function approximation

    ![image10]

## Linear Function Approximation <a name="lin_func_approx"></a> 
- Let's take a closer look at linear function approximation and how to
estimate the parameter vector w. 
- Initialize weights **w** randomly and compute state value **v(s,w)**
- Use gradient descent to find the optimal parameter vector.
- Note that since **v** hat is a linear function, its derivative with respect to **w** is simply the feature vector **x(s)**.
- Minimize the objective function. --> gradient descent --> chain rule
- If we are able to sample enough states, we can come close to the expected value.

    ![image12]

- Action-value function: use a feature transformation that utilizes both the state and action.
- Use the same gradient descent method as we did for the state-value function.
- Finally, compute all of the action-values at once --> use **weight matrix** instead of **weight vector**.
- Each column of the matrix emulates a separate linear function
- If we have a problem domain with a continuous state space, but a discrete action space which is very common, we can easily select the action with the maximum value.
- If our action space is also continuous, then this form allows us to output more than a single value at once.

    ![image13]


## Kernel Functions <a name="kernel_functions"></a>
- A simple extension to linear function approximation can help us capture non-linear relationships.
- At the heart of this approach is our feature transformation.
- Each element of the feature vector can be produced by a separate function,
which can be non-linear. These functions are called Kernel Functions or Basis Functions.
- Radial Basis Functions are kernel Functions: The closer the state is to the center of the blob, the higher the response returned by the function. The response falls off gradually with the radius. Mathematically, this can be achieved by associating a Gaussian Kernel with each Basis Function with its mean serving as the center of the blob and standard deviation determining how sharply or smoothly the response falls off.

    ![image14]

## Nonlinear Function approximation <a name="nonlin_func_approx"></a> 
- Imagine a non-linear combination of the feature values.
- Such a non-linear function is generally called an activation
function.
- Iteratively update the parameters of any such function using gradient descent.

    ![image15]



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
