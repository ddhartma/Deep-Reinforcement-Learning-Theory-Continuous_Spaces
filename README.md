[image1]: assets/gridworld.png "image1"
[image2]: assets/rl_overview.png "image2"
[image3]: assets/discrete_spaces.png "image3"
[image4]: assets/continuous_spaces.png "image4"
[image5]: assets/non_uniform_discretization.png "image5"


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
- Discrete spaces allow us to represent any function of states and actions as a **dictionary** or **look-up table**.
- Consider the state **value function V** which is a **mapping** from the **set of states** to a **real number**.
- Similarly, consider **the action value function Q** that **maps** every **state action pair** to a **real number**.
- Discreet spaces are also critical to a number of reinforcement learning **algorithms**. For instance, in **value iteration**, the internal for loop goes over each state as one by one, and updates the corresponding value estimate V of s. This is impossible if you have an infinite state space. The loop would go on forever 
- Model-free methods like Q-learning assume discrete spaces as well.

    ![image3]

### Continuous Spaces <a name="continuous_spaces"></a>
- A contnuous space can take a range of values, typically real numbers.
- Discrete visualization of states: real number, bar chart
- Contnuous visualization of states:  vectore, density plot 
- Consider: Most physical actions in nature are continuous 
- Dealing with contnuous spaces: **Discretization** and **Function Approximation**

    ![image4]

## Discretization <a name="discretization"></a> 
- Discretization converts a **continuous space into a discrete one**.
- States and actions can be discretized
- See standard gridworld with obstacles below. Agent could think,
there is no path across these obstacles.
- **Non-Uniform** Discretization: Vary the grid according to these obstacles, then a feasible path for the agent is possible.
- An alternate approach would be to divide up the grid into smaller cells where required.

    ![image5]

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