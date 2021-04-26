[image1]: assets/gridworld.png "image1"
[image2]: assets/rl_overview.png "image2"
[image3]: assets/discrete_spaces.png "image3"
[image4]: assets/continuous_spaces.png "image4"


# Deep Reinforcement Learning Theory - RL in Continuous Spaces

## Content
- [Introduction](#intro)
- [Problem analysis](#problem_analysis)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)


## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

## Problem analysis
- So far all reinforcement learning environments were implemented where the number of states and actions is limited. 
- With small, **finite Markov Decision Processes (MDPs)**, it is possible to represent the **action-value function** with a **table**, **dictionary**, or other finite structure.

- Gridworld: Say the world has 
    - four possible states, 
    - and the agent has four possible actions at its disposal (up, down, left, right).  
    - The estimated optimal action-value function in a table, with a row for each state and a column for each action. --> Q-table.

    ![image1]

    ![image2]

## Discrete vs. Continuous Spaces
### Discrete Spaces
- Discrete spaces allow us to represent any function of states and actions as a **dictionary** or **look-up table**.
- Consider the state **value function V** which is a **mapping** from the **set of states** to a **real number**.
- Similarly, consider **the action value function Q** that **maps** every **state action pair** to a **real number**.
- Discreet spaces are also critical to a number of reinforcement learning **algorithms**. For instance, in **value iteration**, the internal for loop goes over each state as one by one, and updates the corresponding value estimate V of s. This is impossible if you have an infinite state space. The loop would go on forever 
- Model-free methods like Q-learning assume discrete spaces as well.

    ![image3]

### Contnuous Spaces
- A contnuous space can take a range of values, typically real numbers.
- Discrete visualization of states: real number, bar chart
- Contnuous visualization of states:  vectore, density plot 
- Consider: Most physical actions in nature are continuous 
- Dealing with contnuous spaces: **Discretization** and **Function Approximation**

    ![image4]

let's try to build some intuition for why continuous state spaces are important.

Where do they even come from?

When you consider a high-level decision making task like playing chess,
you can often think of the set of possible states as discrete.

What piece is in which square on the board.

You don't need to bother with precisely where
each piece is located within its square or which way it is facing.

Although these details are available for you to inspect and wonder about,
why is your knight staring at my queen.

These things are not relevant to the problem at hand
and you can abstract them away in your model of the game.

In general, grid-based worlds are very popular in reinforcement learning.

They give you a glimpse at how agents might act in spatial environments.

But real physical spaces are not always neatly divided up into grids.

There is no cell 5-3 for the vacuum cleaner robot to go to.

It has to chart a course from its current position to say 2.5
meters from the west wall by 1.8 meters from the north wall.

It also has to keep track of its heading and
turn smoothly to face the direction it wants to move in.

These are all real numbers that the agent may need
to process and represent as part of the state.

Actions too can be continuous.

Take for example a robot that plays darts.

It has to set the height and angle it wants to release the dart at,
choose an appropriate level of power with which to throw et cetera.

Even small differences in these values can have
a large impact on where the dart ultimately lands on the board.

In general, most actions that need to take place
in a physical environment are continuous in nature.

Clearly, we need to modify our representation or
algorithms or both to accommodate continuous spaces.

The two main strategies we'll be looking at are
Discretization and Function Approximation.



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