# Deep Reinforcement Learning : Continuous Control

This project repository contains my work for the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Project 2: Continuous Control.

### The goal of this project

In this project, the goal is to train an agent that controls a robotic arm within Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. The goal is to get 20 different robotic arms to maintain contact with the green spheres.


![In Project 1, train an agent that controls a robotic arm.](images/reacher_20_u.gif)

### Enviroment 

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, we will provide you with two separate versions of the Unity environment:

 - The first version contains a single agent.
 - The second version contains 20 identical agents, each with its own copy of the environment.

Please note that in this solution the second version has been chosen to be solved.
More detailed description of the environment can be found in this [paper](https://arxiv.org/pdf/1809.02627.pdf).


### Getting Started
Please note that the instruction bellow is for Linux enviroment and they have been tested on Ubuntu 18.04 LTS.
1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/index.html#)

2. Prepare a `python 3.6` environment and install required python libraries in Conda virtual environment 

	```
	conda env create -f environment.yml
	```
	
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
Version 2: Twenty (20) Agents

 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
 - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
