from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from unityagents import UnityEnvironment
from ddpg_agent import Agent
#%matplotlib inline

# Load Reacher env with 20 agents
env = UnityEnvironment(
    file_name='Reacher_Linux_20/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Instantiate DDPG Agent
random_seed = 7
train_mode = True
agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)


def ddpg(n_episodes=2000, max_t=1000, print_every=10, learn_every=20, num_learn=10, goal_score=30, ):
    total_scores_deque = deque(maxlen=100)
    total_scores = []

    for i_episode in range(1, n_episodes + 1):
        # reset environment
        env_info = env.reset(train_mode=train_mode)[brain_name]
        # get current state for each agent
        states = env_info.vector_observations
        # initialize the score (for each agent)
        scores = np.zeros(num_agents)
        agent.reset()
        start_time = time.time()

        for t in range(max_t):
            # select an action
            actions = agent.act(states, add_noise=True)
            # send actions to environment
            env_info = env.step(actions)[brain_name]
            # get next state
            next_states = env_info.vector_observations
            # get reward
            rewards = env_info.rewards
            # check if episode has finished
            dones = env_info.local_done

            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                # send actions to the agent
                agent.step(state, action, reward, next_state, done)
            # pass states to next time step
            states = next_states
            # update the scores
            scores += rewards

            if t % learn_every == 0:
                for _ in range(num_learn):
                    agent.start_learn()
            # exit loop if episode finished
            if np.any(dones):
                break

        duration = time.time() - start_time
        # save lowest score for a single agent
        min_score = np.min(scores)
        # save highest score for a single agent
        max_score = np.max(scores)
        # save mean score for the episode
        mean_score = np.mean(scores)
        total_scores_deque.append(mean_score)
        total_scores.append(mean_score)
        total_average_score = np.mean(total_scores_deque)

        print('\rEpisode {}\tTotal Average Score: {:.2f}\tMean: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tDuration: {:.2f}'
              .format(i_episode, total_average_score, mean_score, min_score, max_score, duration))

        if i_episode % print_every == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print(
                '\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}'.format(i_episode, round(duration),
                                                                                            min_score, max_score,
                                                                                            mean_score))

        if total_average_score >= goal_score and i_episode >= 100:
            print(
                'Problem Solved after {} epsisodes Total Average score: {:.2f}'.format(i_episode, total_average_score))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return total_scores


scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()
