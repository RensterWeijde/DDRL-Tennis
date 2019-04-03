# import packages
import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

# import Actor, Critic from Model
from Model import Actor, Critic

# initialize all parameters
BUFFER_SIZE = int(1e6)  # Replay Buffer
BATCH_SIZE = 128        # Batch size 128 (could be 256 possibly)
LR_ACTOR = 1e-3         # Alpha of Actor
LR_CRITIC = 3e-4        # Alpha of Critic, same
WEIGHT_DECAY = 0        # No weight decay
LEARN_EVERY = 1         # Learn every timestep, soft updates
LEARN_NUM = 1           # Learning passes
GAMMA = 0.99            # Reward discount factor
TAU = 6e-2              # Soft update TAU function
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck to add stochasticity (noise)
OU_THETA = 0.13         # Ornstein-Uhlenbeck to add stochasticity (noise)
EPS_START = 5.5         # Epsilon
EPS_EP_END = 250        # Epsilon
EPS_FINAL = 0           # Epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Agent class that can have multiple copies of the agent

class Agent():
    '''Agent to interact with the environment.'''

    def __init__(self, state_size, action_size, num_agents, random_seed):
        
        # link all variables to self
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.eps = EPS_START
        self.eps_decay = 1/(EPS_EP_END*LEARN_NUM) 
        self.timestep = 0 # time initialized to zero

        # Initialize Actor network, local and target
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Initialize Critic network, local and target
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise 
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, agent_number):
        '''Saves the experience, samples randomly from Replay Buffer to learn at specific time intervals. Needs to have enough experiences in memory, and needs to align with LEARN-EVERY param.'''
        
        self.timestep += 1
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > BATCH_SIZE and self.timestep % LEARN_EVERY == 0:
                for _ in range(LEARN_NUM):
                    experiences = self.memory.sample() # random
                    self.learn(experiences, GAMMA, agent_number)

    def act(self, states, add_noise):
        '''Returns the actions given the current policy. Both agents included.'''
        
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        
        with torch.no_grad():
            for agent_num, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_num, :] = action
        self.actor_local.train()
        
        # add noise
        if add_noise:
            actions += self.eps * self.noise.sample()
        actions = np.clip(actions, -1, 1)
        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_number):
        '''Update policy given batch of experiences. Actor target learns the action. Critic target learns the Q-value.'''

        states, actions, rewards, next_states, dones = experiences

        ## UPDATE CRITIC
        
        actions_next = self.actor_target(next_states)
        
        # make actions specific for the 2 agents
        if agent_number == 0:
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)
            
        # Compute Q targets 
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        ## UPDATE ACTOR
        
        actions_pred = self.actor_local(states)
        
        # Make specific for each agent
        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
            
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ## UPDATE TARGET NETWORKS
        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # update noise params
        self.eps -= self.eps_decay
        self.eps = max(self.eps, EPS_FINAL)
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        '''Renew small % of all weights at each timestep.'''
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data) # copy and replace, only small %

class OUNoise:
    '''Ornstein-Uhlenbeck process.'''

    def __init__(self, size, seed, mu=0.0, theta=OU_THETA, sigma=OU_SIGMA):
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
       
        self.state = copy.copy(self.mu)

    def sample(self):
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    '''Buffer with experiences tuples to provide random samples for learning.'''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''Add a new experience.'''
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''Random sampling of tuples.'''
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''Returns length of memory.'''
        return len(self.memory)
