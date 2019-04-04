# DDRL-Tennis
Tennis environment in DDRL nanodegree course - project III.

# Environment
The environment used to showcase the performance of the algorithm was an environment from Unity, called Tennis. Two agents have to bounce a ball over the net and receive a reward of +0.1 if this is done successfully, and -0.1 if not. The primary difficulty in the environment is the fact that it is no longer stationary with another agent present. Note that multi-agent environments can be set up as either competitive or collaborative. 

The goal in the environment is to keep the ball in play, and through such means increase the reward. The task is finished when the reward of 0.500 is reached. In this environment, both agents receive their own local observations (observation space of 8 variables) and has two actions to choose from (towards or away from the net). The task is in 2D to limit the action space.

# Algorithm
The algorithm used is called MADDPG (Multi Agent Deep Deterministic Policy Gradient), as also described by Open AI. This algorithm uses centralized learning and decentralized execution with two agents (agent 0 and agent 1); it also uses the actor-critic logic that we explored in past projects for a more stable learning algorithm. The algorithm is new in the sense that it beats traditional algorithms (DDPG, A2C, Q-learning etc.) in a multi-agent environment. The algorithm furthermore makes use of Experience Replay, similar to past projects, where the experience tuples are sampled from memory. For exploration, the algorithm uses a noise function that adds random noise. 

When it comes to activation layers; the networks use simple ReLU activation functions with tanh on the last layer. I tried implementing batchnorm on the first layer but got a size mismatch that I couldn’t fix, so in the end trained without it.

The algorithm uses a lot of parameters, which I have optimized by learning from others’ recommendations. The final list of parameters used in this code is below. More information on the MADDPG algorithm can be found here: https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf

BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 512         
LR_ACTOR = 1e-3         
LR_CRITIC = 3e-4)        
WEIGHT_DECAY = 0        
LEARN_EVERY = 1         
LEARN_NUM = 1           
GAMMA = 0.99            
TAU = 6e-2  
OU_SIGMA = 0.2          
OU_THETA = 0.13         
EPS_START = 5.5         
EPS_EP_END = 250        
EPS_FINAL = 0  

# Dependencies
Make sure that numpy, matplotlib, pytorch are installed. Use Python 3.6+. You can follow this tutorial for an easy installation of gym: https://github.com/openai/gym

# Running the file
With all the files in the same folder, just follow the Tennis.ipynb notebook.


