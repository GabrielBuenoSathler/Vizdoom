import gymnasium
from vizdoom import gymnasium_wrapper
import cv2
import numpy as np
import torch
import random
from collections import deque
from skimage.color import rgb2gray
import matplotlib.pyplot as plt# inicializia o ambiente
env = gymnasium.make("VizdoomCorridor",render_mode='human')
observation, info = env.reset()
#key to state = transforma o key em um stata para uso posterior
print(f"ações possiveis {env.action_space.n}")
#print(env.env.get_action_meanings())
print(env.action_space.n)
print('----------------------------')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Agent:
    def __init__(self,model, device='cuda',epilson=1.0,  min_epilson = -0.1,nb_warmup = 10000, n_action=None, memory_capacity=10000,batch_size=32, learnign_rate=0.00001):
        self.device = device 
        self.epilson = epilson 
        self.epilson = epilson
        self.min_epilson = 
        
class ExperienceBuffer:
    def __init__(self, capacity,device='cuda'):
        self.device = device
        self.buffer = deque(maxlen=capacity)
 
    def __len__(self):
        return len(self.buffer)
 
    def append(self, experience):
        self.buffer.append(experience)
 
    def sample(self, batch_size=32):
        batch = random.sample(self.buffer,batch_size)
        batch = zip(*batch)
        
        states, actions, rewards, dones, next_states = batch
        states = np.array(states)
        actions= np.array(actions)
        rewards =    np.array(rewards, dtype=np.float32)
        dones=  np.array(dones, dtype=np.uint8)
        new_states =    np.array(next_states)
        state1_t = torch.tensor(states).to(device)
        state2_t = torch.tensor(new_state).to(device)
        reward_t = torch.from_numpy(actions).to(device)
        actions_t = torch.from_numpy(rewards).to(device)
        return state1_t , state2_t , reward_t , actions_t
        
def key_to_state(dic):
    for key, value in observation.items():
   # print(f'{key}: {value.shape}')
        if key == 'screen':
            state_1 = value
            return state_1
                

def resize(img):
    s1 = np.dot(img[...,:3], [0.299, 0.587, 0.144])
    res = cv2.resize(s1, dsize=(84, 84), interpolation=cv2.INTER_LINEAR)
    res = np.expand_dims(res, axis=0)
    return res

    

replay = ExperienceBuffer(100000)
count = 0
for _ in range(100):
    state, info = env.reset()
    state = key_to_state(state)
    state = resize(state)
    count = count + 1 
    done = False
    while not done:
        action = env.action_space.sample()  # this is where you would insert your policy
        new_state, reward, terminated, truncated, info = env.step(action)
  
        new_state = key_to_state(observation)
        
        new_state  = resize(new_state)   
       
        done = terminated or truncated
        print(f"new_state{new_state.shape}")
        experience = ([state,new_state,reward,action,done])
        replay.append(experience)
        
        print(f"done{type(done)},{type(new_state)} , {type(state)} , {type(reward)}, {type(action)}" )
env.close()

