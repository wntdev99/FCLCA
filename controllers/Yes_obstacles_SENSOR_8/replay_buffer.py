REPALY_MEMORY = 1000000
MIN_BATCH_SIZE = 1024
CL_NAME = "CL_YES_world_easy_6"
CL_LOAD_NAME = "CL_NO_world_3"

import random
import pickle
import datetime
import numpy as np
from collections import deque

DAY_NUM = str(datetime.date.today())
REPLAY_MEMORY_SAVE_FILE = f"/home/muros/문서/JOURNAL_PKL_FILE/{CL_NAME}_replay_memory.pkl"
REPLAY_MEMORY_LOAD_FILE = f"/home/muros/문서/JOURNAL_PKL_FILE/{CL_LOAD_NAME}_replay_memmory.pkl"

# experience replay 
class ReplayBuffer:

    def __init__(self):                                        
        self.epuck_experiences = self.load_replay_memory()
    # store experience
    def store_experience(self, state, next_state, reward, action, done):                       
        self.epuck_experiences.append((state, next_state, reward, action, done))

    # smapling experiences in storage
    def sample_batch(self):                                                                    
        batch_size = min(MIN_BATCH_SIZE, len(self.epuck_experiences))                                           
        sampled_epuck_batch = random.sample(self.epuck_experiences, batch_size)                   
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for epuck_experience in sampled_epuck_batch:
            state_batch.append(epuck_experience[0])
            next_state_batch.append(epuck_experience[1])
            reward_batch.append(epuck_experience[2])
            action_batch.append(epuck_experience[3])
            done_batch.append(epuck_experience[4])
        return np.array(state_batch), np.array(next_state_batch), np.array(
            action_batch), np.array(reward_batch), np.array(done_batch)   
    # Replay Memory를 파일에 저장하는 함수
    def save_replay_memory(self):
        with open(REPLAY_MEMORY_SAVE_FILE, 'wb') as f:
            pickle.dump(self.epuck_experiences, f)
    
    # 파일에서 Replay Memory를 불러오는 함수
    def load_replay_memory(self):
        try:
            with open(REPLAY_MEMORY_LOAD_FILE, 'rb') as f:
                self.epuck_experiences = pickle.load(f)
            return self.epuck_experiences
        except FileNotFoundError:
            return deque(maxlen=REPALY_MEMORY)   
        