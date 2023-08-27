INPUT_SIZE = 30
ACTION_SIZE = 6
LEARNING_RATE = 1e-7
EPSILION = 1e-2
GAMMA = 0.95
DECLINE_LIMIT = 0.1
MODEL_NAME = 'Curriculum Yes new model_1_2'

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# DQN Agent 객체
class DqnAgent:
    def __init__(self, initial_learning_rate=LEARNING_RATE, learning_rate_decay=0.97):
        self.q_net = self.Dqn_model()
        self.target_q_net = self.Dqn_model()
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate = initial_learning_rate  # Initialize the learning rate


    # DQN model sturture
    @staticmethod
    def Dqn_model():
        q_net = tf.keras.models.load_model(MODEL_NAME)
        return q_net

    # random action set
    def random_policy(self, state):
        return np.random.randint(0, ACTION_SIZE)                                                                     

    # policy or random select
    def collect_policy(self,max_episodes, episode_cnt, state):
        # heuristics set 0.01 , 10 , 0.8 ㆍㆍㆍ
        epsilon = EPSILION + (1 - EPSILION) * np.exp(-(10 * (episode_cnt) / max_episodes) * DECLINE_LIMIT)
        if np.random.random() < epsilon:                                                                       
            return self.random_policy(state)
        return self.policy(state)

    # policy action set
    def policy(self, state):                                                                                
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)                                
        action_q = self.q_net(state_input)                                                                  
        action = np.argmax(action_q.numpy()[0], axis=0)                                                                    
        return action

    # target network update
    def update_target_network(self):
        self.target_q_net.set_weights(self.q_net.get_weights())                                             

    # model train
    def train(self, batch):                                                                                 
        state_batch, next_state_batch, action_batch, reward_batch, done_batch \
            = batch
        current_q = self.q_net(state_batch).numpy()                                                         
        target_q = np.copy(current_q)                                                                       
        next_q = self.target_q_net(next_state_batch).numpy()                                                
        max_next_q = np.amax(next_q, axis=1)                                                             
        for i in range(state_batch.shape[0]):                                                               
            target_q_val = reward_batch[i]                                                           
            if not done_batch[i]:                                                                           
                target_q_val += GAMMA * max_next_q[i]                                                       
            target_q[i][action_batch[i]] = target_q_val                                                     
        training_history = self.q_net.fit(x=state_batch, y=target_q, verbose=0)                             
        loss = training_history.history['loss']                                                             
        return loss
        
    def update_learning_rate(self, episode):
        self.learning_rate = self.initial_learning_rate * (self.learning_rate_decay ** episode)
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.q_net.compile(optimizer=optimizer, loss='mse')
        