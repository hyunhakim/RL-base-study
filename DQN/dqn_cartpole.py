import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 300

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.train_start = 1000
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 64
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.discount_factor = 0.99
        
        self.update_target_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(units=24, input_dim=self.state_size, activation='relu',
                       kernel_initializer='he_uniform'))
        model.add(Dense(units=24, activation='relu',
                       kernel_initializer='he_uniform'))
        model.add(Dense(units=self.action_size, activation='linear',
                       kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def get_action(self, state):
        action_probs = np.ones(self.action_size) * (self.epsilon / self.action_size)
        best_action = np.argmax(self.model.predict(state)[0])
        action_probs[best_action] += 1.0 - self.epsilon
        
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)
    
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_model(self):
        # epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
        # replay memory에서 random하게 batch size만큼 뽑아냄
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])
        
        # target model / model의 Q function 계산
        q_val = self.model.predict(states)
        target_q_val = self.target_model.predict(next_states)
        
        # 벨만 최적 방정식을 이용하여 target_q_val 계산
        for i in range(self.batch_size):
            if dones[i]:
                q_val[i][actions[i]] = rewards[i]
            else:
                q_val[i][actions[i]] = rewards[i] + self.discount_factor * np.max(target_q_val[i])
        
        # update
        self.model.fit(states, q_val, batch_size=self.batch_size, epochs=1, verbose=0)
    
    
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    episodes = []
    
    for e in range(EPISODES):
        score = 0
        done = False
        state = env.reset()
        state = np.reshape(state, [1,state_size])
        
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100
            
            agent.append_sample(state, action, reward, next_state, done)
            
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
            
            score += reward
            state = next_state
            
            if done:
                agent.update_target_model()
                
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
                
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()