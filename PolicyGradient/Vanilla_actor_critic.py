import copy
import pylab
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../") 
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras import backend as K
import gym

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.discount_factor = 0.99
        self.actor_opt = self.actor_optimizer()
        self.critic_opt = self.critic_optimizer()
        
    def build_actor(self):
        input_layer = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu',
                 kernel_initializer='he_uniform')(input_layer)
        output = Dense(self.action_size, activation='softmax',
                      kernel_initializer='he_uniform')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        
        return model
        
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        td_error = K.placeholder(shape=[None, ])
        #discounted_rewards = K.placeholder(shape=[None, ])
        
        # 크로스 엔트로피 오류함수 계산
        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * td_error
        loss = -K.sum(cross_entropy)
        
        # 정책신경망을 업데이트하는 훈련함수 생성
        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, td_error], [],
                           updates=updates)

        return train
        
    def train_model(self, state, action, reward, next_state, done):
        
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]
        
        act = np.zeros([1, self.action_size])
        act[0][action] = 1
        
        if done:
            target = reward
            td_error = reward - value
        else:
            target = reward + self.discount_factor * next_value
            td_error = (reward + self.discount_factor * next_value) - value
        
        self.actor_opt([state, act, td_error])
        self.critic_opt([state, target])
        
    def build_critic(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size, ), activation='relu',
                       kernel_initializer='he_uniform'))
        model.add(Dense(12, activation='relu',
                       kernel_initializer='he_uniform'))
        model.add(Dense(self.value_size, activation='linear',
                       kernel_initializer='he_uniform'))
        
        return model
    
    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])
        loss = K.mean(K.square(target - self.critic.output))
        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [],
                           updates=updates)
        
        return train

        
    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]


if __name__ == "__main__":
    EPISODES = 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = A2CAgent(state_size, action_size)
    
    scores, episodes = [], []
    
    for e in range(EPISODES):
        score = 0
        done = False
        
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100
            
            agent.train_model(state, action, reward, next_state, done)
            
            score += reward
            state = next_state
            
            if done:
                score = score if score == 500.0 else score + 100
                print("Episode: ", e, "score: ", score)
                scores.append(score)
                episodes.append(e)
                
                if np.mean(scores[-min(10,len(scores)):]) > 490:
                    sys.exit()