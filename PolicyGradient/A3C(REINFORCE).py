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
import threading
import time
import tensorflow as tf
import keras

global episode
episode = 0
global scores
scores = []
EPISODES = 3000
env_name = "CartPole-v1"

class A3CGobal:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.discount_factor = 0.99
        
        # 스레드 갯수
        self.num_thread = 8
        
        # Global_network의 actor /  critic
        self.g_actor = self.build_actor()
        self.g_critic = self.build_critic()
        
        self.g_actor_opt = self.actor_optimizer()
        self.g_critic_opt = self.critic_optimizer()
        
    def build_actor(self):
        input_layer = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu',
                 kernel_initializer='he_uniform')(input_layer)
        output = Dense(self.action_size, activation='softmax',
                      kernel_initializer='he_uniform')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model._make_predict_function()
        
        return model
    
    def build_critic(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size, ), activation='relu',
                       kernel_initializer='he_uniform'))
        model.add(Dense(12, activation='relu',
                       kernel_initializer='he_uniform'))
        model.add(Dense(self.value_size, activation='linear',
                       kernel_initializer='he_uniform'))
        
        model._make_predict_function()
        
        return model
        
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])
        #discounted_rewards = K.placeholder(shape=[None, ])
        
        # 크로스 엔트로피 오류함수 계산
        action_prob = K.sum(action * self.g_actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)
        
        # 정책신경망을 업데이트하는 훈련함수 생성
        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.g_actor.trainable_weights, [], loss)
        train = K.function([self.g_actor.input, action, advantage], [],
                           updates=updates)

        return train
    
    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])
        loss = K.mean(K.square(target - self.g_critic.output))
        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.g_critic.trainable_weights, [], loss)
        train = K.function([self.g_critic.input, target], [],
                           updates=updates)
        
        return train
    
    def train(self):
        actor_learners = [ActorLearner(self.state_size, self.action_size, self.value_size,
                                      self.g_actor, self.g_critic,
                                      self.g_actor_opt, self.g_critic_opt,
                                      self.discount_factor)
                         for _ in range(self.num_thread)]
        
        # 각 스레드 시작
        for actor_learner in actor_learners:
            time.sleep(1)
            actor_learner.start()

class ActorLearner(threading.Thread):
    def __init__(self, state_size, action_size, value_size, global_actor, global_critic,
                g_actor_opt, g_critic_opt, discount_factor):
        threading.Thread.__init__(self)
        
        self.discount_factor = discount_factor
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = value_size
        
        # Global_network의 actor /  critic
        self.g_actor = global_actor
        self.g_critic = global_critic
        
        self.g_actor_opt = g_actor_opt
        self.g_critic_opt = g_critic_opt
        
        # local model
        self.local_actor = self.build_local_actor()
        self.local_critic = self.build_local_critic()
        
        # 지정된 타임스텝 동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards = [], [], []
        
        # 모델 업데이트 주기
        self.t = 0
        self.t_max = 30
    
    def build_local_actor(self):
        input_layer = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu',
                 kernel_initializer='he_uniform')(input_layer)
        output = Dense(self.action_size, activation='softmax',
                      kernel_initializer='he_uniform')(x)
        
        local_actor = Model(inputs=input_layer, outputs=output)
        
        local_actor._make_predict_function()
        local_actor.set_weights(self.g_actor.get_weights())
        
        return local_actor
    
    def build_local_critic(self):
        local_critic = Sequential()
        local_critic.add(Dense(24, input_shape=(self.state_size, ), activation='relu',
                       kernel_initializer='he_uniform'))
        local_critic.add(Dense(12, activation='relu',
                       kernel_initializer='he_uniform'))
        local_critic.add(Dense(self.value_size, activation='linear',
                       kernel_initializer='he_uniform'))
        
        local_critic._make_predict_function()
        local_critic.set_weights(self.g_critic.get_weights())
        
        return local_critic
    
    def train_global_model(self):
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        values = self.local_critic.predict(states)[0]
        discounted_rewards = self.discount_rewards(self.rewards)
        advantages = discounted_rewards - values
        
        self.g_actor_opt([states, actions, advantages])
        self.g_critic_opt([states, discounted_rewards])
        
        self.states, self.actions, self.rewards = [], [], []
    
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    
    def update_local_model(self):
        self.local_actor.set_weights(self.g_actor.get_weights())
        self.local_critic.set_weights(self.g_critic.get_weights())
    
    
    def get_action(self, state):
        policy = self.local_actor.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]
    
    
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
    
    def run(self):
        global episode
        env = gym.make(env_name)
        
        step = 0
        
        while episode < EPISODES:
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1,4])

            while not done:
                #env.render()
                step += 1
                self.t += 1
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1,4])
                reward = reward if not done or score == 499 else -100
                self.append_sample(state, action, reward)

                score += reward
                state = next_state
                
                if self.t >= self.t_max or done:
                    self.train_global_model()
                    self.update_local_model()
                    self.t = 0
                    
                if done:
                    episode += 1
                    score = score if score == 500.0 else score + 100
                    scores.append(score)
                    print("Episode: ", episode, " Score: ", score, " step: ", step)
                    step = 0
                    
                    if np.mean(scores[-min(10,len(scores)):]) > 490:
                        print("processing time: ", time.time() - start)
                        sys.exit()


if __name__ == "__main__":
    episode = 0
    start = time.time()
    global_agent = A3CGobal(state_size=4, action_size=2)
    global_agent.train()