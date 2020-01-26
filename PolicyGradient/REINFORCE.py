import copy
import pylab
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../") 
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import gym

# 그리드월드 예제에서의 REINFORCE 에이전트
class ReinforceAgent:
    def __init__(self):
        # 가능한 모든 행동 정의
        self.action_space = [0, 1]
        # 상태와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 4
        self.discount_factor = 0.99 
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

        
    
    # 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        return model
    
    # 정책신경망을 업데이트 하기 위한 오류함수와 훈련함수의 생성
    def optimizer(self):
        action = K.placeholder(shape=[None, 2])
        discounted_rewards = K.placeholder(shape=[None, ])
        
        # 크로스 엔트로피 오류함수 계산
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)
        
        # 정책신경망을 업데이트하는 훈련함수 생성
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)

        return train

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]
    
    # 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        self.optimizer([states, actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

EPISODES = 800

if __name__=="__main__":
    env = gym.make('CartPole-v1')
    agent = ReinforceAgent()
    
    global_step = 0
    scores, episodes = [], []
    
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1,4])
        
        while not done:
            env.render()
            global_step += 1
            
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1,4])
            reward = reward if not done or score == 499 else -100
            agent.append_sample(state, action, reward)
            
            score += reward
            state = copy.deepcopy(next_state)
            
            if done:
                agent.train_model()
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                score = round(score, 2)
                print("episode: ", e, " score: ", score, " time_step: ", global_step)
                
                # 최근 10번의 에피소드의 평균점수가 490이 넘으면 종료
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()