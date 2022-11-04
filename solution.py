
from constants import *
from environment import *
from state import State
import random
import matplotlib.pyplot as plt

"""
solution.py

This file is a template you should use to implement your solution.

You should implement code for each of the TODO sections below.

COMP3702 2022 Assignment 3 Support Code

Last updated by njc 12/10/22
"""


class RLAgent:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.epsilon = 0.5
        self.states = [self.environment.get_init_state()]
        self.q_table = {(self.environment.get_init_state(), action) : 0 for action in ROBOT_ACTIONS}

        plt.xlabel('Episode')
        plt.ylabel('50-step moving average reward')

    # === Q-learning ===================================================================================================


    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        state = self.environment.get_init_state()
        rewards = []
        
        reward_averages = []


        total_episode_reward = 0
        episodes = 0
        # while self.environment.get_total_reward() > self.environment.training_reward_tgt:
        while episodes < 50000:

            action = self.epsilon_greedy(state)
            
            reward, next_state = self.environment.perform_action(state, action)

            total_episode_reward += reward



            if next_state not in self.states:
                self.states.append(next_state)
                for robot_action in ROBOT_ACTIONS:
                    self.q_table[(next_state, robot_action)] = 0
            
            old_q = self.q_table[(state, action)]
            best_next_q = 0
            if not self.environment.is_solved(next_state):
                best_next_q = self.q_table[(next_state, self.best_action(state))] 
            
            target = reward + self.environment.gamma * best_next_q
            new_q = old_q + self.environment.alpha * (target - old_q)
            self.q_table[(state, action)] = new_q

            
            if self.environment.is_solved(next_state):
                episodes += 1
                rewards.append(total_episode_reward)
                if len(rewards) < 50:
                    reward_average = sum(rewards)/ len(rewards)
                else:
                    reward_average = sum(rewards[-50:]) / 50

                reward_averages.append(reward_average)
                total_episode_reward = 0
                state = self.environment.get_init_state()
                
            else:
                state = next_state
        
        return episodes, reward_averages



    
    


    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        res = self.best_action(state)
        print(res)
        return self.best_action(state)
        
    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        state = self.environment.get_init_state()
        action = None

        rewards = []
        
        reward_averages = []


        total_episode_reward = 0
        episodes = 0

        # while self.environment.get_total_reward() > self.environment.training_reward_tgt:
        while episodes < 50000:

            
            if action is None:
                action = self.epsilon_greedy(state)
            
            reward, next_state = self.environment.perform_action(state, action)
            
            total_episode_reward += reward


            if next_state not in self.states:
                self.states.append(next_state)
                for robot_action in ROBOT_ACTIONS:
                    self.q_table[(next_state, robot_action)] = 0
            
            next_action = self.epsilon_greedy(next_state)
            
            old_q = self.q_table[(state, action)]
            best_next_q = 0
            if not self.environment.is_solved(next_state):
                best_next_q = self.q_table[(next_state, next_action)]
            
            target = reward + self.environment.gamma * best_next_q
            new_q = old_q + self.environment.alpha * (target - old_q)
            self.q_table[(state, action)] = new_q

            if self.environment.is_solved(next_state):
                episodes += 1
                rewards.append(total_episode_reward)
                if len(rewards) < 50:
                    reward_average = sum(rewards)/ len(rewards)
                else:
                    reward_average = sum(rewards[-50:]) / 50

                reward_averages.append(reward_average)
                total_episode_reward = 0

                state = self.environment.get_init_state()                
                action = None
            else:
                state = next_state
                action = next_action

        return episodes, reward_averages


    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        res = self.best_action(state)
        print(res)
        return self.best_action(state)
         
    # === Helper Methods ===============================================================================================


    def best_action(self, state: State):
        best_q = float('-inf')
        best_action = None
        for action in ROBOT_ACTIONS:
            q = self.q_table[(state, action)]
            if q is not None and q > best_q:
                best_q = q
                best_action = action
        return best_action


    def epsilon_greedy(self, state):
        explore_expoit_tradeoff = random.uniform(0, 1)
        if explore_expoit_tradeoff > self.epsilon:
            return self.best_action(state)
        else:
            return random.choice(ROBOT_ACTIONS)

