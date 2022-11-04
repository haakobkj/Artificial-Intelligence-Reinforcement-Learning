import sys
import time
from constants import *
from environment import *
from state import State
import numpy as np
import random
"""
solution.py

This file is a template you should use to implement your solution.

You should implement code for each of the TODO sections below.

COMP3702 2022 Assignment 3 Support Code

Last updated by njc 12/10/22
"""


class RLAgent:

    #
    # TODO: (optional) Define any constants you require here.
    #

    def __init__(self, environment: Environment):
        self.environment = environment
        #
        # TODO: (optional) Define any class instance variables you require (e.g. Q-value tables) here.
        #
        self.number_of_episodes = 15000
        self.max_steps = 100

        # Exploration parameters
        self.epsilon = 1.0                 # Exploration rate
        self.max_epsilon = 1.0             # Exploration probability at start
        self.min_epsilon = 0.01            # Minimum exploration probability 
        self.decay_rate = 0.005             # Exponential decay rate for exploration prob

        self.state_space_size = self.number_of_states()
        self.action_space_size = len(ROBOT_ACTIONS)
        
        # self.Q = np.random.rand(len(self.state_space_size), self.action_space_size)
        # self.Q[:, :] = np.zeros(self.action_space_size) 
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.rewards = None

    # === Q-learning ===================================================================================================

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        # https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
        self.rewards = {}

        for episode in range(self.number_of_episodes):
            state = self.environment.get_init_state()
            step = 0
            total_rewards = 0

            for step in range(self.max_steps):
                
                # Epsilon greedy step
                exp_exp_tradeoff = random.uniform(0, 1)
                if exp_exp_tradeoff > self.epsilon:
                    action = np.argmax(self.q_table[state,:])
                else:
                    action = random.choice(ROBOT_ACTIONS)
                

                reward, next_state =  self.environment.perform_action(state, action)
                self.q_table[state, action] = self.q_table[state, action] + self.environment.alpha * (reward + self.environment.gamma * np.max(self.q_table[next_state, :]) - self.q_table[state, action]) # Tabular QL
                # Hvorfor får jeg denne? IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
                state = next_state
                
                if self.environment.is_solved(state): 
                    break

            
        
            # epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*episode) 
            self.rewards[state] = self.environment.get_total_reward()



    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  Q-learning Q-values) here.
        #
        return max(self.rewards, key=self.rewards.get)
    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        #
        # TODO: Implement your SARSA training loop here.
        #
        pass

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  SARSA Q-values) here.
        #
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: (optional) Add any additional methods here.
    #
    #

    
    def number_of_states(self):
        states = []
        states.append(self.environment.get_init_state())
        visited = [self.environment.get_init_state()]
        while len(visited) > 0:
            current_state = visited.pop()
            for action in ROBOT_ACTIONS:
                cost, new_state = self.environment.perform_action(current_state, action) # Denne er private så bør egt ikke bruke den.
                if new_state not in states and new_state not in visited:
                    visited.append(new_state)
            if current_state not in states:
                states.append(current_state)
        return len(states)

    def epsilon_greedy(self, state):
        if np.random.uniform(0, 1) < 0.1: #EPSILON
            return np.random.choice(self.action_space) #Explore (choose random number between 0 and len of action space)
        else:
            return np.argmax(self.Q[state, :])