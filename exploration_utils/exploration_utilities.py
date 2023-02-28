"""
exploration_utilities.py - Exploration Utilities for enhanced exploration in deep RL
Author: Cláudio Rodrigues
Date: 2023

Description: A class that maintains state-related information to enable exploration pattern
calculations.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pandas as pd
import numpy as np

class ProgressionExploration:

    def __init__(self):
        # an array to store the reward values
        self.reward_store = []
        # an array to store the previous actions during a conversation
        self.action_record = []
        # a boolean that helps select between exploration and exploitation
        self.exploration = True

    def select_exploration_by_reward(self, reward):
        try:
            current_reward = self.reward_store[-1]
        except:
            print("no rewards in store yet")
        average_reward = sum(self.reward_store) / len(self.reward_store)

        if current_reward > average_reward or self.reward_store[-2] > average_reward:
            self.exploration = True
        else:
            self.reward_store = False

    def select_exploration_by_progression(self, action):
        try:
            current_action = self.action_record[-1]
            if current_action == self.action_record[-2]:
                self.exploration = False
            else:
                self.exploration = True
        except Exception as e:
            print("no actions in store yet - first step")
    def reset(self):
        # resetting all values after an episode ends
        self.exploration = True
        self.reward_store = []
        self.action_record = []

class RiskIndexCalculator:

    def __init__(self, dataframe, threshold):
        self.df = dataframe
        self.threshold = threshold
        self.success_rates = self.df["Success"].values

    def calculate_risk_index(self):
            disruptiveness = self.calculate_disruptiveness(self.threshold, self.success_rates)
            persistence = self.calculate_persistence(self.threshold, self.success_rates)
            risk_index = np.trapz(disruptiveness * persistence)
            return abs(risk_index)

    def calculate_disruptiveness(self, threshold, success_rates):
        disruptiveness = []
        for i in range(len(success_rates)):
            dis_t = threshold - success_rates[i]
            disruptiveness.append(dis_t)
        return disruptiveness

    def calculate_persistence(self, threshold, success_rates):
        persistence = 0
        for i in range(len(success_rates)):
            if threshold >= success_rates[i]:
                persistence += 1

        return persistence


