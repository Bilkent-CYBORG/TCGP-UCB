import numpy as np
from tqdm import tqdm

"""
This class represents a benchmark algorithm that picks the optimal super arm in each round.
"""


class Benchmark:

    def __init__(self, problem_model, budget):
        self.budget = budget
        self.num_rounds = problem_model.num_rounds
        self.problem_model = problem_model

    def run_algorithm(self):
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        slate_list = []
        num_avai_groups = np.zeros(self.num_rounds)

        for t in tqdm(range(1, self.num_rounds + 1)):
            available_arms = self.problem_model.get_available_arms(t)
            true_grp_means = [arm.grp_outcome for arm in available_arms]
            available_groups = self.problem_model.oracle1(true_grp_means, t, available_arms, scale_outcomes=False)
            num_avai_groups[t - 1] = len(available_groups)
            slate_indices = self.problem_model.oracle2(self.budget, true_means, t, available_arms, available_groups)
            slate = [available_arms[idx] for idx in slate_indices]
            slate_list.append(slate)
            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards, t)
            regret_arr[t - 1] = 0

        return {
            "bench_slate_list": slate_list,
            "num_avai_groups": num_avai_groups,
            'total_reward_arr': total_reward_arr,
            'regret_arr': regret_arr,
        }