import gpflow
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from Arm import Arm
from Reward import Reward

"""
This file contains code for a synthetic problem model (used in supplemental simulation).
"""
DEFAULT_DF_NAME = "synthetic_df.hdf5"


class SyntheticProblemModel:
    df: pd.DataFrame

    def __init__(self, num_rounds, exp_num_clients, num_requests_per_client, exp_max_num_reqs,
                 budget, noise_std, context_dim, use_saved=False, seed=98765, saved_file_name=DEFAULT_DF_NAME):
        self.num_rounds = num_rounds
        self.exp_max_num_reqs = exp_max_num_reqs
        self.num_requests_per_client = num_requests_per_client
        self.exp_num_clients = exp_num_clients
        self.saved_file_name = saved_file_name
        self.context_dim = context_dim
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.budget = budget

        if not use_saved:
            self.initialize_dataset()
        self.h5_file = None
        self.bench_num_avai_groups = self.benchmark_superarm_list = None

    def context_to_grp_outcome(self, context):
        x = context
        grp_outcome = 0.05 + 0.95 * np.exp(-5*x)
        return grp_outcome

    def context_to_sup_outcome(self, context):
        x = context
        sup_outcome = 1 / (1 + np.exp(5 - 10 * x))
        return sup_outcome

    def set_benchmark_info(self, superarm_list, num_avai_groups):
        self.benchmark_superarm_list = superarm_list
        self.bench_num_avai_groups = num_avai_groups

    def get_available_arms(self, t):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        t = t - 1
        # Construct a list of Arm objects
        num_groups = self.h5_file[f"{t}"].attrs["num_groups"]

        round_dset = self.h5_file[f"{t}"]

        arm_list = []
        for i in range(num_groups):
            for context, exp_grp_outcome, exp_sup_outcome in \
                    zip(round_dset[f"group_{i}_context"], round_dset[f"group_{i}_grp_outcome"], round_dset[f"group_{i}_sup_outcome"]):
                arm_list.append(Arm(len(arm_list), np.array(context), exp_sup_outcome, exp_grp_outcome, i))
        return arm_list

    def get_superarm_regret(self, t, slate):
        t = t - 1

        # compute the algorithm's expected reward
        algo_exp_reward = np.sum([arm.true_mean for arm in slate])
        opt_slate = self.benchmark_superarm_list[t]
        opt_exp_reward = np.sum([arm.true_mean for arm in opt_slate])
        return max(opt_exp_reward - algo_exp_reward, 0)

    def get_group_regret(self, t, slate):
        t = t - 1

        # determine what groups arms belong to
        round_h5 = self.h5_file[f"{t}"]
        group_arm_outcomes = dict()
        for arm in slate:
            group_arm_outcomes[arm.group_id] = group_arm_outcomes.get(arm.group_id, []) + [arm.true_mean]

        group_reg = 0
        num_good_groups = 0
        for group_id, arm_outcomes in group_arm_outcomes.items():
            group_thresh = self.h5_file.attrs['group_thresh']
            group_reward = self.group_reward(arm_outcomes)
            if group_reward >= group_thresh:
                num_good_groups += 1
            group_reg += max(0, group_thresh - group_reward)

        return group_reg, num_good_groups / self.bench_num_avai_groups[t]

    def get_total_reward(self, rewards, t=None):
        # returns total super arm reward
        reward = np.sum([reward.grp_perf for reward in rewards])
        return reward

    def play_arms(self, t, slate):
        reward_list = [Reward(arm, arm.sup_outcome + np.random.normal(0, self.noise_std),
                              arm.grp_outcome + np.random.normal(0, self.noise_std)) for arm in slate]
        return reward_list

    def oracle1(self, est_outcome_arr, t, available_arms, scale_outcomes=True):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        # This function assumes that est_outcome_arr is in the order of available arms list

        t = t - 1  # because time starts from 1 but index starts from 0

        round_h5 = self.h5_file[f"{t}"]
        num_groups = round_h5.attrs["num_groups"]
        group_est_reward_dict = dict()  # maps group id to list of arm outcomes
        for est_outcome, arm in zip(est_outcome_arr, available_arms):
            arm_group_id = arm.group_id
            group_est_reward_dict[arm_group_id] = group_est_reward_dict.get(arm_group_id, []) + [est_outcome]

        # return groups whose estimate rewards are above threshold
        group_thresh = self.h5_file.attrs['group_thresh']
        passed_groups = [group_id for group_id in range(num_groups) if
                         self.group_reward(group_est_reward_dict[group_id]) > group_thresh]

        return passed_groups

    def oracle2(self, budget, est_outcome_arr, t, available_arms, avai_groups=None):
        """ budget is the number groups in the superarm this oracle will sort groups based on their estimate reward
        and return indices of basearms in the top "budget" groups """

        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        # This function assumes that est_outcome_arr is in the order of available arms list

        t = t - 1  # because time starts from 1 but index starts from 0

        round_h5 = self.h5_file[f"{t}"]
        num_groups = round_h5.attrs["num_groups"]
        avai_groups = set(range(num_groups)) if avai_groups is None or len(avai_groups) < budget else set(
            avai_groups)  # if None, assume all groups available
        group_est_reward_dict = dict()  # maps group id to estimate group reward
        group_arm_ids_dict = dict()  # maps group id to list of arm ids that are in that group
        for i, (est_outcome, arm) in enumerate(zip(est_outcome_arr, available_arms)):
            arm_group_id = arm.group_id
            if arm_group_id in avai_groups:
                group_arm_ids_dict[arm_group_id] = group_arm_ids_dict.get(arm_group_id, []) + [i]
                group_est_reward_dict[arm_group_id] = group_est_reward_dict.get(arm_group_id, 0) + est_outcome
        # sort groups in descending by estimate reward
        sorted_groups = sorted(group_est_reward_dict.items(), key=lambda x: -x[1])

        # get arms in top "budget" groups
        top_group_ids = [x[0] for x in sorted_groups[:budget]]
        top_arm_ids = np.concatenate([group_arm_ids_dict[group_id] for group_id in top_group_ids])
        return top_arm_ids

    def group_reward(self, arm_outcomes):
        return np.mean(arm_outcomes)

    def initialize_dataset(self):
        print("Generating synthetic dataset...")

        # sample availability probability for each client
        num_clients_arr = self.rng.poisson(self.exp_num_clients, self.num_rounds)

        client_request_arr = np.linspace(0.01, 1, self.num_requests_per_client)

        # create h5 dataset
        h5_file = h5py.File(self.saved_file_name, "w")

        max_num_basearms = 0  # this is M in paper
        for time in tqdm(range(self.num_rounds)):
            curr_time_group = h5_file.create_group(f"{time}")

            num_groups = num_clients_arr[time]

            # sample privacy threshold (gamma) for each client
            client_thresh_arr = self.rng.uniform(0, 1, num_groups)

            max_num_requests_arr = self.rng.poisson(self.exp_max_num_reqs, num_groups)

            # sample num clients per round
            curr_time_group.attrs["num_groups"] = num_groups

            num_basearms = num_groups * self.num_requests_per_client
            max_num_basearms = max(num_basearms, max_num_basearms)
            for g_id in range(num_groups):
                curr_time_group.attrs[f"threshold_{g_id}"] = client_thresh_arr[g_id]
                curr_time_group.attrs[f"max_num_reqs_{g_id}"] = max_num_requests_arr[g_id]
                # for each client we have requests in client_request_arr

                # create an H5 dataset for the group (available client)
                context_dataset_size = (self.num_requests_per_client)  # (base arm in group, context)
                grp_outcome_dataset_size = (self.num_requests_per_client)  # (base arm in group, expected group outcome)
                sup_outcome_dataset_size = (self.num_requests_per_client)  # (base arm in group, expected superarm outcome)

                group_basearm_context = curr_time_group.create_dataset(f"group_{g_id}_context",
                                                                       context_dataset_size, dtype=float)
                group_basearm_grp_outcome = curr_time_group.create_dataset(f"group_{g_id}_grp_outcome",
                                                                           grp_outcome_dataset_size, dtype=float)

                group_basearm_sup_outcome = curr_time_group.create_dataset(f"group_{g_id}_sup_outcome",
                                                                           sup_outcome_dataset_size, dtype=float)

                group_basearm_context[:] = client_request_arr
                group_basearm_grp_outcome[:] = self.context_to_grp_outcome(client_request_arr)
                group_basearm_sup_outcome[:] = self.context_to_sup_outcome(client_request_arr)


        # plot histogram of outcomes (for debugging/info purposes)
        group_outcomes = np.array([h5_file[f"{time}"][f"group_{g_id}_grp_outcome"] for time in range(self.num_rounds) for g_id in range(h5_file[f"{time}"].attrs["num_groups"])])
        plt.figure()
        plt.hist(group_outcomes.flatten(), bins=20, density=True)
        plt.xlabel("Group outcome")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig("synth_grp_outcome_dist.pdf", bbox_inches='tight', pad_inches=0.03)

        sup_outcomes = np.array([h5_file[f"{time}"][f"group_{g_id}_sup_outcome"] for time in range(self.num_rounds) for g_id in range(h5_file[f"{time}"].attrs["num_groups"])])
        plt.figure()
        plt.hist(sup_outcomes.flatten(), bins=20, density=True)
        plt.xlabel("Superarm outcome")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig("synth_sup_outcome_dist.pdf", bbox_inches='tight', pad_inches=0.03)


if __name__ == '__main__':
    test = SyntheticProblemModel(100, 50, 100, 10, 5, 0.001, 1, True, saved_file_name="temp_synth.hdf5")

    # df['']
    print('donerooni')
