import time

import gpflow
import math
import numpy as np
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from math import sqrt
from tqdm import tqdm

from problem_models.ProblemModel import ProblemModel
from problem_models.synth_problem_model import SyntheticProblemModel

"""
This class represents the TCGP-UCB algorithm that is presented in the paper.
"""


class TCGP_UCB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, dim, budget, delta, max_arriving_arms, zeta, use_marginal_rewards=False,
                 num_inducing=100, mean_function=None, kernel=None, noise_variance=1e-5, seed=54324):
        self.max_arriving_arms = max_arriving_arms
        self.use_marginal_rewards = use_marginal_rewards
        self.problem_model = problem_model
        self.num_inducing = num_inducing
        X = np.zeros((1, dim))
        Y = np.zeros((1, 2))
        Z = X[:, :].copy()

        if kernel is None:
            self.kernel = gpflow.kernels.SharedIndependent(gpflow.kernels.SquaredExponential(), output_dim=2)
        else:
            self.kernel = kernel
        self.kernel = gpflow.utilities.freeze(self.kernel)
        iv = gpflow.inducing_variables.SharedIndependentInducingVariables(gpflow.inducing_variables.InducingPoints(Z))
        self.model = gpflow.models.SVGP(self.kernel, gpflow.likelihoods.Gaussian(noise_variance), inducing_variable=iv,
                                        num_latent_gps=2)

        # ensure that neither the kernel nor likelihood hyperparameters are optimized during posterior update
        gpflow.set_trainable(self.model.kernel, False)
        gpflow.set_trainable(self.model.likelihood, False)
        gpflow.set_trainable(self.model.inducing_variable, False)

        # since we use SVGP and SVGP does not have a closed-form for the posterior update, we must use an optimizer to
        # find the posterior. Note that we do NOT update the kernel or likelihood hyperparameters and only the mu
        # and sigma of the GP
        self.optimizer = gpflow.optimizers.NaturalGradient(gamma=1.0)

        self.max_iter = 10
        self.budget = budget
        self.dim = dim
        self.num_rounds = problem_model.num_rounds
        self.delta = delta
        self.zeta = zeta
        self.use_sparse = True
        self.noise_variance = noise_variance
        self.rng = np.random.default_rng()

    def set_model_data(self, train_X, train_Y, dim):
        train_X_np = np.array(train_X).reshape((-1, dim))
        train_Y_np = np.array(train_Y).reshape((-1, 1))

        num_inducing = min(train_X_np.shape[0], self.num_inducing)

        # randomly pick points among training samples
        inducing_indices = self.rng.choice(np.arange(train_X_np.shape[0]), num_inducing, replace=False)
        inducing_pts = train_X_np[inducing_indices].copy()

        # create multi-output inducing variables from inducing_pts
        iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(inducing_pts)
        )

        self.model = gpflow.models.SVGP(self.kernel, gpflow.likelihoods.Gaussian(self.noise_variance), inducing_variable=iv,
                                        num_latent_gps=2)

        # ensure that neither the kernel nor likelihood hyperparameters are optimized during posterior update
        gpflow.set_trainable(self.model.kernel, False)
        gpflow.set_trainable(self.model.likelihood, False)
        gpflow.set_trainable(self.model.inducing_variable, False)

        # compute posterior
        training_loss = self.model.training_loss_closure((self.obs_x_arr, self.obs_y_arr), compile=False)
        self.optimizer.minimize(training_loss, [(self.model.q_mu, self.model.q_sqrt)])

    def beta(self, t):
        m = self.max_arriving_arms
        return 2 * np.log(m * (t ** 2) * (np.pi ** 2) / (3 * self.delta))

    def run_algorithm(self):
        print(f"Running with zeta={self.zeta}...")
        self.num_rounds = self.problem_model.num_rounds
        total_reward_arr = np.zeros(self.num_rounds)
        superarm_regret_arr = np.zeros(self.num_rounds)
        group_regret_arr = np.zeros(self.num_rounds)
        percent_good_groups_arr = np.zeros(self.num_rounds)
        opt_logs = []
        time_taken_arr = np.zeros(self.num_rounds)
        train_X = []
        train_Y = []

        for t in tqdm(range(1, self.num_rounds + 1)):
            starting_time = time.time()
            available_arms = self.problem_model.get_available_arms(t)

            # rank based on index
            rand_arm_indices = np.arange(len(available_arms))
            # index list contains tuples of (i',i) for each arm
            indices_arr = np.array([self.get_arm_indices(arm, t) for arm in available_arms])
            group_index_arr = indices_arr[:, 0]
            superarm_index_arr = indices_arr[:, 1]

            idx_to_play = self.problem_model.oracle(self.budget, superarm_index_arr, t)

            slate = [available_arms[idx] for idx in rand_arm_indices[idx_to_play]]

            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards, t)
            # superarm_regret_arr[t - 1] = self.problem_model.get_superarm_regret(t, slate)
            percent_good_groups_arr[t - 1] = self.problem_model.get_perc_satisfied_groups(t, rewards)

            # Update the GP model
            train_X.extend([arm.context for arm in slate])
            train_Y.extend([reward.grp_perf for reward in rewards])

            self.set_model_data(train_X, train_Y, self.dim)
            time_taken_arr[t - 1] = time.time() - starting_time

        return {
            'time_taken_arr': time_taken_arr,
            'total_reward_arr': total_reward_arr,
            'superarm_regret_arr': superarm_regret_arr,
            'group_regret_arr': group_regret_arr,
            "percent_good_groups_arr": percent_good_groups_arr,
            'model': gpflow.utilities.freeze(self.model)
        }

    # returns a tuple: (arm_index1, arm_index2) for group filtering and super arm selection, respectively
    def get_arm_indices(self, arm, t):
        beta = self.beta(t)
        mean, var = self.model.predict_f(arm.context.reshape(1, -1))
        sup_mean, grp_mean = mean.numpy().flatten()

        sup_var, grp_var = var.numpy().flatten()
        index_grp = grp_mean + 1 / self.zeta * np.sqrt(beta * grp_var)
        index_sup = sup_mean + 1 / (1 - self.zeta) * np.sqrt(beta * sup_var)
        return index_grp, index_sup

    def calc_confidence(self, num_times_node_played):
        if num_times_node_played == 0:
            return float('inf')
        return sqrt(2 * math.log(self.num_rounds) / num_times_node_played)

if __name__ == '__main__':
    # debugging
    test_prob = SyntheticProblemModel(100, 50, 100, 10, 5, 0.001, 1, True, saved_file_name="temp_synth.hdf5")
    algo = TCGP_UCB(test_prob, 1, 10, 0.01, 100, 0.5)
    print("sfddsf")