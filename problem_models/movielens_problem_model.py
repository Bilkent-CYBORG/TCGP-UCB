import os

import h5py
import pickle
import random
from typing import List
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from Reward import Reward
from Arm import Arm
from problem_models.ProblemModel import ProblemModel

"""
Dynamic probabilistic maximum coverage problem model.
"""

SAVED_FILE_NAME = "movielens_simulation.hdf5"  # file where the saved simulation-ready hdf5 dataset will be written to
TEMP_TIM_PATH = "temp_graphs/"
TIM_EPSILON = 0.1


# def context_to_mean_fun(context):
#     """
#     context[0] = task location
#     context[1] = worker location
#     context[2] = task context
#     context[3] = worker context
#     """
#     return norm.pdf(np.linalg.norm(context[0] - context[1]), loc=0, scale=0.25) * \
#            (context[3] + context[2]) / 2 / norm.pdf(0, loc=0, scale=0.25)


def context_to_sup_mean_fun(context):
    """
    """
    return 1 / (1 + np.exp(5 - 10 * context))


def context_to_grp_mean_fun(context):
    """
    """
    return 0.05 + 0.95 * np.exp(-5 * context)


class MovielensProblemModel(ProblemModel):
    df: pd.DataFrame

    def __init__(self, num_rounds, exp_left_nodes, exp_right_nodes, use_saved, budget, num_locations,
                 dataset_path="dataset_ml-25m", edge_retrain_percentage=0.5,
                 seed=98765, tim_graph_name="", scale_contexts=False,
                 saved_file_name=SAVED_FILE_NAME):
        super().__init__(num_rounds)
        self.saved_file_name = saved_file_name
        self.edge_retrain_percentage = edge_retrain_percentage
        self.scale_contexts = scale_contexts
        self.tim_graph_name = tim_graph_name
        self.dataset_path = dataset_path
        self.exp_left_nodes = exp_left_nodes
        self.exp_right_nodes = exp_right_nodes
        self.num_locations = num_locations
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        if not use_saved:
            self.initialize_dataset()
        self.h5_file = None
        self.benchmark_superarm_list = None

    def set_benchmark_superarm_list(self, superarm_list):
        self.benchmark_superarm_list = superarm_list

    def get_available_arms(self, t):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        t = t - 1
        # Construct a list of Arm objects (i.e., available edges)
        context_dataset = self.h5_file[f"{t}"]["context_dataset"]
        exp_sup_outcome_dataset = self.h5_file[f"{t}"]["mean_dataset"]
        exp_grp_outcome_dataset = self.h5_file[f"{t}"]["grp_mean_dataset"]
        location_dataset = self.h5_file[f"{t}"]["location_dataset"]
        grp_threshold_dataset = self.h5_file[f"{t}"]["grp_threshold_dataset"]

        arm_list = []
        for context, exp_sup_outcome, exp_grp_outcome, group_id in zip(context_dataset, exp_sup_outcome_dataset,
                                                                       exp_grp_outcome_dataset, location_dataset):
            arm_list.append(Arm(len(arm_list), np.array(context).reshape(-1), exp_sup_outcome, exp_grp_outcome,
                                grp_threshold_dataset[group_id], group_id))
        return arm_list

    def get_regret(self, t, budget, slate):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        return 0

    def get_total_reward(self, rewards, t=None):
        return self.superarm_reward_fun(rewards, t)

    def superarm_reward_fun(self, rewards: List[Reward], t=None):
        return sum([reward.sup_perf for reward in rewards])


    def play_arms(self, t, slate):
        reward_list = [Reward(arm, arm.sup_outcome + np.random.normal(0, 0.05), arm.grp_outcome + np.random.normal(0, 0.05)) for arm in slate]
        return reward_list

    def get_perc_satisfied_groups(self, t, rewards: List[Reward]):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        t = t - 1

        grp_threshold_dataset = self.h5_file[f"{t}"]["grp_threshold_dataset"][:]
        group_reward_dict = {}
        for reward in rewards:
            group_id = reward.arm.group_id
            group_reward_dict[group_id] = group_reward_dict.get(group_id, 0) + reward.grp_perf

        num_satisfied_groups = 0
        for group_id, group_reward in group_reward_dict.items():
            if group_reward >= grp_threshold_dataset[group_id]:
                num_satisfied_groups += 1

        return num_satisfied_groups / len(group_reward_dict)

    def sigmoid(self, x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def oracle(self, budget, g_list, t=None):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        # This function assumes that g_list is in the order of available arms list. i.e., the ith element of g_list
        # is the estimate prob (i.e., index/gain) of the ith edge of the edge dataset of t

        t = t - 1  # because time starts from 1 but index starts from 0

        edge_arr = self.h5_file[f"{t}"]["edge_dataset"][:]
        num_edges = len(edge_arr)
        num_nodes = len(np.unique(edge_arr.flatten()))
        if g_list is not np.ndarray:
            g_list = np.array(g_list)
        edge_probs = np.minimum(1, g_list)
        edge_indices = np.argsort(edge_probs)[::-1][:budget]
        return edge_indices

    def get_task_budget(self, t):
        return self.budget

    def initialize_dataset(self):
        print("Generating dataset from MovieLens...")

        # sample number of left and right nodes for all rounds. Needed to allocate h5 dataset
        left_node_count_arr = self.rng.poisson(self.exp_left_nodes, self.num_rounds)
        right_node_count_arr = self.rng.poisson(self.exp_right_nodes, self.num_rounds)

        # create h5 dataset
        h5_file = h5py.File(self.saved_file_name, "w")

        # load movielens dataset
        movies_df = pd.read_csv(f"{self.dataset_path}/movies.csv")
        ratings_df = pd.read_csv(f"{self.dataset_path}/ratings.csv")

        # remove ratings older than Jan 2015
        ratings_df = ratings_df[ratings_df.timestamp > 1420070400]

        genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western', 'IMAX', '(no genres listed)']

        # remove movies with fewer than 200 ratings
        user_id_num_ratings = ratings_df.userId.value_counts()
        filtered_ratings = ratings_df[ratings_df.userId.isin(user_id_num_ratings.index[user_id_num_ratings.ge(200)])]
        filtered_movie_ids = filtered_ratings.movieId.unique()

        # make movie df index movie ID
        filtered_movie_df = movies_df.set_index("movieId").loc[filtered_movie_ids]

        # determine max number of genres in any movie
        max_num_genres = filtered_movie_df["genres"].apply(lambda x: len(x.split("|"))).max()

        movie_id_genre_dict = dict()
        user_id_prefs_dict = dict()  # map of user id to weighted average of genres they rated

        min_context = 1
        max_context = 0
        total_num_edges = 0
        for time in tqdm(range(self.num_rounds)):
            curr_time_group = h5_file.create_group(f"{time}")

            # randomly select left nodes (i.e, movies) and right nodes (i.e., users)
            movie_ids = self.rng.choice(filtered_movie_ids, left_node_count_arr[time], replace=False)

            # filters users who did not rate the selected movies
            temp_ratings = filtered_ratings[filtered_ratings.movieId.isin(movie_ids)]
            user_ids = temp_ratings.userId.unique()
            user_ids = self.rng.choice(user_ids, right_node_count_arr[time], replace=False)

            # filter ratings and movie ids based on chosen users
            temp_ratings = temp_ratings[temp_ratings.userId.isin(user_ids)]
            movie_ids = temp_ratings.movieId.unique()

            # compute movie genre vector and save in dict for future rounds
            for movie_id in movie_ids:
                if movie_id not in movie_id_genre_dict:
                    movie_id_genre_dict[movie_id] = np.array(
                        [1 if g in filtered_movie_df.genres.loc[movie_id] else 0 for g in genres])

            # compute user preference vector and save in dict for future rounds
            for user_id in user_ids:
                if user_id not in user_id_prefs_dict:
                    # weighted sum of the genres of all the movies user has rated
                    user_pref_vec = np.zeros(len(genres))
                    num_movies_user_rated = 0
                    for rating, movie_id in temp_ratings[temp_ratings.userId == user_id][['rating', 'movieId']].values:
                        user_pref_vec += (rating - 0.5) / (5 - 0.5) * movie_id_genre_dict[movie_id]
                        num_movies_user_rated += 1
                    user_id_prefs_dict[user_id] = user_pref_vec / num_movies_user_rated

            # compute edge contexts and save into h5
            # randomly discard some edges
            movie_user_id_list = list(temp_ratings[["movieId", "userId"]].values)
            num_edges_to_keep = int(self.edge_retrain_percentage * len(movie_user_id_list))
            kept_movie_user_ids = self.rng.choice(movie_user_id_list, num_edges_to_keep, replace=False)

            num_edges = num_edges_to_keep
            total_num_edges += num_edges
            edge_dataset_size = (num_edges, 2)  # 2 because left node id and tight node id
            edge_dataset = curr_time_group.create_dataset("edge_dataset", edge_dataset_size, dtype=np.int32)

            context_dataset_size = (num_edges)  # because context is 1-dim
            context_dataset = curr_time_group.create_dataset("context_dataset", context_dataset_size, dtype=float)

            mean_dataset_size = (num_edges)  # because expected outcome is a number
            mean_dataset = curr_time_group.create_dataset("mean_dataset", mean_dataset_size, dtype=float)
            grp_mean_dataset = curr_time_group.create_dataset("grp_mean_dataset", mean_dataset_size, dtype=float)

            location_dataset = curr_time_group.create_dataset("location_dataset", num_edges, dtype=int)
            grp_threshold_dataset = curr_time_group.create_dataset("grp_threshold_dataset", self.num_locations,
                                                                   data=self.rng.uniform(0.5, 4, self.num_locations))

            movie_id_temp_id_dict = {movie_id: i for i, movie_id in enumerate(np.unique(kept_movie_user_ids[:, 0]))}
            user_id_temp_id_dict = {user_id: i for i, user_id in enumerate(np.unique(kept_movie_user_ids[:, 1]))}
            for i, (movie_id, user_id) in enumerate(kept_movie_user_ids):
                context = np.dot(user_id_prefs_dict[user_id], movie_id_genre_dict[movie_id]) / max_num_genres
                sup_mean = context_to_sup_mean_fun(context)
                grp_mean = context_to_grp_mean_fun(context)

                context_dataset[i] = context
                mean_dataset[i] = sup_mean
                grp_mean_dataset[i] = grp_mean
                edge_dataset[i] = (movie_id_temp_id_dict[movie_id], user_id_temp_id_dict[user_id])
                location_dataset[i] = self.rng.choice(self.num_locations)
                min_context = min(context, min_context)
                max_context = max(context, max_context)

        print(f"Average number of edges: {total_num_edges / self.num_rounds}")

        if self.scale_contexts:
            # scale all contexts to [0, 1]
            for t in range(self.num_rounds):
                h5_file[f"{t}"]["context_dataset"][:] = (h5_file[f"{t}"]["context_dataset"] - min_context) / (
                        max_context - min_context)
                h5_file[f"{t}"]["mean_dataset"][:] = context_to_sup_mean_fun(h5_file[f"{t}"]["context_dataset"][:])
        h5_file.close()


if __name__ == '__main__':
    exp_left_nodes = 75  # i.e., expected number of movies in each round
    exp_right_nodes = 200  # i.e., expected number of users in each round
    max_num_basearms = 150
    budget = movielens_budget = 3
    ml_num_locations = 10
    test = MovielensProblemModel(200, exp_left_nodes, exp_right_nodes, False, movielens_budget, num_locations=ml_num_locations,
                                 saved_file_name="temp_movielens.hdf5", dataset_path="dataset_ml-25m")

    print('donerooni')
