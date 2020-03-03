
from collections import defaultdict
from user_games import DataSet
from itertools import combinations, islice
import numpy as np
import pandas as pd
import random
import evaluation
import time
from sklearn.decomposition import NMF
import nn_similarity

random.seed(0)

def cosine_sim(v1, v2):
	""" simply calculate the cosine similarity between two vectors."""
	dot = np.dot(v1, v2)
	norm1 = np.linalg.norm(v1)
	norm2 = np.linalg.norm(v2)
	cos = dot / (norm1 * norm2)
	return cos

class Recommender(object):

	def preprocess_game_list(self, game_list):
		""" returns a vector representation of a user with the
		given games."""		
		## remove unknown games
		game_list = [g for g in game_list if g in self.games_idx]

		## create list with all games and set to 1 if it's in game list
		game_ratings = []
		for game in self.games_idx:
			if game in game_list:
				game_ratings.append(1)
			else:
				game_ratings.append(0)

		ratings_arr = np.asarray(game_ratings).reshape(1, -1)

		return ratings_arr

	def train(self, dataset):
		""" Train/prepare the model."""
		## implemented in specific classes
		pass

	def generate_recommendation(self, game_list, count):
		""" Generate recommendations.
		Returns a dictionary of rank, product_id and score.
		Score is a number between 0.0 and 1.0."""
		user_array = self.preprocess_game_list(game_list)
		preds = self.generate_recommendation_core(user_array)
		return self.normalize_preds(preds, game_list, count)


	def normalize_preds(self, preds, game_list, count):
		""" normalize values to probability values and return a dictionary."""
		## remove negatives
		min_val = preds.min()
		if min_val < 0:
			preds += abs(min_val)
		## scale to sum of 1
		preds /= preds.sum()

		preds_series = pd.Series(preds).sort_values(ascending=False)

		## create dictionary with recommendations
		recs = []
		rank = 1
		for game_index, proba in preds_series.items():
			## skip games that the user already owns
			if self.games_idx[game_index] not in game_list:
				recs.append({
					"rank": rank,
					"appid": self.games_idx[game_index],
					"proba": proba
					})
				rank += 1
			if len(recs) == count:
				break

		return recs


class RecommenderPop(Recommender):
	## Popularity based recommender

	def __init__(self):
		super().__init__()
		self.game_array = None

	def generate_recommendation_core(self, user_array):
		""" not adapting anything to specific users."""
		return self.game_array

	def train(self, dataset):
		""" Create an numpy array with game popularity counts."""

		self.games_idx = dataset.games_idx
		game_pop = [0]*len(self.games_idx)

		## count how often each game is owned
		for user in dataset.user_coll.users:
			for game in user.games:
				game_pop[self.games_idx.index(game.id)] += 1

		self.game_array = np.asarray(game_pop, dtype=np.float64).reshape(1, -1)[0]

class RecommenderMatrixFac(Recommender):
	## Matrix Factorization with SGD

	def __init__(self, K, learning_rate=0.01, num_epochs=50):
		super().__init__()
		self.alpha = learning_rate
		self.num_epochs = num_epochs
		self.K = K

	def train(self, dataset):
		""" creates two matrices P,Q and tries to reconstruct matrix
		P by its multiplication. This way, a more dense representation
		is calculated for users as well as items.

		https://www.kaggle.com/robottums/probabalistic-matrix-factorization-with-suprise
		"""
		train_matrix = dataset.to_matrix()
		self.games_idx = dataset.games_idx

		users_n = train_matrix.shape[0]
		games_n = train_matrix.shape[1]
		#randomly initialize user/item factors
		P = np.random.normal(0, .1, (users_n, self.K))
		Q = np.random.normal(0, .1, (games_n, self.K))

		## repeat learning process
		for epoch in range(self.num_epochs):
			## iterate through all cells
			for user in range(users_n):
				for game in range(games_n):
					rating = train_matrix[user][game]
					residual = rating - np.dot(P[user],Q[game])
					## update matrices
					temp = P[user,:] 
					P[user,:] +=  self.alpha * residual * Q[game]
					Q[game,:] +=  self.alpha * residual * temp 

		self.P = P
		self.Q = Q
		self.train_matrix = train_matrix

	def generate_recommendation_core(self, user_array):
		""" construct user embedding and then predict games"""
		userW = np.dot(user_array, self.Q)
		game_preds = np.dot(userW, self.Q.transpose())

		return game_preds[0]

class RecommenderNMF(RecommenderMatrixFac):
	## instead of factorizing the matrix myself, we use scikit
	## recommendation generation works as with RecommenderMatrixFac

	def __init__(self, K):
		super().__init__(K)

	def train(self, dataset):
		train_matrix = dataset.to_matrix()

		model = NMF(n_components=self.K, init="random", random_state=0)
		self.P = model.fit_transform(train_matrix)
		self.Q = model.components_.transpose()
		self.train_matrix = train_matrix
		self.games_idx = dataset.games_idx
				
class RecommenderItemSim(Recommender):
	## Recommender based on item similarity

	def __init__(self):
		super().__init__()

	def train(self, dataset):
		self.games_idx = dataset.games_idx
		self.feature_mat = dataset.game_coll.to_feature_matrix()

	def generate_recommendation_core(self, user_array):
		""" take game_list and choose those games that are most similar.
		this is done by a simple matrix multiplication; most likely 
		cosine similarity would be better, but it's too slow for now. """

		## get game feature rows where user_array is 1
		indices = np.argwhere(user_array[0] == 1)
		flat_inds = indices.flatten().tolist()
		game_matrix = self.feature_mat.loc[flat_inds]

		## calculate game similarities with matrix multiplication
		res_mat = np.dot(game_matrix, self.feature_mat.transpose())
		games_sims = res_mat.max(axis=0).astype('float64')

		return games_sims

		# # with cosine similarity
		# for idx in g_indices:
		# 	vec = self.feature_mat.iloc[idx]
		# 	## calculate similarities from game to all others
		# 	# sims.append(self.feature_mat.apply(cosine_sim, axis=1, v2=vec))

		# ## now we have a list of similarities, find the most similar now
		# sims_df = pd.DataFrame(sims)
		# games_sims = sims_df.sum(axis=0).sort_values(0, ascending=False).index
		# #


class TorchCommender(Recommender):

	def __init__(self, K):
		super().__init__()
		self.K = K

	def train(self, dataset):
		self.games_idx = dataset.games_idx
		self.feature_mat = nn_similarity.main(vec_size=self.K, 
			repeat_trips=5, epochs=10)

	def generate_recommendation_core(self, user_array):
		prod_mat = np.dot(user_array, self.feature_mat)
		return prod_mat.reshape(-1,)


def main():
	evalK = 10
	evalN = 3
	K = 30

	## load and create dataset
	ds = DataSet.from_csv("data/user_games.csv", json_path="data/full_games.json")
	ds.shuffle_users()
	train_ds, test_ds = ds.split_users(ratio=0.8)
	print("created datasets")

	## create recommenders
	recs = [
		("Pop", RecommenderPop()),
		("ItemSim", RecommenderItemSim()),
		("MatFac", RecommenderMatrixFac(K)),
		("NMF", RecommenderNMF(K)),
		("Torch", TorchCommender(K)),
	]

	# ## train and evaluate recommenders
	evals = []
	for name, rec in recs:
		print(f"evaluating {name}...")
		rec.train(train_ds)
		ev = evaluation.evaluate(rec, test_ds, evalN, evalK)
		ev["title"] = name
		evals.append(ev)
	evaluation.print_evaluation_comparison(evals)

	## actual example
	print()
	game_list = [10, 80, 240, 730] ## all counterstrike games
	count = 5
	for name, rec in recs:
		print(f"Recommendations from {name}:")
		print(rec.generate_recommendation(game_list, count))

if __name__ == '__main__':
	main()