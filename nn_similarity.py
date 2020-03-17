import torch.nn as nn
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# from torchviz import make_dot

import numpy as np
import argparse

import random

import string
from collections import defaultdict

from user_games import DataSet

device = torch.device("cpu")
if torch.cuda.is_available():
	device = torch.device("cuda:3")

random.seed(0)

def cosine_sim(v1, v2):
	""" simply calculate the cosine similarity between two vectors."""
	dot = np.dot(v1, v2)
	norm1 = np.linalg.norm(v1)
	norm2 = np.linalg.norm(v2)
	cos = dot / (norm1 * norm2)
	return cos

class GameNetwork(nn.Module):

	def __init__(self, vocab_size, vector_size):
		super().__init__()
		## reserve 1 for new user embedding
		self.vector_size = vector_size
		self.embedding_layer = nn.Embedding(vocab_size, vector_size)
		self.user_embedding = nn.Embedding(1, vector_size)
		self.user_idx_tensor = torch.zeros([1, 1], dtype=torch.int64).to(device)

	def forward(self, anchor, pos, neg):
		anchor_vec = self.embedding_layer(anchor)
		pos_vec = self.embedding_layer(pos)
		neg_vec = self.embedding_layer(neg)
		return (anchor_vec.view(-1, 1), pos_vec.view(-1, 1), neg_vec.view(-1, 1))

	def infer_user(self, neg, pos):
		""" calculates a new embedding for a new user, without
		changing any of the other embeddings."""
		anchor_vec = self.user_embedding(self.user_idx_tensor)
		anchor_vec = torch.cat(pos.shape[0]*[anchor_vec.view(1, self.vector_size)])
		pos_vec = self.embedding_layer(pos)
		neg_vec = self.embedding_layer(neg)
		return (anchor_vec.view(-1, 1), pos_vec.view(-1, 1), neg_vec.view(-1, 1))

	@classmethod
	def load_from_file(cls, model_path, vocab_size, vector_size):
		""" attempt to load model from file. 
		will work in all necessary cases, but is definitely not
		robust to more general use. (e.g. won't detect dropout 
		layers etc.)"""
		state_dict = torch.load(path)

		network = cls(vocab_size, vector_size)
		network.load_state_dict(state_dict)

		return network

	def get_user_embedding(self):
		self.eval()
		return self.user_embedding(self.user_idx_tensor)

	def get_eval_embedding(self, idx):
		self.eval()
		return self.embedding_layer(idx)

	def reset_user_vec(self):
		torch.manual_seed(32)
		self.user_embedding.weight.data.uniform_(-1, 1)

def get_game_matrix(gnet, games_idx, model_obj_idx):
	game_vecs = []
	for i, game in enumerate(games_idx):
		game_obj_idx = model_obj_idx.index(f"game_{game}")
		game_idx_tensor = torch.tensor(game_obj_idx).to(device)
		game_vecs.append(gnet.get_eval_embedding(game_idx_tensor).detach().cpu().numpy())

	game_mat = np.stack(game_vecs)
	return game_mat

def get_sim_matrix(gnet, games_idx, model_obj_idx):
	game_vecs = {}
	for i, game in enumerate(games_idx):
		game_obj_idx = model_obj_idx.index(f"game_{game}")
		game_idx_tensor = torch.tensor(game_obj_idx).to(device)
		game_vecs[i] = gnet.get_eval_embedding(game_idx_tensor).detach().cpu().numpy()

	games_cnt = len(games_idx)
	sim_mat = np.zeros((games_cnt, games_cnt))
	for i in range(games_cnt):
		for j in range(games_cnt):
			sim_mat[i][j] = cosine_sim(game_vecs[i], game_vecs[j])

	print("it's filled")

	return sim_mat

class TripletLoss(nn.Module):
	"""
	Triplet loss
	Takes embeddings of an anchor sample, a positive sample and a negative sample
	"""

	def __init__(self, margin):
		super(TripletLoss, self).__init__()
		self.margin = margin

	def forward(self, anchor, positive, negative, size_average=True):
		distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
		distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
		losses = F.relu(distance_positive - distance_negative + self.margin)
		return losses.mean() if size_average else losses.sum()


class NewUserTripletsDataset(data.Dataset):

	def __init__(self, triplets, uniq_objs):
		super().__init__()
		self.triplets = triplets
		self.obj_keys = uniq_objs

	def __len__(self):
		return len(self.triplets)

	def item_to_tensor(self, item):
		idx = self.obj_keys.index(item)
		return torch.tensor(idx).to(device)

	def __getitem__(self, index):
		pos = self.item_to_tensor(self.triplets[index][1])
		neg = self.item_to_tensor(self.triplets[index][2])
		return pos, neg

class TripletsDataset(data.Dataset):

	def __init__(self, triplets, uniq_objs):
		super().__init__()
		self.triplets = triplets
		self.obj_keys = uniq_objs

	def __len__(self):
		return len(self.triplets)

	def item_to_tensor(self, item):
		idx = self.obj_keys.index(item)
		return torch.tensor(idx).to(device)

	def __getitem__(self, index):
		anchor = self.item_to_tensor(self.triplets[index][0])
		pos = self.item_to_tensor(self.triplets[index][1])
		neg = self.item_to_tensor(self.triplets[index][2])
		return anchor, pos, neg

def create_reversed_dicts(dataset):
	""" creates for each user, genre, platform etc. a dict which maps it
	to all games that are associated with it."""
	game_dict = defaultdict(set)

	for user in dataset.user_coll.users:
		game_dict[f"user_{user.id}"] = set([f"game_{g.id}" for g in user.games])

	for game in dataset.game_coll.games.values():
		# for obj in game.developers:
		# 	game_dict[f"dev_{obj}"].add(f"game_{game.id}")
		# for obj in game.platforms:
		# 	game_dict[f"platf_{obj}"].add(f"game_{game.id}")
		# for obj in game.categories:
		# 	game_dict[f"cat_{obj}"].add(f"game_{game.id}")
		for obj in game.genres:
			game_dict[f"genre_{obj}"].add(f"game_{game.id}")
		# for obj in game.publishers:
		# 	game_dict[f"pub_{obj}"].add(f"game_{game.id}")

	print(f"full length: {len(game_dict.keys())}")
	game_dict = {k: v for k, v in game_dict.items() if len(v) > 1 \
											or k.startswith("game_")}
	print(f"filtered length: {len(game_dict.keys())}")

	return game_dict

def create_training_set(game_dict, all_games, repeat=1):
	""" creates training triplets from dataset with the given vocabulary."""
	triplets = []
	for tag, games in game_dict.items():
		for game in games:
			anchor = tag
			pos = game
			## choose negative 
			other_games = all_games - games
			# print(f"\t{anchor}, {repeat}, {len(other_games)}")
			for i in range(min(repeat, len(other_games))):
				neg = random.sample(other_games, 1)[0]
				triplets.append((anchor, pos, neg))
				other_games.remove(neg)
				# print(f"\ttrip: {anchor}, {pos}, {neg}")

	return triplets

def train_batch(model, triplets, opt, criterion, infer_user=False):
	## prepare network
	model.train()
	opt.zero_grad()

	## feed data through network
	if not infer_user:
		y_hat = model(*triplets)
	else:
		y_hat = model.infer_user(*triplets)
	loss = criterion(*y_hat)

	## optimize network
	loss.backward()
	opt.step()

	return loss

def calc_user_embedding(gnet, games, games_idx, obj_to_idx, n_epochs=4):
	## reserved idx for new embedding
	user_gdict = {"new_user": set(games)}
	user_trips = create_training_set(user_gdict, set(games_idx), 
		repeat=10)

	triplets_set = NewUserTripletsDataset(user_trips, obj_to_idx)
	train_generator = data.DataLoader(triplets_set, shuffle=True,
		batch_size=1)

	for param in gnet.embedding_layer.parameters():
		param.requires_grad = False

	## start fresh so we're not biased by the last inference
	gnet.reset_user_vec()
	# print("calculated user vec:")
	# print(f"\tfrom {gnet.get_user_embedding()}")
	fit_net(gnet, train_generator, n_epochs=n_epochs, lr=0.001, 
		infer_user=True)
	# print(f"\tto   {gnet.get_user_embedding()}")


def fit_net(gnet, triplets_set, n_epochs=40, lr=0.001, infer_user=False):
	loss_fn = TripletLoss(1)
	if infer_user:
		opt = optim.SGD(gnet.parameters(), lr=0.01, momentum=0.9)
	else:
		# opt = optim.SGD(gnet.parameters(), lr=0.01, momentum=0.9)
		opt = optim.Adam(gnet.parameters(), lr=lr)
	# scheduler = StepLR(opt, step_size=1, gamma=0.1)

	for epoch in range(n_epochs):
		# scheduler.step()
		losses = []
		## iterate through batches
		for i, triplets in enumerate(triplets_set):
			loss = train_batch(gnet, triplets, opt, loss_fn, infer_user)
			## save loss and print state
			losses.append(loss.item())
			loss_str = round(loss.item(), 3)
			print(f"\r{epoch+1:{4}}/{i+1:{4}}\tloss: {loss_str:{6}}", end="")

		## evaluate on test set
		avg_loss = round(sum(losses) / len(losses), 2)
		print(f"\r{epoch+1:{4}}/{i+1:{4}}\tloss: {avg_loss:{6}}", end="")
		print()
		# evaluate_nn(network, x_test_torch, y_test)

		## save model after each epoch
		torch.save(gnet.state_dict(), "model_saved.pt")

def prepare_dataset(dataset, repeat_trips):
	""" creates two objects: a list of triplets used for
	training, and a list of all embedded objects in the
	order of their idx reference."""
	## creates dict with object -> games
	game_dict = create_reversed_dicts(dataset)

	all_games = set([f"game_{g.id}" for g in dataset.game_coll.games.values()])

	train_set = create_training_set(game_dict, all_games,
		repeat=repeat_trips)

	all_objs = set(game_dict.keys()) | set(all_games)

	print(f"length of training set: {len(train_set)}")

	return train_set, sorted(list(all_objs))


def main(vec_size=100, repeat_trips=1, epochs=40):
	ds = DataSet.from_csv("data/user_games.csv", json_path="data/full_games.json")

	train_set, obj_to_idx = prepare_dataset(ds, repeat_trips=repeat_trips)

	triplets_set = TripletsDataset(train_set, obj_to_idx)
	train_generator = data.DataLoader(triplets_set, shuffle=True,
		batch_size=32)

	gnet = GameNetwork(len(obj_to_idx), vec_size).to(device)
	fit_net(gnet, train_generator, n_epochs=epochs)

	return (gnet, obj_to_idx)

if __name__ == '__main__':
	## parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--vector_size", 
		required=True,
		type=int)
	parser.add_argument("--repeat_trips", 
		default=1,
		type=int)
	parser.add_argument("--epochs", 
		default=40,
		type=int)
	args = parser.parse_args()

	main(args.vector_size, args.repeat_trips, args.epochs)
