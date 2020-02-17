import urllib.request
import json
from urllib.error import HTTPError
from typing import Set
import csv
from collections import defaultdict, Counter
import time
import random
import numpy as np
import pandas as pd

## snippet from https://stackoverflow.com/a/12965254
def fetch_json(url):
	""" downloads json from url and returns it, None otherwise"""
	try:
		with urllib.request.urlopen(url) as page:
			json_data = json.loads(page.read().decode())
			return json_data
	except HTTPError:
		print(f"Warning: couldn't fetch {url}")
		return None

def read_csv(csv_path):
	with open(csv_path, "r", encoding="utf8") as f:
		reader = csv.reader(f, delimiter="\t")
		## skip header row
		next(reader)
		return list(reader)

class Game(object):

	DELAY = 10
	last_req = 0

	def __init__(self, attr_dict):
		self.id = int(attr_dict["id"])
		self.name = attr_dict["name"]
		self.developers = attr_dict["developers"]
		self.publishers = attr_dict["publishers"]
		self.price = attr_dict["price"]
		self.platforms = attr_dict["platforms"]
		self.categories = [f"cat_{c}" for c in attr_dict["categories"]]
		self.genres = [f"genre_{g}" for g in attr_dict["genres"]]
		self.recommendations = attr_dict["recommendations"]
		self.release_year = attr_dict["release_year"]
		self.meta_score = attr_dict["meta_score"]

	@staticmethod
	def steam_delay():
		""" to avoid spamming the Steam website with requests, we introduce a 
		delay here."""
		current_time = time.time()
		if Game.last_req + Game.DELAY > current_time:
			print("sleeping...")
			time.sleep(Game.DELAY)
		Game.last_req = current_time

	@classmethod
	def from_steam(cls, appid):
		Game.steam_delay()
		url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
		full_json = fetch_json(url)

		try:
			## top node is appid again
			game_details = full_json[str(appid)]["data"]
		except (KeyError, TypeError):
			print(f"Warning: game {appid} has incomplete game information")
			game_details = {}

		attr_dict = {"id": appid}
		attr_dict["name"] = game_details.get("name", "")
		attr_dict["developers"] = game_details.get("developers", [])
		attr_dict["publishers"] = game_details.get("publishers", [])
		attr_dict["price"] = game_details.get("price_overview", {}).get("final", 0)
		attr_dict["platforms"] = [p for p,avail in game_details.get("platforms", {}).items() if avail]
		attr_dict["categories"] = [val for cat in game_details.get("categories", [{}]) \
			for attr, val in cat.items() if attr == "id"]
		attr_dict["genres"] = [val for cat in game_details.get("genres", [{}]) \
			for attr, val in cat.items() if attr == "id"]
		attr_dict["recommendations"] = game_details.get("recommendations", {}).get("total", 0)
		attr_dict["release_year"] = game_details.get("release_date", {}).get("date", "2020")[-4:]
		attr_dict["meta_score"] = game_details.get("metacritic", {}).get("score", 50)

		return cls(attr_dict)

class GameCollection(object):

	def __init__(self, games: Set[Game]=set()):
		""" keeps dictionary with game_id -> Game"""
		self.games = {g.id: g for g in games}
		self.games_idx = sorted(self.games.keys())

	@classmethod
	def from_steam_ids(cls, appids):
		games = [Game.from_steam(appid) for appid in appids]
		return cls(games)

	def contains(self, game_id):
		return game_id in self.games

	def add_game(self, game):
		self.games[game.id] = game

	def get(self, game_id):
		return self.games[game_id]

	def print(self):
		""" just to show the games"""
		print("Game Collection:")
		for game in self.games.values():
			print(f"\t{game.name} ({game.id}), released {game.release_year} with score {game.meta_score} ({game.price}€)")

	def save_as_json(self, json_path):
		game_dicts = [game.__dict__ for game in self.games.values()]
		with open(json_path, "w", encoding="utf8") as f:
			json.dump(game_dicts, f, indent=4)

	@classmethod
	def from_json(cls, json_path):
		games = set()
		with open(json_path, "r", encoding="utf8") as f:
			game_data = json.load(f)
			for g in game_data:
				games.add(Game(g))

		return cls(games)

	def to_feature_matrix(self):
		"""converts the game collection into a matrix with one row
		per game and one column per feature."""
		game_dicts = [game.__dict__ for game in self.games.values()]
		# print(game_dicts)
		df = pd.DataFrame(game_dicts)
		genre_dummies = df["genres"].str.join(sep='*').str.get_dummies(sep="*")
		devs_dummies = df["developers"].str.join(sep='*').str.get_dummies(sep="*")
		pubs_dummies = df["publishers"].str.join(sep='*').str.get_dummies(sep="*")
		plats_dummies = df["platforms"].str.join(sep='*').str.get_dummies(sep="*")
		cats_dummies = df["categories"].str.join(sep='*').str.get_dummies(sep="*")

		## remove values that occur very rarely
		genre_dummies = genre_dummies.loc[:, (genre_dummies.sum(axis=0) >= 5)]
		devs_dummies = devs_dummies.loc[:, (devs_dummies.sum(axis=0) >= 5)]
		pubs_dummies = pubs_dummies.loc[:, (pubs_dummies.sum(axis=0) >= 5)]
		cats_dummies = cats_dummies.loc[:, (cats_dummies.sum(axis=0) >= 5)]

		## combine dummies
		df_complete = pd.concat([df["id"], genre_dummies, devs_dummies, pubs_dummies, cats_dummies],
						axis=1)
		## add index from games_idx and sort by it
		df_complete["idx"] = df_complete["id"].apply(lambda x: self.games_idx.index(x))
		df_complete.sort_values(by=["idx"], axis=0, ascending=True, inplace=True)
		df_complete.drop(["id"], inplace=True, axis=1)
		df_complete.set_index(["idx"], inplace=True)

		## normalize rows/games, so that e.g. games with many developers are not
		## automatically more likely to be regarded as similar
		df_norm = df_complete.div(df_complete.sum(axis=1), axis=0)
		df_norm.fillna(0, inplace=True)

		return df_complete

		# colls = defaultdict(list)
		# for g in self.games.values():
		# 	colls["dev"] += g.developers
		# 	colls["pubs"] += g.publishers
		# 	colls["plats"] += g.platforms
		# 	colls["cats"] += g.categories
		# 	colls["genres"] += g.genres
		# for item, cnt in Counter(devs_all).most_common():
		# 	print(f"\t{item}\t{cnt}") ## 5
		# for item, cnt in Counter(colls["pubs"]).most_common():
		# 	print(f"\t{item}\t{cnt}") ## 5
		# for item, cnt in Counter(colls["plats"]).most_common():
		# 	print(f"\t{item}\t{cnt}")
		# for item, cnt in Counter(colls["cats"]).most_common():
		# 	print(f"\t{item}\t{cnt}")
		# for item, cnt in Counter(colls["genres"]).most_common():
		# 	print(f"\t{item}\t{cnt}")


class User(object):

	def __init__(self, user_id: str, games: Set[Game]=set()):
		self.id = user_id
		self.games = games

class UserCollection(object):

	def __init__(self, users: Set[User]=set()):
		self.users = users

	def print(self):
		""" just to show users"""
		print("User Collection:")
		for user in self.users:
			user_games = [game.name for game in user.games]
			print(f"\t{user.id}: {', '.join(user_games)}")

class DataSet(object):

	def __init__(self, user_coll: UserCollection, game_coll: GameCollection):
		self.user_coll = user_coll
		self.game_coll = game_coll

	@property
	def games_idx(self):
		return self.game_coll.games_idx

	@classmethod
	def from_csv(cls, csv_path, json_path=None):
		""" create UserCollection and GameCollection from csv file.
		takes a json file with game descriptions, if it already exists."""
		rows = read_csv(csv_path)

		games = GameCollection()
		if json_path:
			games = GameCollection.from_json(json_path)
		user_to_games = defaultdict(set)
		## iterate through rows and fill games&users
		for user_id, game_id in rows:
			if games.contains(int(game_id)):
				game = games.get(int(game_id))
			else:
				game = Game.from_steam(game_id) 
				games.add_game(game)
			user_to_games[user_id].add(game)

		## create user collection
		users = set(User(user_id, user_games) \
			for user_id, user_games in user_to_games.items())
		user_coll = UserCollection(users)

		return cls(user_coll=user_coll, game_coll=games)

	def shuffle_users(self):
		""" shuffle users for better testing."""
		random.seed(0) ## for reproducibility
		self.user_coll.users = set(random.sample(list(self.user_coll.users), 
												len(self.user_coll.users)))

	def split_users(self, ratio):
		""" splits the UserCollection of this dataset;
		returns a dataset with a subset of the users and removes those
		users from this dataset."""
		user_cnt = len(self.user_coll.users)
		split_point = int(user_cnt * ratio)

		## create two new collections
		user_list = list(self.user_coll.users)
		first_user_coll = UserCollection(set(user_list[:split_point]))
		first_users_ds = DataSet(first_user_coll, self.game_coll)
		## second
		sec_user_coll = UserCollection(set(user_list[split_point:]))
		sec_users_ds = DataSet(sec_user_coll, self.game_coll)

		return (first_users_ds, sec_users_ds)

	def to_matrix(self):
		""" converts dataset into matrix with users * items """
		mat = np.zeros((len(self.user_coll.users), len(self.games_idx)))
		## fill cells with 1 when user owns that game
		for i, user in enumerate(self.user_coll.users):
			for game in user.games:
				game_idx = self.games_idx.index(game.id)
				mat[i, game_idx] = 1

		return mat