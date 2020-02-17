#!/usr/bin/env python

import unittest
from user_games import Game, DataSet, GameCollection
from collections import defaultdict

class GameFetchTester(unittest.TestCase):

	def test_fetch_valid_game(self):
		## fetch a sample game and test sample attributes
		test_game = {
			"id": 204360,
			"name": "Castle Crashers®",
			"category": 37,
			"developer": "The Behemoth",
		}
		game_obj = Game.from_steam(test_game["id"])

		self.assertTrue(game_obj.id == test_game["id"])
		self.assertTrue(game_obj.name == test_game["name"])
		self.assertTrue(test_game["category"] in game_obj.categories)
		self.assertTrue(test_game["developer"] in game_obj.developers)

	def test_fetch_invalid(self):
		game_obj = Game.from_steam("invalidid")

		self.assertTrue(game_obj.name == "")

class DataSetTester(unittest.TestCase):

	# def test_creation(self):
	# 	""" basic sanity test based on the test CSV"""
	# 	ds = DataSet.from_csv("data/test_set.csv")
	# 	ds.game_coll.print()
	# 	ds.user_coll.print()

	# 	self.assertTrue(len(ds.game_coll.games) == 9)
	# 	self.assertTrue(len(ds.user_coll.users) == 9)

	# 	users_w_1game = [user for user in ds.user_coll.users if len(user.games) == 1]
	# 	self.assertTrue(len(ds.user_coll.users) -1 == len(users_w_1game))

	# def test_gamecoll_dump(self):
	# 	""" test json dumping & loading of game collection"""
	# 	gc = GameCollection.from_steam_ids([730, 333600])

	# 	gc.save_as_json("gc.json")

	# 	gc_json = GameCollection.from_json("gc.json")

	# 	self.assertTrue(len(gc.games) == len(gc_json.games))
	# 	self.assertTrue(gc_json.contains(730))	
	# 	self.assertTrue(gc_json.contains(333600))	

	def test_fullset(self):
		ds = DataSet.from_csv("data/test_set.csv", json_path="data/full_games.json")
		self.assertTrue(len(ds.game_coll.games) == 923)

if __name__ == '__main__':
	unittest.main()