
from itertools import combinations
import numpy as np
import random

random.seed(0)

def gen_sublists(game_list, length, max_cnt=1000):
    """ generates combinations of game_list with the given length.
    Returns max. max_cnt results."""
    list_perms = list(combinations(game_list, length))

    if len(list_perms) > max_cnt:
        list_perms = random.sample(list_perms, max_cnt)

    return list_perms

def evaluate(recommender, dataset, N=1, K=25):
    """ evaluates recommender on given dataset by iterating through
    each user in the given dataset, removing N games from their
    collection, and then evaluating how much the recommendations match
    with these N games.
    NOTE that this is too slow for N>3 at the moment, because there are
    too many combinations of sublists for the combinations() method.
    """

    tp = fp = fn = 0
    for user in dataset.user_coll.users:
        ## we only need the game ids
        user_game_ids = [int(game.id) for game in user.games]

        # print(f"Iterating through user {user.id} with games {user_game_ids}")
        list_perms = gen_sublists(user_game_ids, N, 50)
        for i, game_sublist in enumerate(list_perms):
            ## take all the user's games and remove those in the minor list
            game_base = set(user_game_ids) - set(game_sublist)
            recs_dict = recommender.generate_recommendation(list(game_base), K)
            recs = [r["appid"] for r in recs_dict]

            tp += sum(1 for game in recs if game in game_sublist)
            fp += sum(1 for game in recs if game not in game_sublist)
            fn += sum(1 for game in game_sublist if game not in recs)

    if tp == 0:
        evals = {
            "Precision": 0,
            "Recall": 0,
            "F-score": 0,
        }
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        evals = {
            "Precision": precision,
            "Recall": recall,
            "F-score": (2 * precision * recall) / (precision + recall),
        }

    return evals

def print_evaluation(evals):
    """simply prints result of single run from evaluate()"""
    print()
    if "title" in evals:
        print(evals["title"])
    for metric, res in evals.items():
        if metric != "title":
            print(f"\t{metric}: {res*100:.2f}%")


def print_evaluation_comparison(evals):
    """compare evaluation results in printed table"""
    colwidth = 15
    print()
    print(f"{'':{colwidth}}", end='')
    for e in evals:
        print(f'{e["title"]:{colwidth}}', end='')
    print()
    for metric in evals[0].keys():
        if metric != "title":
            print(f'{metric:{colwidth}}', end='')
            for e in evals:
                s = f"{e[metric]*100:.2f}%"
                print(f"{s:{colwidth}}", end='')
            print()

