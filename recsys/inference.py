import pickle
import argparse
from os.path import abspath, dirname, join

from recsys import evaluate


from surprise import Reader
from surprise import model_selection
from surprise import accuracy
from surprise import Dataset

from surprise import KNNWithMeans
from surprise import SVD, SVDpp
from surprise import SlopeOne, CoClustering

DATA_DIR = join(dirname(dirname(abspath((__file__)))), "data")
OUTPUT_FILE = join(DATA_DIR, "best_model_predictions.pkl")
# best_model_predictions


def load_output(file=OUTPUT_FILE):
    f = open(file, "rb")
    output = pickle.load(f)
    f.close()
    return output


def predictions(user_id, item_id, algo):
    algo = algo
    _, _, _, est, _ = algo.predict(user_id, item_id)

    return est


def rec_top_n_items(user_id, predictions, n=10):
    rec_item_ls = {}
    top_n = evaluate.get_top_n(predictions, n)
    for uid, rating, in top_n.items():
        rec_item_ls[uid] = [iid for (iid, _) in rating]
    return rec_item_ls[user_id]


def main():
    output = load_output()
#    predictions, algo = output["predictions"], output["algo"]
    predictions, algo = output["predictions"], output["algo"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_uid")
    parser.add_argument("--input_iid")
    parser.add_argument("--input_rec_uid")
    parser.add_argument("--input_n", default=10, type=int)

#    parser.add_argument("--input_uid_n", required=True)
    args = vars(parser.parse_args())
    _, _, _, est, _ = algo.predict(args["input_uid"], args["input_iid"])
    rec_ls = rec_top_n_items(
        args["input_rec_uid"], predictions, args["input_n"])

    print("input user id: {}, item id: {}, estimated rating: {}".format(
        args["input_uid"], args["input_iid"], est))
    print("top {} recommended items for input user id {}: {}".format(
        args["input_n"], args["input_rec_uid"], rec_ls))


#    top_n_ls(user_id, n, pred=predictions)
if __name__ == "__main__":
    main()
