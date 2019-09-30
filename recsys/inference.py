"""
Compute estimate for defined user-item, recommend items for defined user.
usage example:

python inference.py --input_uid "A3R27T4HADWFFJ" --input_iid "0005019281" --input_rec_uid "A3R27T4HADWFFJ" --input_n 5

"""
import pickle
import argparse
from os.path import abspath, dirname, join

from recsys.evaluate import get_top_n

from surprise import SVDpp

DATA_DIR = join(dirname(dirname(abspath((__file__)))), "data")
OUTPUT_FILE = join(DATA_DIR, "best_model_predictions.pkl")


def load_output(file=OUTPUT_FILE):
    f = open(file, "rb")
    output = pickle.load(f)
    f.close()
    return output


def rec_top_n_items(user_id, pred, n=10):
    rec_item_ls = {}
    top_n = get_top_n(pred, n)
    for uid, rating, in top_n.items():
        rec_item_ls[uid] = [iid for (iid, _) in rating]
    return rec_item_ls[user_id]


def main():
    output = load_output()
    pred, algo = output["predictions"], output["algo"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_uid", type=str,
                        help="input userid to predict rating")
    parser.add_argument("--input_iid", type=str,
                        help="input itemid to predict rating")
    parser.add_argument("--input_rec_uid", type=str,
                        help="input userid to recommend items")
    parser.add_argument("--input_n", default=10, type=int,
                        help="input # of recommended items")

    args = vars(parser.parse_args())
    uid = args["input_uid"]
    iid = args["input_iid"]
    rec_uid = args["input_rec_uid"]
    n = args["input_n"]
    _, _, _, est, _ = algo.predict(uid, iid)

    rec_ls = rec_top_n_items(args["input_rec_uid"], pred, args["input_n"])
    uid = args["input_uid"]
    print(f'input user id: {uid}, item id: {iid}, estimated rating: {est}')
    print(f'top {n} recommended items for input user id {rec_uid}: {rec_ls}')


if __name__ == "__main__":
    main()
