#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot  # it will be used through pd.plot
from sklearn.metrics.pairwise import cosine_similarity


def get_top_n(predictions, n):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def personalization(prediction, n):
    # prediction
    # n top n recommendation

    top_n = get_top_n(prediction, n)

    rec_dict = {}
    for uid, user_ratings in top_n.items():
        rec_dict[uid] = [iid for (iid, _) in user_ratings]

    rec_user_ls = [pred[0] for pred in prediction]
    rec_item_ls = [pred[1] for pred in prediction]

    unique_rec_user_ls = np.unique(rec_user_ls)
    unique_rec_item_ls = np.unique(rec_item_ls)

    # assign each item with index number
    unique_rec_item_dict = {item: ind for ind,
                            item in enumerate(unique_rec_item_ls)}

    n_unique_rec_user = len(unique_rec_user_ls)
    n_unique_rec_item = len(unique_rec_item_ls)

    # recommended user item matrix
    rec_matrix = np.zeros(shape=(n_unique_rec_user, n_unique_rec_item))

    # represent recommended item for each user as binary 0/1
    for user in range(n_unique_rec_user):
        # get userid
        user_id = unique_rec_user_ls[user]
        # get rec item list
        item_ls = rec_dict[user_id]

        for item_id in item_ls:
            # get item index
            item = unique_rec_item_dict[item_id]
            rec_matrix[user, item] = 1

    # calculate cosine similarity matrix across all user recommendations
    similarity = cosine_similarity(X=rec_matrix, dense_output=False)
    # calculate average of upper triangle of cosine matrix
    upper_right = np.triu_indices(similarity.shape[0], k=1)
    # personalization is 1-average cosine similarity
    score = 1 - np.mean(similarity[upper_right])
    return score


def precision_recall_at_k(predictions, k, threshold):
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    # Precision and recall can then be averaged over all users
    overall_precisions = sum(
        prec for prec in precisions.values()) / len(precisions)
    overall_recalls = sum(rec for rec in recalls.values()) / len(recalls)

    return overall_precisions, overall_recalls


def metrics_dataframe(metrics):
    """Convert metrics dictionary into three groups of pandas dataframe.

    Parameters
    ----------
    metrics : dict
        keys: rmse, precision, recall, fit time, prediction time,
                personalization, algorithm name
        items: list of 5 fold cross validation measurement

    Returns
    -------
    type
        list of pandas dataframe
            df_precision, df_recall, df_general_metrics

    """
    general_metrics = {}
    precision = []
    recall = []

    general_metrics["rmse"] = [np.mean(i) for i in metrics["cv_rmse"]]
    general_metrics["fit_time"] = [np.mean(i) for i in metrics["cv_fit_time"]]
    general_metrics["pred_time"] = [
        np.mean(i) for i in metrics["cv_pred_time"]]
    general_metrics["personalization"] = [
        np.mean(i) for i in metrics["cv_personalization"]]
    general_metrics["algo_name"] = metrics["algo_name"]

    for i in metrics["cv_precision"]:
        precision.append({k: np.mean(v) for k, v in i.items()})
    for i in metrics["cv_recall"]:
        recall.append({k: np.mean(v) for k, v in i.items()})

    df_general_metrics = pd.DataFrame(general_metrics).set_index("algo_name")
    df_precision = pd.DataFrame(precision, index=metrics["algo_name"]).T
    df_recall = pd.DataFrame(recall, index=metrics["algo_name"]).T

    return df_precision, df_recall, df_general_metrics


def plot_precision_recall_k(df_precision, df_recall):
    ax1 = df_precision.plot.line(
        title="Precision COmparsion of Algo at different k")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Precision")

    ax2 = df_recall.plot.line(title="Recall COmparsion of Algo at different k")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Recall")

    return ax1, ax2


def show_results(metrics):
    df_precision, df_recall, df_metrics = metrics_dataframe(metrics)
    print("RMSE, Personalization, Fit Time, Preidction Time")
    print(df_metrics)
    print("Precision Table")
    print(df_precision)
    print("Recall Table")
    print(df_recall)
    plot_precision_recall_k(df_precision, df_recall)
