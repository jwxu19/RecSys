import pickle
import pandas as pd
from os.path import abspath, dirname, join
import ast

DATA_DIR = join(dirname(dirname(abspath((__file__)))),
                "data")
GAME_FILE = join(DATA_DIR, "steam_games.json")
AU_REVIEW_FILE = join(DATA_DIR, "australian_user_reviews.json")
AU_ITEM_FILE = join(DATA_DIR, "australian_users_items.json")
PKL_FILE1 = join(DATA_DIR, "filtered_explicit_data.pkl")
PKL_FILE2 = join(DATA_DIR, "implicit_data.pkl")
PKL_FILE3 = join(DATA_DIR, "details.pkl")
PKL_FILE4 = join(DATA_DIR, "review_text.pkl")
PKL_FILE5 = join(DATA_DIR, "dashboard_item_summary.pkl")
PKL_FILE6 = join(DATA_DIR, "dashboard_user_summary.pkl")


def load_data(file):
    data = []
    for line in open(file, "r"):
        data.append(ast.literal_eval(line))
    return data


def save_data(data, file):
    outfile = open(file, "wb")
    pickle.dump(data, outfile)
    outfile.close()


def get_review_ls(review_info):
    review_ls = []
    review_ls_name = ["user_id", "item_id", "recommend", "review"]
    for user in review_info:
        for review in user["reviews"]:
            temp_ls = [user["user_id"], review["item_id"],
                       review["recommend"], review["review"]]
            review_ls.append(temp_ls)
    df_review = pd.DataFrame(review_ls, columns=review_ls_name)
    return df_review


def review_by_item_summary(df_review):
    df_review = df_review.drop("review", axis=1)
    df_num_review_game = df_review.groupby("item_id")["user_id"].count(
    ).reset_index().rename(columns={"user_id": "n_review"})
    df_num_recommend_game = df_review.groupby("item_id")["recommend"].sum(
    ).reset_index().rename(columns={"recommend": "n_recommend"})
    df_review_item = df_num_review_game.merge(
        df_num_recommend_game,
        how="left", on="item_id")

    return df_review_item


def get_item_detail(df_review_item, game_info):
    df_game = pd.DataFrame(game_info)
    remain_col = ['id', 'app_name', 'publisher', 'developer', 'genres',
                  'price', 'early_access', 'release_date']
    df_game = df_game[remain_col]
    # this is for dashboard, visualize game info
    df_review_item_overall = df_review_item.merge(df_game, how="left",
                                                  left_on="item_id",
                                                  right_on="id")

    return df_review_item_overall


def review_by_user_summary(df_review):
    df_num_review_user = df_review.groupby("user_id")["item_id"].count(
    ).reset_index().rename(columns={"item_id": "n_review"})
    df_num_recommend_user = df_review.groupby("user_id")["recommend"].sum(
    ).reset_index().rename(columns={"recommend": "n_recommend"})
    df_review_user = df_num_review_user.merge(df_num_recommend_user,
                                              how="left", on="user_id")

    return df_review_user


def get_user_items_detail(user_info, df_review_user):
    # this is for dashboard user behavior distribution table and graph
    user_items_ls = []
    user_items_ls_name = ["user_id", "item_count",
                          "item_id", "item_name", "playtime_forever",
                          "playtime_2weeks"]
    for user in user_info:
        for item in user["items"]:
            temp_ls = [user["user_id"], user["items_count"],
                       item["item_id"], item["item_name"],
                       item["playtime_forever"], item["playtime_2weeks"]]
            user_items_ls.append(temp_ls)
    df_user_items = pd.DataFrame(user_items_ls, columns=user_items_ls_name)

    df_user_items["play_forever"] = df_user_items["playtime_forever"] > 0
    df_n_game_played = df_user_items.groupby("user_id")["play_forever"].sum(
    ).reset_index().rename(columns={"play_forever": "n_played"})
    df_user_items_merged = df_user_items.merge(df_n_game_played, on="user_id")[
        ["user_id", "item_count", "n_played"]].drop_duplicates()
    df_user_overall = df_user_items_merged.merge(df_review_user, on="user_id")
    df_user_overall["play_pct"] = (
        df_user_overall["n_played"] / df_user_overall["item_count"]) * 100
    df_user_overall["review_pct"] = (
        df_user_overall["n_review"] / df_user_overall["item_count"]) * 100
    df_user_overall["recommend_pct"] = (
        df_user_overall["n_recommend"] / df_user_overall["item_count"]) * 100
    df_user_overall = df_user_overall[(df_user_overall["play_pct"] <= 100)
                                      & (df_user_overall["review_pct"] <= 100)
                                      & (df_user_overall["recommend_pct"]
                                         <= 100)]
    return df_user_overall, df_user_items


def get_explicit_ls(df_review):
    df_review = df_review.drop(columns="review")
    df_review["recommend"] = df_review["recommend"].astype(str)
    # convert True False to binary rating
    df_review["recommend"] = df_review["recommend"].map(
        {"True": 1, "False": 0})
    df_review_explicit = df_review
    return df_review_explicit


def get_implicit_ls(df_user_items):
    df_review_implitic = df_user_items[[
        "user_id", "item_id", "playtime_forever"]]
    return df_review_implitic


def get_filtered_explicit_ls(data, min=5):

    item_ls_filter = data["item_id"].value_counts() >= min
    user_ls_filter = data["user_id"].value_counts() >= min
    item_ls_after_filter = item_ls_filter[item_ls_filter].index.tolist()
    user_ls_after_filter = user_ls_filter[user_ls_filter].index.tolist()
    data_after_filter = data[(data["item_id"].isin(item_ls_after_filter)) & (
        data["user_id"].isin(user_ls_after_filter))]
    data_after_filter = data_after_filter[['user_id', 'item_id', 'recommend']]
    return data_after_filter


def get_related_game_info(game_info, data_after_filter):
    df_game = pd.DataFrame(game_info)
    remain_col = ['id', 'app_name', 'publisher', 'developer', 'genres',
                  'price', 'early_access']
    df_game = df_game[remain_col]
    related_ls = data_after_filter.item_id.values.tolist()
    df_related_game_info = df_game.loc[df_game["id"].isin(related_ls)]
    return df_related_game_info


def main():
    game_info = load_data(GAME_FILE)
    au_review_info = load_data(AU_REVIEW_FILE)
    au_item_info = load_data(AU_ITEM_FILE)

    df_review = get_review_ls(au_review_info)
    df_review_user = review_by_user_summary(df_review)
    df_review_item = review_by_item_summary(df_review)

    # for dashboard
    df_user_overall, df_user_items = get_user_items_detail(
        au_item_info, df_review_user)
    df_review_item_overall = get_item_detail(df_review_item, game_info)

    # for recommender system
    df_review_explicit = get_explicit_ls(df_review)
    df_review_implitic = get_implicit_ls(df_user_items)

    data_after_filter = get_filtered_explicit_ls(df_review_explicit)
    df_related_game_info = get_related_game_info(game_info, data_after_filter)

    print("now save df_review_explicit")
    save_data(data_after_filter, PKL_FILE1)
    print("now save df_review_implitic")
    save_data(df_review_implitic, PKL_FILE2)
    print("now save df_related_game_info")
    save_data(df_related_game_info, PKL_FILE3)
    print("now save df_review")
    save_data(df_review, PKL_FILE4)
    print("now save df_review_item_overall and df_user_overall")
    save_data(df_review_item_overall, PKL_FILE5)
    save_data(df_user_overall, PKL_FILE6)
    print("Finish")


if __name__ == "__main__":
    main()
