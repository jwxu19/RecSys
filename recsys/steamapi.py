import requests
import json
import pandas as pd
from os.path import abspath, dirname, join


DATA_DIR = join(dirname(dirname(abspath((__file__)))),
                "data")
APP_GENERAL_FILE = join(DATA_DIR, "app_review_general_part1.csv")
APP_DETAIL_FILE = join(DATA_DIR, "app_review_details_part1.csv")


def get_applist():
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2"
    r = requests.get(url)
    data = r.json()
    applist = data["applist"]["apps"]
    app_dict = {app["appid"]: [app["name"]] for app in applist}
    return app_dict


def get_appreview():
    app_dict = get_applist()
    app_ls = list(app_dict.keys())[:500]
    app_review_detail = []
    counter = 0
    total = len(app_ls)
    for appid in app_ls:
        counter += 1
        left = total - counter
        print(f"This is #{counter}, remaining app #{left}")
        print(f"start app {appid}")
        url = f"https://store.steampowered.com/appreviews/{appid}?json=1"
        r = requests.get(url)
        data = r.json()
        if "query_summary" and "reviews" in data.keys():
            if len(data["query_summary"]) != 0 and len(data["reviews"]) != 0:
                #'num_reviews', 'review_score', 'review_score_desc',
                #'total_positive', 'total_negative', 'total_reviews'
                app_dict[appid].extend(data["query_summary"].values())
                reviews = data["reviews"]
                for i in reviews:
                    author = i["author"]
                    author["appid"] = appid
                    author["voted_up"] = i["voted_up"]  # most important info
                    author["votes_up"] = i["votes_up"]
                    author["comment_count"] = i["comment_count"]
                    author["weighted_vote_score"] = i["weighted_vote_score"]
                    author["review"] = i["review"]
                    author["steam_purchase"] = i["steam_purchase"]
                    author["received_for_free"] = i["received_for_free"]
                    author["written_during_early_access"] = i["written_during_early_access"]
                    app_review_detail.append(author)
        print(f"finish app {appid}")
    return app_dict, app_review_detail


def get_dict_param_name():
    dict_param_name = ["name", 'num_reviews', 'review_score', 'review_score_desc',
                       'total_positive', 'total_negative', 'total_reviews']
    return dict_param_name


def to_df():
    app_dict, app_review_detail = get_appreview()
    dict_param_name = get_dict_param_name()
    df_app_review_detail = pd.DataFrame(app_review_detail)
    df_app_dict = pd.DataFrame(app_dict, index=dict_param_name).T
    return df_app_review_detail, df_app_dict


def save_csv(df, file):
    df.to_csv(file, sep="\t")


def main():
    df_app_review_detail, df_app_dict = to_df()
    save_csv(df_app_review_detail, APP_DETAIL_FILE)
    save_csv(df_app_dict, APP_GENERAL_FILE)


if __name__ == "__main__":
    main()
