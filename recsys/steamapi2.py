import requests
import pandas as pd
from os.path import abspath, dirname, join


DATA_DIR = join(dirname(dirname(abspath((__file__)))),
                "data")
APP_INFO_FILE = join(DATA_DIR, "app_info_details.csv")


def get_applist():
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2"
    r = requests.get(url)
    data = r.json()
    applist = data["applist"]["apps"]
    app_dict = {app["appid"]: [app["name"]] for app in applist}
    return app_dict


def get_appinfo():
    app_dict = get_applist()
    app_ls = list(app_dict.keys())
    app_info_detail = []
    counter = 0
    total = len(app_ls)
    info_ls = ['type', 'name', 'steam_appid', 'required_age', 'is_free',
               'detailed_description',
               'fullgame', 'header_image',
               'website', 'developers',
               'publishers', 'platforms',
               'categories', 'genres',
               'release_date']

    for appid in app_ls:
        counter += 1
        left = total - counter
        print(f"This is #{counter}, remaining app #{left}")
        print(f"start app {appid}")
        url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
        r = requests.get(url)
        data = r.json()
        if data is not None:
            data = list(data.values())[0]
            if "data" in data.keys():
                data = data["data"]
                remove_ls = []
                for i in info_ls:
                    if i not in data.keys():
                        remove_ls.append(i)

                info_ls_temp = info_ls.copy()
                for i in remove_ls:
                    info_ls_temp.remove(i)
                info = {i: data[i] for i in info_ls_temp}
                app_info_detail.append(info)
                print(f"finish app {appid}")
    return app_info_detail


def to_df():
    app_info_detail = get_appinfo()
    df_app_info = pd.DataFrame(app_info_detail)
    return df_app_info


def save_csv(df, file):
    df.to_csv(file, sep="\t")


def main():
    df_app_info = to_df()
    save_csv(df_app_info, APP_INFO_FILE)


if __name__ == "__main__":
    main()
