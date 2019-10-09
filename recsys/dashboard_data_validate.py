from os.path import abspath, dirname, join
import pandas as pd
import pickle


DATA_DIR = join(dirname(dirname(abspath((__file__)))), "data")
ITEM_FILE = join(DATA_DIR, "dashboard_item_summary.pkl")
USER_FILE = join(DATA_DIR, "dashboard_user_summary.pkl")


def load_pkl(file):
    f = open(file, "rb")
    output = pickle.load(f)
    f.close()
    return output


def validate_df(df):
    df = df.drop("id", axis=1)

    price_text_free = ["free", "play", "install", "third"]
    for i in price_text_free:
        df.loc[df["price"].str.contains(i, case=False, na=False), "price"] = 0

    col_fillna_unknow = ["app_name", "publisher",
                         "developer", "genres", "early_access"]
    for i in col_fillna_unknow:
        df[i] = df[i].fillna("unknown")

    # df["price"] = df["price"].fillna(-1)
    # df["release_date"] = df["release_date"].fillna("1900-01-01")

    df["n_review"] = df["n_review"].astype("int64")
    df["n_recommend"] = df["n_review"].astype("int64")
    df["genres"] = df["genres"].astype(str)
    df["price"] = df["price"].astype("float64", skipna=True)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="ignore")
    df["release_year"] = df["release_date"].dt.year

    return df


def get_data(file1=ITEM_FILE, file2=USER_FILE):
    df = load_pkl(file1)
    df2 = load_pkl(file2)
    df = validate_df(df)
    df_summary = df.describe().reset_index().round(2).drop("release_year", axis=1)
    df2_summary = df2.describe().reset_index().round(2)
    return df, df2, df_summary, df2_summary
