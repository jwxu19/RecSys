# Flask is a lightweight WSGI web application framework. It is designed to
# make getting started quick and easy, with the ability to scale up to complex
# applications. It began as a simple wrapper around Werkzeug and Jinja and has
# become one of the most popular Python web application frameworks.

# import flask library and its underlying objects
from flask import Flask, render_template, request
from recsys.inference import load_output, rec_top_n_items, get_game_info
import pandas as pd
from random import sample


app = Flask(__name__)

# __name__ might be the most mentioned dunder variables that is
# given a string value __main__ when file is executed


print(__name__)


# Connect the root directory to a home function call
# the app.route decorator defines a route on url, it takes
# in a url and when the client requests for this url, the app calls
# the corresponding function home() in this case
@app.route('/', methods=["GET", "POST"])
def index():

    output = load_output()
    pred, algo = output["predictions"], output["algo"]
    df_pred = pd.DataFrame(pred)
    sample_uid = sample(list(df_pred.uid.unique()), 15)
    sample_iid = sample(list(df_pred.iid.unique()), 15)

    if request.method == "POST":
        data = request.form
        uid = data["uid"]
        iid = data["iid"]
        rec_uid = data["recuid"]
        num = int(data["n"])
        _, _, _, est, _ = algo.predict(uid, iid)
        rec_ls = rec_top_n_items(rec_uid, pred, num)
        col = ['id', 'title', 'publisher', 'developer',
               'price']
        info_dict = {}
        for i in col:
            info_dict[i] = get_game_info(rec_ls, i)
        df_info = pd.DataFrame(info_dict)
        return render_template('./index.html', uid=uid, iid=iid,
                               rec_uid=rec_uid, n=num,
                               est=est, rec_ls=rec_ls,
                               tables=[df_info.to_html(classes="data")],
                               titles=df_info.columns.values,
                               sample_uid=sample_uid, sample_iid=sample_iid)
    return render_template('./index.html',
                           sample_uid=sample_uid, sample_iid=sample_iid)

# To render a html template you can use the render_template() method. All you
# have to do is provide the name of the template and the variables you want to
# pass to the template engine as keyword arguments.


@app.route("/Dashboard")
def get_name(name):
    return render_template(
        './Dashboard.html', name=name)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
