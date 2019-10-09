# Import libraries
from flask import Flask, jsonify, request, Response, make_response, render_template
from recsys.inference import load_output, rec_top_n_items, get_game_info
from recsys.dashboard_data_validate import get_data
import logging
import json
# import pre-create components for dashapp, easy to layout
from recsys.dashboard_components import *
import pandas as pd
from random import sample

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

# Example
# curl -X POST -H 'content-type: application/json' --data '{"uid":"76561198107703934","iid":"12210"}' http://127.0.0.1:8080/predict
# curl -X POST -H 'content-type: application/json' --data '{"rec_uid":"76561198107703934"}' http://127.0.0.1:8080/rec

server = Flask(__name__)
print(__name__)

output = load_output()
pred, algo = output["predictions"], output["algo"]


def precondition(data, accpetable_keys):
    if isinstance(data, dict):
        if all(keys in data.keys() for keys in accpetable_keys):
            return True
        else:
            return False
    return False


@server.route('/ping', methods=["GET"])
def ping():
    logging.info("checking health status of recsys api")
    content = {"status", "OK"}
    try:
        return Response(content, status=200,
                        mimetype="application/json"
                        )
    except Exception:
        logging.critical("RecSys API is not working")


@server.route('/predict', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        logging.warning("Response data type is not json")
        return make_response(
            jsonify({
                "errors": ["only supports JSON data"]}),
            400)  # bad request

    # Get the data from the POST request.

    try:
        encoding = request.headers.get('content-encoding', '')
        data = request.get_data()
        if encoding == "gzip":
            data = gzip.decompress(data)
        data = json.loads(data)

        if precondition(data, accpetable_keys=["uid", "iid"]):
            try:
                uid = data["uid"]
                iid = data["iid"]
                _, _, _, est, _ = algo.predict(uid, iid)
                result = {"est": est}
            except Exception as e:
                result = {"error": e.args}
        else:
            result = {"error": "Inference failed"}
    except Exception as e:
        result = {"error": e.args}
    return jsonify(result)


@server.route('/rec', methods=['POST'])
def rec():
    if request.content_type != 'application/json':
        logging.warning("Response data type is not json")
        return make_response(
            jsonify({
                "errors": ["only supports JSON data"]}),
            400)  # bad request

    # Get the data from the POST request.
    try:
        encoding = request.headers.get('content-encoding', '')
        data = request.get_data()
        if encoding == "gzip":
            data = gzip.decompress(data)
        data = json.loads(data)
        rec_ls = []
        if precondition(data, accpetable_keys=["rec_uid"]):
            try:
                rec_uid = data["rec_uid"]
                rec = rec_top_n_items(rec_uid, pred)
                result = {"rec": rec}
            except Exception as e:
                result = {"error": e.args}
        else:
            logging.warning("precondition not satisfied")
            result = {"error": "precondition not satisfied"}
    except Exception as e:
        logging.critical("Inference failed")
        result = {"error": e.args}
    return jsonify(result)


@server.route('/', methods=["GET", "POST"])
def index():

    output = load_output()
    pred, algo = output["predictions"], output["algo"]
    df_pred = pd.DataFrame(pred)
    sample_uid = [
        '76561198044716809',
        '76561198056863768',
        '76561198066312344',
        '76561197970982479',
        'pwnddumass']
    sample_iid = ['248820', '237930', '107200', '224500', '250320']

    if request.method == "POST":
        data = request.form
        uid = data["uid"]
        iid = data["iid"]
        rec_uid = data["recuid"]
        num = int(data["n"])
        _, _, _, est, _ = algo.predict(uid, iid)
        rec_ls = rec_top_n_items(rec_uid, pred, num)
        col = ['id', 'app_name', 'publisher', 'developer',
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




# Init Dash app


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, server=server,
                external_stylesheets=external_stylesheets)


# Set app layout
app.layout = html.Div([
    html.H1('Steam Games and Users Dashboard'),
    div_input,
    html.Div([
        html.H2('Game Distribution Graph'),
        scatter_plot,
        html.Br(),
        html.P('Number of rows selected: {}'.format(
            len(df.index)), id='dataset-rows-p'),
    ], style={'marginLeft': 25, 'marginRight': 25}),

    html.Br(),

    html.Div([
        html.H2("Distribution Plot and Statistics Table"),
        row([game_table, user_table]),
        html.Br(),
        game_hist,  # div user table
        user_hist,  # div user review
        user_pct_hist
    ]),  # div

    html.Br(),
    html.H3(dcc.Link(
        "Back to Homepage (Note: if not nevigate, please refresh the webpage)", href="/"))
], style={"marginLeft": 25, "marginRight": 25})  # end of overall div


@app.callback(
    dash.dependencies.Output('scatter-plot-graph', 'figure'),
    [
        dash.dependencies.Input('nb-review-input', 'value'),
        dash.dependencies.Input('nb-recommend-input', 'value'),
        dash.dependencies.Input('price-min-input', 'value'),
        dash.dependencies.Input('price-max-input', 'value'),
        dash.dependencies.Input('genre-dropdown', 'value'),
        dash.dependencies.Input('more-genre-input', 'value'),
        dash.dependencies.Input('publisher-input', 'value'),
        dash.dependencies.Input('developer-input', 'value'),
        dash.dependencies.Input('year-released-range-slider', 'value'),
        dash.dependencies.Input('x-axis-dropdown', 'value'),
        dash.dependencies.Input('y-axis-dropdown', 'value')
    ]
)
def update_scatter_plot(selected_nb_reviews, selected_nb_recommend,
                        input_min_price, input_max_price, selected_genre,
                        input_more_genre, input_publisher, input_developer,
                        selected_years_released, x_axis_var, y_axis_var):

    nb_reviews = selected_nb_reviews or df.n_review.min()
    year_released_start, year_released_end = selected_years_released or (
        df.release_year.min(), df.release_year.max())
    nb_recommend = selected_nb_recommend or df.n_recommend.min()
    price_min = input_min_price or 0
    price_max = input_max_price or df.price.max()
    game_genre = selected_genre or None
    more_genre = input_more_genre.strip() or None
    publisher = input_publisher.strip() or None
    developer = input_developer.strip() or None
    x_axis = x_axis_var or "price"
    y_axis = y_axis_var or "n_review"

    filtered_df = (
        df.pipe(lambda df: df[df['n_review'] >= nb_reviews])
        .pipe(lambda df: df[(df['release_year'] >= year_released_start) & (df['release_year'] <= year_released_end)])
        .pipe(lambda df: df[df['n_recommend'] >= nb_recommend])
        .pipe(lambda df: df[(df['price'] >= price_min) & (df['price'] <= price_max)])
        .pipe(lambda df: df[df['genres'].str.contains(game_genre, case=False, na=False)] if game_genre else df)
        .pipe(lambda df: df[df['genres'].str.contains(more_genre, case=False, na=False)] if more_genre else df)
        .pipe(lambda df: df[df['publisher'].str.contains(publisher, case=False, na=False)] if publisher else df)
        .pipe(lambda df: df[df['developer'].str.contains(developer, case=False, na=False)] if developer else df)
    )

    return {
        'data':

        [
            go.Scatter(
                x=filtered_df[filtered_df.early_access == True][x_axis],
                y=filtered_df[filtered_df.early_access == True][y_axis],
                text=filtered_df[filtered_df.early_access ==
                                 True]['app_name'],
                mode='markers',
                opacity=0.8,
                marker={
                    'color': 'orange',
                    'size': 10,
                    'line': {'width': 1, 'color': 'black'}
                },
                name="early_access"
            ),
            go.Scatter(
                x=filtered_df[filtered_df.early_access == False][x_axis],
                y=filtered_df[filtered_df.early_access == False][y_axis],
                text=filtered_df[filtered_df.early_access ==
                                 False]['app_name'],
                mode='markers',
                opacity=0.5,
                marker={
                    'color': 'blue',
                    'size': 10,
                    'line': {'width': 1, 'color': 'black'}
                },
                name="no early_access"
            )
        ],

        'layout': {"autosize": True,
                   "automargin": True,
                   "margin": dict(l=30, r=30, b=20, t=40),
                   "hovermode": "closest"}
    }


@app.callback(
    dash.dependencies.Output('dataset-rows-p', 'children'),
    [
        dash.dependencies.Input('nb-review-input', 'value'),
        dash.dependencies.Input('nb-recommend-input', 'value'),
        dash.dependencies.Input('price-min-input', 'value'),
        dash.dependencies.Input('price-max-input', 'value'),
        dash.dependencies.Input('genre-dropdown', 'value'),
        dash.dependencies.Input('more-genre-input', 'value'),
        dash.dependencies.Input('publisher-input', 'value'),
        dash.dependencies.Input('developer-input', 'value'),
        dash.dependencies.Input('year-released-range-slider', 'value'),
        dash.dependencies.Input('x-axis-dropdown', 'value'),
        dash.dependencies.Input('y-axis-dropdown', 'value')
    ]
)
def update_nb_rows_selected(selected_nb_reviews, selected_nb_recommend,
                            input_min_price, input_max_price, selected_genre, input_more_genre, input_publisher, input_developer,
                            selected_years_released, x_axis_var, y_axis_var):
    nb_reviews = selected_nb_reviews or df.n_review.min()
    year_released_start, year_released_end = selected_years_released or (
        df.release_year.min(), df.release_year.max())
    nb_recommend = selected_nb_recommend or df.n_recommend.min()
    price_min = input_min_price or 0
    price_max = input_max_price or df.price.max()
    game_genre = selected_genre or None
    more_genre = input_more_genre.strip() or None
    publisher = input_publisher.strip() or None
    developer = input_developer.strip() or None
    x_axis = x_axis_var or "price"
    y_axis = y_axis_var or "n_review"

    filtered_df = (
        df.pipe(lambda df: df[df['n_review'] >= nb_reviews])
        .pipe(lambda df: df[(df['release_year'] >= year_released_start) & (df['release_year'] <= year_released_end)])
        .pipe(lambda df: df[df['n_recommend'] >= nb_recommend])
        .pipe(lambda df: df[(df['price'] >= price_min) & (df['price'] <= price_max)])
        .pipe(lambda df: df[df['genres'].str.contains(game_genre, case=False, na=False)] if game_genre else df)
        .pipe(lambda df: df[df['genres'].str.contains(more_genre, case=False, na=False)] if more_genre else df)
        .pipe(lambda df: df[df['publisher'].str.contains(publisher, case=False, na=False)] if publisher else df)
        .pipe(lambda df: df[df['developer'].str.contains(developer, case=False, na=False)] if developer else df)
    )
    return 'Number of rows selected: {}'.format(len(filtered_df.index))


if __name__ == "__main__":
    app.run_server(host="0.0.0.0",debug=True, port=8080)
