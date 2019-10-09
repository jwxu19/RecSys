import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from recsys.dashboard_data_validate import get_data

df, df2, df_summary, df2_summary = get_data()
most_frequent_genres = ["Indie", "Action",
                        "Adventure", "Casual", "Strategy", "RPG"]
variable_labels = {"n_recommend": "Number of Recommend (Vote Up or not)",
                   "n_review": "Number of Reviews",
                   "release_year": "Release Year",
                   "price": "Price (dollars)"}

div_review_input = html.Div(children=[
    html.Label('Minimum number of reviews on Game'),
    dcc.Input(id='nb-review-input',
              value=1,
              type='number',
              min=df["n_review"].min(),
              max=df["n_review"].max()),
    html.Br(), ],
    className="three columns")

div_rec_input = html.Div(children=[
    html.Label('Minimum number of recommend on Game'),
    dcc.Input(id='nb-recommend-input',
              value=df["n_recommend"].min(),
              type='number',
              min=df["n_recommend"].min(),
              max=df["n_recommend"].max()),
    html.Br(), ],
    className="three columns")

div_min_price_input = html.Div(children=[
    html.Label('Minimum Price (dollars)'),
    dcc.Input(id='price-min-input',
              type="number",
              min=df["price"].min(),
              max=df["price"].max(),
              value=0
              ),
    html.Br(), ],
    className="three columns")

div_max_price_input = html.Div(children=[
    html.Label('Maximum Price (dollars)'),
    dcc.Input(id='price-max-input',
              type="number",
              min=df["price"].min(),
              max=df["price"].max(),
              value=df["price"].max()
              ),
    html.Br(),
], className="three columns")

div_genre_input = html.Div(children=[
    html.Label('Most Frequent Genre (a game can have multiple genres)'),
    dcc.Dropdown(
        id='genre-dropdown',
        options=[
            {'label': genre.upper(), 'value': genre} for genre in most_frequent_genres
        ],
        multi=False
    ),
    html.Br(), ],
    className="three columns")
div_more_genre_input = html.Div(children=[
    html.Label('More Genre (e.g., Simulation, Sports)'),
    dcc.Input(id='more-genre-input',
              value='',
              type='text'),
    html.Br(), ],
    className="three columns")
div_publisher_input = html.Div(children=[
    html.Label('Publisher (e.g., Ubisoft, Square Enix)'),
    dcc.Input(id='publisher-input',
              value='',
              type='text'),
    html.Br(), ],
    className="three columns")
div_developer_input = html.Div(children=[
    html.Label('Developer (e.g. Valve, Paradox)'),
    dcc.Input(id='developer-input', value='', type='text'), ],
    className="three columns")

div_release_year_input = html.Div(children=[
    html.Label('Year released'),
    dcc.RangeSlider(
        id='year-released-range-slider',
        min=df["release_year"].min(),
        max=df["release_year"].max(),
        marks={str(y): str(y)
               for y in range(int(df["release_year"].min()),
                              int(df["release_year"].max()), 1)},
        value=[df["release_year"].min(
        ), df["release_year"].max()]
    ),
],)

div_x_var_input = html.Div(children=[
    html.Label('X-axis variable'),
    dcc.Dropdown(
        id='x-axis-dropdown',
        options=[
            {'label': label, 'value': value}
            for value, label in variable_labels.items()],
        multi=False,
        value='price'), ],
    className="six columns")

div_y_var_input = html.Div(children=[
    html.Label('Y-axis variable'),
    dcc.Dropdown(
        id='y-axis-dropdown',
        options=[
            {'label': label, 'value': value}
            for value, label in variable_labels.items()
        ],
        multi=False,
        value='n_review'), ],
    className="six columns")


def row(children_ls):
    return html.Div(children=children_ls, className="row")


# div input
div_input = html.Div([
    row([div_review_input, div_rec_input,
         div_min_price_input, div_max_price_input]),
    html.Br(),
    row([div_genre_input, div_more_genre_input,
         div_publisher_input, div_developer_input]),
    html.Br(),
    row(div_release_year_input),
    html.Br(),
    row([div_x_var_input, div_y_var_input])
],
    style={'marginLeft': 25, 'marginRight': 25}
)


scatter_plot = dcc.Graph(
    id='scatter-plot-graph',
    animate=True,
    figure={
        'data': [
            go.Scatter(
                x=df[df.early_access == "True"].price,
                y=df[df.early_access == "True"].n_review,
                text=df[df.early_access == "True"].app_name,
                mode='markers',
                opacity=0.7,
                marker={
                    'color': 'orange',
                    'size': 15,
                    'line': {'width': 1, 'color': 'black'}
                },
                name="early_access"
            ),
            go.Scatter(
                x=df[df.early_access == False].price,
                y=df[df.early_access == False].n_review,
                text=df[df.early_access == False].app_name,
                mode='markers',
                opacity=0.5,
                marker={
                    'color': 'blue',
                    'size': 15,
                    'line': {'width': 1, 'color': 'black'}
                },
                name="no early_access"
            )
        ],

        "layout": {"autosize": True,
                   "automargin": True,
                   "margin": dict(l=30, r=30, b=20, t=40),
                   "hovermode": "closest"}},
)

game_table = html.Div(
    [html.H3("Games Statistics Table"),
     dash_table.DataTable(
        id="item-table",
        columns=[{"name": i, "id": i}
                 for i in df_summary.columns],
        data=df_summary.to_dict('records'),
        style_cell={"textAlign": "center", "font_size": "15px"},
        style_header={
            "backgroundColor": 'rgb(230, 230, 230)',
            "fontWeight": "bold"
        },
        style_data_conditional=[{
            "if": {"row_index": 5},
            "backgroundColor": "#3D9970",
            'color': 'white'
        }]
    )], className="four columns offset-by-one")

game_hist = html.Div([
    html.H3("Game Review and Recommend Distribution"),
    dcc.Graph(id="game-histogram-plot-graph",
              animate=True,
              figure={"data": [go.Histogram(x=df.n_review, name="# review", opacity=0.6),
                               go.Histogram(x=df.n_recommend, name="# recommend", opacity=0.6)],
                      "layout": go.Layout(
                  barmode="overlay"
              ), }
              )
], )

user_table = html.Div([
    html.H3("User Statistics Table"),
    dash_table.DataTable(id="user-table",
                         columns=[{"name": i, "id": i}
                                  for i in df2_summary.columns],
                         data=df2_summary.to_dict('records'),
                         style_cell={"textAlign": "center",
                                     "font_size": "15px"},
                         style_header={
                             "backgroundColor": 'rgb(230, 230, 230)',
                             "fontWeight": "bold"
                         },
                         style_data_conditional=[{
                             "if": {"row_index": 5},
                             "backgroundColor": "#3D9970",
                             'color': 'white'}]
                         )
], className="five columns offset-by-one")

user_hist = html.Div([html.H3("User Behavior Chain: Play, Review, and Recommend Distribution"),
                      dcc.Graph(id='user-histogram-plot-graph',
                                animate=True,
                                figure={'data':
                                        [go.Histogram(x=df2.n_review, name="# review", opacity=0.6),
                                         go.Histogram(
                                            x=df2.n_recommend, name="# recommend", opacity=0.6),
                                         go.Histogram(x=df2.n_played, name="# play", opacity=0.6)],
                                        "layout":go.Layout(barmode="overlay")}
                                )
                      ])
user_pct_hist = html.Div([html.H3("User Behavior Chain: Play, Review, and Recommend Distribution (pct)"),
                          dcc.Graph(id='user-pct-histogram-plot-graph',
                                    animate=True,
                                    figure={'data':
                                            [go.Histogram(x=df2.review_pct, name="review%", opacity=0.6),
                                             go.Histogram(
                                                x=df2.recommend_pct, name="recommend%", opacity=0.6),
                                             go.Histogram(x=df2.play_pct, name="play%", opacity=0.6)],
                                            "layout":go.Layout(barmode="overlay")}
                                    )
                          ])
