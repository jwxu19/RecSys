# Import libraries
from flask import Flask, jsonify, request
from recsys.inference import load_output, rec_top_n_items, get_game_info

app = Flask(__name__)

# curl -X POST -H 'content-type: application/json' --data '{"uid":"76561198107703934","iid":"12210", "rec_uid":"76561198067243010", "n":5}' http://127.0.0.1:8080/api

# Load the model
output = load_output()
pred, algo = output["predictions"], output["algo"]


@app.route('/api', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    est = {"est": None}
    rec_ls = {"rec_ls": None}
    rec_name = {"rec_title_ls": None}
    if "uid" in data.keys() and "iid" not in data.keys():
        return "missing argument iid"
    elif "uid" not in data.keys() and "iid" in data.keys():
        return "missing argument uid"
    elif "uid" in data.keys() and "iid" in data.keys():
        uid, iid = data["uid"], data["iid"]
        _, _, _, est["est"], _ = algo.predict(uid, iid)

    if "rec_uid" in data.keys() and "n" not in data.keys():
        return "missing argument n"
    elif "rec_uid" not in data.keys() and "n" in data.keys():
        return "missing argument rec_uid"
    elif "rec_uid" in data.keys() and "n" in data.keys():
        rec_uid, n = data["rec_uid"], data["n"]
        rec_ls["rec_ls"] = rec_top_n_items(rec_uid, pred, n)
        rec_name["rec_title_ls"] = get_game_info(
            rec_top_n_items(rec_uid, pred, n), "title")

    """
    uid, iid = data["uid"], data["iid"]
    rec_uid, n = data["rec_uid"], data["n"]
    _, _, _, est["est"], _ = algo.predict(uid, iid)
    rec_ls["rec_ls"] = rec_top_n_items(rec_uid, pred, n)
    """

    output = est, rec_ls, rec_name

    return jsonify(output)


if __name__ == '__main__':
    app.run(port=8080, debug=True, host='0.0.0.0')
