import requests

# URL
url = 'http://localhost:8080/api'

# Change the value of experience that you want to test
r = requests.post(url, json={"uid": "76561198107703934",
                             "iid": "12210",
                             "rec_uid": "76561198067243010",
                             "n": 5})
print(r.json())
