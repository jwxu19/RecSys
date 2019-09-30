import requests

# URL
url = 'http://localhost:8080/api'

# Change the value of experience that you want to test
r = requests.post(url, json={"uid": "A3R27T4HADWFFJ",
                             "iid": "0005019281",
                             "rec_uid": "A3R27T4HADWFFJ",
                             "n": 5})
print(r.json())
