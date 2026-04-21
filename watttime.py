
# To login and obtain an access token, use this code:

import requests
from requests.auth import HTTPBasicAuth
login_url = 'https://api.watttime.org/login'
rsp = requests.get(login_url, auth=HTTPBasicAuth('Himanshi', 'Swami@999'))
TOKEN = rsp.json()['token']
print(rsp.json())
