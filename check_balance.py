import jwt
import hashlib
import os
import requests
import uuid
from urllib.parse import urlencode, unquote
from argparse import ArgumentParser

def main(args):
    with open(os.path.join(args.config_path, 'upbit_access.txt'), 'r') as f:
        for line in f.readlines():
            access_key = line
            break
    with open(os.path.join(args.config_path, 'upbit_secret.txt'), 'r') as f:
        for line in f.readlines():
            secret_key = line
            break
    with open(os.path.join(args.config_path, 'upbit_server_url.txt'), 'r') as f:
        for line in f.readlines():
            server_url = line
            break

    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
    }

    jwt_token = jwt.encode(payload, secret_key)
    authorization = 'Bearer {}'.format(jwt_token)
    headers = {
      'Authorization': authorization,
    }
    params = {
    }

    res = requests.get(server_url + '/v1/accounts', params=params, headers=headers)
    print(res.json())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config')
    args = parser.parse_args()
    main(args)