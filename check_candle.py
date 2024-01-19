import json
import requests
from argparse import ArgumentParser

def main(args):
    with open(args.market_codes_path, 'r') as f:
        market_codes = json.load(f)
    
    for market_code in market_codes:
        url = f"https://api.upbit.com/v1/candles/minutes/1?market={market_code}&count={args.count}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        
        with open(args.result_path, 'w') as f:
            f.write(response.text)
        with open(args.result_path, 'r') as f:
            result = json.load(f)

        print(result[0])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_path', type=str, default='./result/candle_result.json')
    parser.add_argument('--market_codes_path', type=str, default='./result/top_market_codes.json')
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    main(args)