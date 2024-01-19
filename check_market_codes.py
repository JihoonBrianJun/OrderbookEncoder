import json
import requests
import pandas as pd
from argparse import ArgumentParser
from pycoingecko import CoinGeckoAPI

def main(args):
    url = "https://api.upbit.com/v1/market/all?isDetails=false"

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)

    with open(args.result_path, 'w') as f:
        f.write(response.text)
    with open(args.result_path, 'r') as f:
        results = json.load(f)

    krw_results = []
    for result in results:
       if 'KRW' in result['market']:
           krw_results.append(result) 
    krw_results_df = pd.DataFrame(krw_results)
    krw_results_df['short_name'] = krw_results_df['market'].apply(lambda x: x.split('-')[1].lower())
           
    cg = CoinGeckoAPI()
    markets = cg.get_coins_markets(vs_currency="KRW") 
    market_cap_df = pd.DataFrame(markets)[['symbol', 'market_cap']]
    
    krw_results_df = pd.merge(krw_results_df, market_cap_df, left_on=['short_name'], right_on=['symbol'], how='left')
    sorted_krw_results_df = krw_results_df.sort_values(by=['market_cap'], ascending=False).reset_index(drop=True)[:args.topn][['market', 'market_cap']]
    print(sorted_krw_results_df)
    top_krw_codes = sorted_krw_results_df.iloc[:args.topn]['market'].to_list()
    with open(args.top_path, "w") as f:
        json.dump(top_krw_codes, f)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_path', type=str, default='./result/market_codes.json')
    parser.add_argument('--top_path', type=str, default='./result/top_market_codes.json')
    parser.add_argument('--topn', type=int, default=10)
    args = parser.parse_args()
    main(args)