import requests
import pandas as pd
import json

def get_data(begin, end):
    for i in range(begin, end+1):
        i_str = str(i)
        url = "https://api.stockgeist.ai/stock/us/hist/message-metrics"
        params = {
            "symbols": "MSFT",
            "start": f"{i_str}-01-01T00:00:00",
            "end": f"{i_str}-07-01T00:00:00",
            "timeframe": "1d"
        }

        # Define the headers
        headers = {
            "Accept": "application/json",
            "token": ""
        }
        r = requests.get(url, headers=headers, params=params, allow_redirects=True)
        open(f'sentiment_data/{i_str}_01_to_{i_str}_06.json', 'wb').write(r.content)
        
        i_str2 = str(i+1)
        url = "https://api.stockgeist.ai/stock/us/hist/message-metrics"
        params = {
            "symbols": "MSFT",
            "start": f"{i_str}-07-01T00:00:00",
            "end": f"{i_str2}-01-01T00:00:00",
            "timeframe": "1d"
        }

        # Define the headers
        headers = {
            "Accept": "application/json",
            "token": ""
        }
        r = requests.get(url, headers=headers, params=params, allow_redirects=True)
        open(f'sentiment_data/{i_str}_07_to_{i_str}_12.json', 'wb').write(r.content)

def calculate_sentiment_score(data):
    w_em = 2  # Weight for emotional messages
    w_nem = 1  # Weight for non-emotional messages
    
    # Destructure the data
    pos_em_count = data['pos_em_count']
    neg_em_count = data['neg_em_count']
    pos_nem_count = data['pos_nem_count']
    neg_nem_count = data['neg_nem_count']
    total_count = data['total_count']
    
    if total_count == 0:  # Avoid division by zero
        return 0
    
    # Calculate the sentiment score normalized between -1 and 1
    max_possible_score = w_em + w_nem
    score = (w_em * (pos_em_count - neg_em_count) + w_nem * (pos_nem_count - neg_nem_count)) / (total_count * max_possible_score)
    return score

def extract_text_and_compute_sentiment(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    results = []
    
    # Iterate through each entry in the 'data' key
    for entry in json_data['data']['MSFT']:
        sentiment_score = calculate_sentiment_score(entry)
        results.append((entry['timestamp'], sentiment_score))
    return results


sent_list = []
for i in range(2016, 2024):
    i_str = str(i)
    file_path = f'sentiment_data/{i_str}_01_to_{i_str}_06.json'
    sentiment_scores = extract_text_and_compute_sentiment(file_path)
    for timestamp, score in sentiment_scores:
        sent_list.append([timestamp, score])
    file_path = f'sentiment_data/{i_str}_07_to_{i_str}_12.json'
    sentiment_scores = extract_text_and_compute_sentiment(file_path)
    for timestamp, score in sentiment_scores:
        sent_list.append([timestamp, score])
file_path = f'sentiment_data/2024_01_to_2024_02.json'
sentiment_scores = extract_text_and_compute_sentiment(file_path)
for timestamp, score in sentiment_scores:
    sent_list.append([timestamp, score])

sentiments = pd.DataFrame(sent_list, columns=['timestamp', 'sentiment_score'])
sentiments.set_index('timestamp', inplace=True)
sentiments.to_csv('sentiments_2016_to_2024.csv')
