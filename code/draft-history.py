import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import numpy as np
import re

# Scrape first rounders from 2020-2024
first_rounders = pd.DataFrame()
for i in range(2020, 2025):
    draft = pd.DataFrame()
    url = f'https://en.wikipedia.org/wiki/{i}_NFL_draft'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    draft = soup.find_all('table', {'class': 'wikitable'})[1]
    rows = []
    for tr in draft.find('tbody').find_all(['tr', 'th']):
        cells = tr.find_all(['td', 'th'])
        if len(cells) > 0:
            rows.append([cell.text.strip() for cell in cells])
    draft = pd.DataFrame(rows)
    draft.columns = draft.iloc[0]
    first_round = draft.iloc[1:33]
    first_rounders = pd.concat([first_rounders, first_round])
    time.sleep(np.random.uniform(1, 10, 1)[0])

# Only want players
first_rounders_list = first_rounders['Player'].to_list()
first_rounders_list = [re.sub(r'\W+$', '', x) for x in first_rounders_list]

# Save to csv
first_rounders.to_csv('data/2020_to_2024_first_rounders.csv', index=False)