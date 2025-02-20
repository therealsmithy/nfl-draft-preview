import requests
import pandas as pd
from bs4 import BeautifulSoup

# Link to SMART stopwords
url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a11-smart-stop-list/english.stop'

# Get the page and convert it to a BeautifulSoup object
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Convert soup to a list
stopwords = soup.get_text().split()

# Convert to dataframe and save to CSV
stopwords_df = pd.DataFrame(stopwords, columns=['stopword'])
stopwords_df.to_csv('data/stopwords.csv', index=False)