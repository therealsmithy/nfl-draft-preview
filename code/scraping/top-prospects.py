import requests
import pandas as pd
from bs4 import BeautifulSoup

# Link to PFF's 2025 NFL Draft Big Board
link = 'https://www.pff.com/news/draft-2025-nfl-draft-board-big-board'

# Get the page and convert it to a BeautifulSoup object
prospects = requests.get(link)
prospects_soup = BeautifulSoup(prospects.text, 'html.parser')

# Scrape page for prospects
top_300 = prospects_soup.select(
    'strong'
)

# Turn into something I can work with
prospect_range = range(0, len(top_300))
top_300 = [top_300[x].text for x in prospect_range]
top_300 = pd.DataFrame(top_300)[8:308]
top_300.reset_index(drop=True, inplace=True)

# Use regular expressions to clean up the data
top_300['Rank'] = top_300.index + 1
top_300[0] = top_300[0].str.replace(r'^\d+\.\s*', '', regex = True)
top_300['Position'] = top_300[0].str.extract(r'^([A-Z/]+)')
top_300[0] = top_300[0].str.replace(r'^([A-Z/]+)', '', regex = True)
top_300['Name'] = top_300[0].str.extract(r'^(.*?),')
top_300[0] = top_300[0].str.replace(r'^(.*?),', '', regex = True)
top_300.rename(columns = {0: 'School'}, inplace = True)
top_300 = top_300[['Rank', 'Name', 'Position', 'School']]
top_300.set_index('Rank', inplace = True)

# Save to CSV
#top_300.to_csv('data/pff_2025_top300.csv')