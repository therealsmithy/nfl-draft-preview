import requests
import pandas as pd
from bs4 import BeautifulSoup

# Link to Tankathon's NFL Draft Order page
link = 'https://www.tankathon.com/nfl'

# Get the page and convert into BeautifulSoup object
order = requests.get(link)
order_soup = BeautifulSoup(order.content, 'html.parser')

# Scrape webpage for updated draft order
draft_order = order_soup.select(
    '.draft-board .pick-row .team-link-section > .desktop' 
)

# Turn into something I can work with
draft_range = range(0, len(draft_order))
draft_order = [draft_order[x].text for x in draft_range]
draft_order = pd.DataFrame(draft_order)
draft_order.index.name = 'Pick'
draft_order.columns = ['Team']
draft_order.index += 1

# Save final draft order
#draft_order.to_csv('data/draft_order.csv')