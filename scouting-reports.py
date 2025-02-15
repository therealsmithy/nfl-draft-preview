# Process
#1. Scrape scouting report 
#2. Scrape old ones if possible
#3. TFIDVectorizer onto the data
#4. Train XGBoost model
#5. SHAP Models
#6. https://dalex.drwhy.ai/python/
#7. https://medium.com/responsibleml/basic-xai-with-dalex-part-6-lime-method-f6aab0af058a 

from bs4 import BeautifulSoup
import requests
import pandas as pd
import re

# Daniel Jeremiah Top 50 2020
url = 'https://www.nfl.com/news/daniel-jeremiah-s-top-50-2020-nfl-draft-prospect-rankings-2-0-0ap3000001102767'
# Scrape and process
response = requests.get(url)
soup_20 = BeautifulSoup(response.text, 'html.parser')
top_50_2020 = soup_20.select('.nfl-c-article__container')
article_range = range(0, len(top_50_2020))
top_50_2020 = [top_50_2020[x].text for x in article_range]
# Convert to dataframe
df_2020 = pd.DataFrame(columns = ['Name', 'Position', 'School', 'Report'])
# Pull data into dataframe
for i in range(2, 52):
    name = re.findall(r'\d+\)\s([A-Za-z\s\.\',\-]+),', top_50_2020[i])[0]
    position = re.findall(r'(?<=, ).*', top_50_2020[i])[0]
    school = re.findall(r'(?<=: ).* (?=[|])', top_50_2020[i])[0]
    report = max(re.findall(r'(?<=[\n\n]).*', top_50_2020[i]), key=len)
    df_2020.loc[len(df_2020)] = {'Name': name, 'Position': position, 'School': school, 'Report': report}

# Scrape Daniel Jeremiah Top 50 2021
url = 'https://www.nfl.com/news/daniel-jeremiah-s-top-50-2021-nfl-draft-prospect-rankings-3-0'
# Scrape and process
soup_21 = BeautifulSoup(requests.get(url).text, 'html.parser')
top_50_2021_info = soup_21.select('.nfl-o-ranked-item__content')
top_50_2021_report = soup_21.select('.nfl-c-body-part--text')
article_range = range(0, len(top_50_2021))
report_range = range(0, len(top_50_2021_report))
top_50_2021_info = [top_50_2021[x].text for x in article_range]
top_50_2021_report = [top_50_2021_report[x].text for x in report_range]
# Convert to dataframe
df_2021 = pd.DataFrame(columns = ['Name', 'Position', 'School', 'Report'])
# Pull info into dataframe
for i in range(0, 50):
    name = re.findall(r'([A-Za-z\s\.\',\-]+)\n\n', top_50_2021_info[i])[0]
    position = re.findall(r'·\s*([^·]+)\s* ·', top_50_2021_info[i])[0]
    school = re.findall(r'\n\s*([^\n\r·]+)\s*·', top_50_2021_info[i])[0]
    df_2021.loc[len(df_2021)] = {'Name': name, 'Position': position, 'School': school, 'Report': top_50_2021_report[i + 1]}
# Clean dataframe
df_2021['Report'] = df_2021['Report'].str.strip()
df_2021['Name'] = df_2021['Name'].str.strip()

# Scrape Daniel Jeremiah Top 50 2022
url = 'https://www.nfl.com/news/daniel-jeremiah-s-top-50-2022-nfl-draft-prospect-rankings-3-0'
# Scrape and process
soup_22 = BeautifulSoup(requests.get(url).text, 'html.parser')
top_50_2022_info = soup_22.select('.nfl-o-ranked-item__content')
top_50_2022_report = soup_22.select('.nfl-c-body-part--text')
article_range = range(0, len(top_50_2022_info))
report_range = range(0, len(top_50_2022_report))
top_50_2022_info = [top_50_2022_info[x].text for x in article_range]
top_50_2022_report = [top_50_2022_report[x].text for x in report_range]
# Convert to dataframe
df_2022 = pd.DataFrame(columns = ['Name', 'Position', 'School', 'Report'])
# Pull info into dataframe
for i in range(0, 50):
    name = re.findall(r'([A-Za-z\s\.\',\-]+)\n\n', top_50_2022_info[i])[0]
    position = re.findall(r'·\s*([^·]+)\s* ·', top_50_2022_info[i])[0]
    school = re.findall(r'\n\s*([^\n\r·]+)\s*·', top_50_2022_info[i])[0]
    df_2022.loc[len(df_2022)] = {'Name': name, 'Position': position, 'School': school, 'Report': top_50_2022_report[i + 1]}
# Clean dataframe
df_2022['Report'] = df_2022['Report'].str.strip()
df_2022['Name'] = df_2022['Name'].str.strip()

# Scrape Daniel Jeremiah Top 50 2023
url = 'https://www.nfl.com/news/daniel-jeremiah-s-top-50-2023-nfl-draft-prospect-rankings-4-0'
# Scrape and process
soup_23 = BeautifulSoup(requests.get(url).text, 'html.parser')
top_50_2023_info = soup_23.select('.nfl-o-ranked-item__content')
top_50_2023_report = soup_23.select('.nfl-c-body-part--text')
article_range = range(0, len(top_50_2023_info))
report_range = range(0, len(top_50_2023_report))
top_50_2023_info = [top_50_2023_info[x].text for x in article_range]
top_50_2023_report = [top_50_2023_report[x].text for x in report_range]
# Convert to dataframe
df_2023 = pd.DataFrame(columns = ['Name', 'Position', 'School', 'Report'])
# Pull info into dataframe
for i in range(0, 50):
    name = re.findall(r'([A-Za-z\s\.\',\-]+)\n\n', top_50_2023_info[i])[0]
    position = re.findall(r'·\s*([^·]+)\s* ·', top_50_2023_info[i])[0]
    school = re.findall(r'\n\s*([^\n\r·]+)\s*·', top_50_2023_info[i])[0]
    df_2023.loc[len(df_2023)] = {'Name': name, 'Position': position, 'School': school, 'Report': top_50_2023_report[i + 1]}
# Clean dataframe
df_2023['Report'] = df_2023['Report'].str.strip()
df_2023['Name'] = df_2023['Name'].str.strip()

# Scrape Daniel Jeremiah Top 150 2024
url = 'https://www.nfl.com/news/daniel-jeremiah-s-top-150-prospects-in-the-2024-nfl-draft-class'
# Scrape and process
soup_24 = BeautifulSoup(requests.get(url).text, 'html.parser')
top_150_2024_info = soup_24.select('.nfl-o-ranked-item__content')
top_150_2024_report = soup_24.select('.nfl-c-body-part--text')
article_range = range(0, len(top_150_2024_info))
report_range = range(0, len(top_150_2024_report))
top_150_2024_info = [top_150_2024_info[x].text for x in article_range]
top_150_2024_report = [top_150_2024_report[x].text for x in report_range]
# Convert to dataframe
df_2024 = pd.DataFrame(columns = ['Name', 'Position', 'School', 'Report'])
# Pull info into dataframe
for i in range(0, 50):
    name = re.findall(r'([A-Za-z\s\.\',\-]+)\n\n', top_150_2024_info[i])[0]
    position = re.findall(r'·\s*([^·]+)\s* ·', top_150_2024_info[i])[0]
    school = re.findall(r'\n\s*([^\n\r·]+)\s*·', top_150_2024_info[i])[0]
    df_2024.loc[len(df_2024)] = {'Name': name, 'Position': position, 'School': school, 'Report': top_150_2024_report[i + 3]}
# Clean dataframe
df_2024['Report'] = df_2024['Report'].str.strip()
df_2024['Name'] = df_2024['Name'].str.strip()

# Scrape Daniel Jeremiah Top 50 2025
url = 'https://www.nfl.com/news/daniel-jeremiah-s-top-50-2025-nfl-draft-prospect-rankings-1-0'
# Scrape and process
soup_25 = BeautifulSoup(requests.get(url).text, 'html.parser')
top_50_2025_info = soup_25.select('.nfl-o-ranked-item__content')
top_50_2025_report = soup_25.select('.nfl-c-body-part--text')
article_range = range(0, len(top_50_2025_info))
report_range = range(0, len(top_50_2025_report))
top_50_2025_info = [top_50_2025_info[x].text for x in article_range]
top_50_2025_report = [top_50_2025_report[x].text for x in report_range]
# Convert to dataframe
df_2025 = pd.DataFrame(columns = ['Name', 'Position', 'School', 'Report'])
# Pull info into dataframe
for i in range(0, 50):
    name = re.findall(r'([A-Za-z\s\.\',\-]+)\n\n', top_50_2025_info[i])[0]
    position = re.findall(r'·\s*([^·]+)\s* ·', top_50_2025_info[i])[0]
    school = re.findall(r'\n\s*([^\n\r·]+)\s*·', top_50_2025_info[i])[0]
    df_2025.loc[len(df_2025)] = {'Name': name, 'Position': position, 'School': school, 'Report': top_50_2025_report[i + 3]}
# Clean dataframe
df_2025['Report'] = df_2025['Report'].str.strip()
df_2025['Name'] = df_2025['Name'].str.strip()

# Save dataframes
#df_2020.to_csv('data/scoutingreports/jeremiah_2020.csv', index=False)
#df_2021.to_csv('data/scoutingreports/jeremiah_2021.csv', index=False)
#df_2022.to_csv('data/scoutingreports/jeremiah_2022.csv', index=False)
#df_2023.to_csv('data/scoutingreports/jeremiah_2023.csv', index=False)
#df_2024.to_csv('data/scoutingreports/jeremiah_2024.csv', index=False)
#df_2025.to_csv('data/scoutingreports/jeremiah_2025.csv', index=False)