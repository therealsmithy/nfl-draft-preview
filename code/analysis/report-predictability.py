import pandas as pd

# Read in data
first_rounders = pd.read_csv('data/2020_to_2024_first_rounders.csv')
report_20 = pd.read_csv('data/scoutingreports/jeremiah_2020.csv')
report_21 = pd.read_csv('data/scoutingreports/jeremiah_2021.csv')
report_22 = pd.read_csv('data/scoutingreports/jeremiah_2022.csv')
report_23 = pd.read_csv('data/scoutingreports/jeremiah_2023.csv')
report_24 = pd.read_csv('data/scoutingreports/jeremiah_2024.csv')
report_25 = pd.read_csv('data/scoutingreports/jeremiah_2025.csv')

# Combine reports
reports_past = pd.concat([report_20, report_21, report_22, report_23, report_24])

# Dummy first rounders
reports_past['first_rounder'] = reports_past['Name'].isin(first_rounders['Player']).astype(int)

# 