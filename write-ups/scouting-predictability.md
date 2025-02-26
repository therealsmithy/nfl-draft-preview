# Scouting Report Predictability


## Introduction

Professional sports “insiders” are paid a lot of money for their
insights and analysis on the game they claim to know. But how much of
what they say actually holds up in practice? Daniel Jeremiah is a former
NFL scot and current NFL Network analyst who posts his ideas on the top
50 prospects in the NFL draft every year. This project will analyze the
relationship between his scouting reports and the actual draft results
to see how well his words hold up.

<p align="center">

<img src="https://github.com/therealsmithy/nfl-draft-preview/blob/main/vizzes/jeremiah-tweet.png"/>
</p>

## Primary Question

The main question this project will answer is this: Can a single
person’s work accurately predict the likelihood of a player going in the
first round of the NFL draft? Daniel Jeremiah is the subject of this
project due to his status as an NFL Network analyst and the availability
of his past scouting reports.

## Scraping and Cleaning

In order to perform analysis on Jeremiah’s statements, they first need
to be scraped from the NFL website. For this project his Top 50
prospects from 2020-2025 were scraped and cleaned, then saved into a csv
file.

``` python
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
```

In order to properly perform text analysis on these reports, the text
needs to be properly cleaned and tokenized. The stopwords used to clean
the data were scraped from the SMART Stopwords.

``` python
# Link to SMART stopwords
url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a11-smart-stop-list/english.stop'

# Get the page and convert it to a BeautifulSoup object
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Convert soup to a list
stopwords = soup.get_text().split()

# Convert to dataframe and save to CSV
stopwords_df = pd.DataFrame(stopwords, columns=['stopword'])
#stopwords_df.to_csv('data/stopwords.csv', index=False)
```

## Methods

Now that the data has been scraped and the stopwords are ready, the next
step is to perform text analysis on the reports. The first step is to
clean the text and remove stopwords. The next step is to tokenize the
text and count the frequency of each word.

``` python
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
import matplotlib.pyplot as plt
import dalex as dx
import shap
import numpy as np
import pandas as pd

# Read in data
first_rounders = pd.read_csv('G:/My Drive/Py Projects/nfl-draft-preview/data/2020_to_2024_first_rounders.csv')
report_20 = pd.read_csv('G:/My Drive/Py Projects/nfl-draft-preview/data/scoutingreports/jeremiah_2020.csv')
report_21 = pd.read_csv('G:/My Drive/Py Projects/nfl-draft-preview/data/scoutingreports/jeremiah_2021.csv')
report_22 = pd.read_csv('G:/My Drive/Py Projects/nfl-draft-preview/data/scoutingreports/jeremiah_2022.csv')
report_23 = pd.read_csv('G:/My Drive/Py Projects/nfl-draft-preview/data/scoutingreports/jeremiah_2023.csv')
report_24 = pd.read_csv('G:/My Drive/Py Projects/nfl-draft-preview/data/scoutingreports/jeremiah_2024.csv')
report_25 = pd.read_csv('G:/My Drive/Py Projects/nfl-draft-preview/data/scoutingreports/jeremiah_2025.csv')
stopwords = pd.read_csv('G:/My Drive/Py Projects/nfl-draft-preview/data/stopwords.csv')

# Combine reports
reports_past = pd.concat([report_20, report_21, report_22, report_23, report_24])

# Dummy first rounders
reports_past['first_rounder'] = reports_past['Name'].isin(first_rounders['Player']).astype(int)

# Lemmatize first
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(word) for word in w_tokenizer.tokenize(text)]
reports_past['report_lemmatized'] = reports_past['Report'].apply(lemmatize_text)

# Remove stopwords
reports_past['report_lemmatized'] = reports_past['report_lemmatized'].apply(lambda x: [item for item in x if item not in stopwords['stopword'].values])

# Define dataframe
xgb_df = reports_past[['report_lemmatized', 'first_rounder']].copy()
```

After this I convert the text into a format that can be used in a
machine learning model. The TfidfVectorizer will be used to convert the
text into a matrix of TF-IDF features.

``` python
# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(xgb_df['report_lemmatized'].apply(lambda x: ' '.join(x)))

# Make sure output is interpretable
feature_names = vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
```

Once this is done, the data is ready to be set up for model training.
Once this is done we can train the model and see how well it performs
initially.

``` python
# INITIAL XGBoost classifier
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=102701)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df_tfidf, xgb_df['first_rounder'], test_size=0.2, random_state=102701)

# Fit to training data
xgb_clf.fit(X_train, y_train)

# Create basic predictions
y_pred = xgb_clf.predict(X_test)
initial_accuracy = accuracy_score(y_test, y_pred)

# Check confusion matrix  to see where the model struggles
conf_matrix = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(conf_matrix).plot()
plt.show()
```

![](scouting-predictability_files/figure-commonmark/cell-6-output-1.png)

The initial accuracy of the model is 48.0%. This is a good starting
point, but the model can be further tuned to improve accuracy.

After this, the model can be tuned to find the best hyperparameters.

``` python
# Split validation set from training set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=102701)

# Fit model to training data for tuning
eval_metrics = ['auc', 'error']
model_1 = xgb.XGBClassifier(learning_rate = 0.1, n_estimators = 500, 
                            random_state = 102701, early_stopping_rounds = 10,
                            eval_metric=eval_metrics, objective='binary:logistic')
eval_set =[(X_train, y_train), (X_val, y_val)]
model_1.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Extract evaluation results
eval_results = model_1.evals_result()

# Graph error vs. number of iterations
plt.figure(figsize = (12, 8))
epochs = len(eval_results['validation_1']['error'])
x_axis = range(0, epochs)
plt.plot(x_axis, eval_results['validation_0']['error'], label = f'Learning Rate: {0.01}')
plt.legend()
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.title('XGBoost Error vs. Iterations')
plt.show()

# Set up tuning grid
param_grid = {  
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 10],
    'learning_rate': [0.1], 
    'n_estimators': [100], 
    'subsample': [1.0],
    'colsample_bytree': [1.0], 
    'reg_alpha': [1e-5],
    'early_stopping_rounds':[10],
    'scale_pos_weight': [0.5626],
}

# Set up grid search
grid_search = GridSearchCV(
    estimator = xgb_clf,
    param_grid = param_grid,
    scoring = 'accuracy',
    n_jobs = 1,
    cv = 5,
    verbose = 1
)

# Fit grid search
grid_search.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Extract best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f'Best parameters: {best_params}')
print(f'Best cross-validation score: {best_score}')

# Tune subsample and colsample_bytree
param_grid = {
    'max_depth': [10],
    'min_child_weight': [1],
    'learning_rate': [0.1], 
    'n_estimators': [100], 
    'subsample': [0.5, 0.75, 1.0],
    'colsample_bytree': [0.5, 0.75, 1.0], 
    'reg_alpha': [1e-5],
    'early_stopping_rounds':[10],
    'scale_pos_weight': [0.5625],
}

# Set up grid search
grid_search = GridSearchCV(
    estimator = xgb_clf,
    param_grid = param_grid,
    scoring = 'accuracy',
    n_jobs = 1,
    cv = 5,
    verbose = 1
)

# Fit grid search
grid_search.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Extract best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f'Best parameters: {best_params}')
print(f'Best cross-validation score: {best_score}')

# Tune reg_alpha
param_grid = {
    'max_depth': [10],
    'min_child_weight': [1],
    'learning_rate': [0.1], 
    'n_estimators': [100], 
    'subsample': [1.0],
    'colsample_bytree': [0.75], 
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    'early_stopping_rounds':[10],
    'scale_pos_weight': [0.5625],
}

# Set up grid search
grid_search = GridSearchCV(
    estimator = xgb_clf,
    param_grid = param_grid,
    scoring = 'accuracy',
    n_jobs = 1,
    cv = 5,
    verbose = 1
)

# Fit grid search
grid_search.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Extract best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f'Best parameters: {best_params}')
print(f'Best cross-validation score: {best_score}')

# Tune learning rates
learning_rates = [0.0005, 0.001, 0.05, 0.01, 0.03, 0.05, 0.1]
eval_results = {}
for lr in learning_rates:
    model = xgb.XGBClassifier(learning_rate = lr, n_estimators = 500,
                            random_state = 102701, early_stopping_rounds = 100,
                            eval_metric=eval_metrics, objective='binary:logistic')
    eval_set =[(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    eval_results[f'lr_{lr}'] = model.evals_result()

# Plot learning rate results
plt.figure(figsize = (12, 8))
for lr, result in eval_results.items():
    epochs = len(result['validation_1']['error'])
    x_axis = range(0, epochs)
    plt.plot(x_axis, result['validation_1']['error'], label = f'Learning Rate: {lr}')
plt.legend()
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.title('XGBoost Error vs. Iterations for Varying Learning Rates')
plt.show()    

# Final tuned model
xgb_final = xgb.XGBClassifier(max_depth = 10, min_child_weight = 1, learning_rate = 0.01, 
                              n_estimators = 100, subsample = 1.0, colsample_bytree = 0.75, 
                              reg_alpha = 0.1, early_stopping_rounds = 10, 
                              scale_pos_weight = 0.5625, objective='binary:logistic', 
                              random_state=102701)

# Fit model
xgb_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
# Predict on test set
y_pred = xgb_final.predict(X_test)
```

``` python
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
```

The accuracy for the tuned model is 54.0%.

The now tuned model can be used to predict the likelihood of a player
going in the first round based on their scouting report for the 2025
class.

``` python
# Apply to 2025 prospects
report_25['report_lemmatized'] = report_25['Report'].apply(lemmatize_text)
report_25['report_lemmatized'] = report_25['report_lemmatized'].apply(lambda x: [item for item in x if item not in stopwords['stopword'].values])

# Convert text to TF-IDF features
tfidf_matrix_25 = vectorizer.transform(report_25['report_lemmatized'].apply(lambda x: ' '.join(x)))

# Make sure output is interpretable
df_tfidf_25 = pd.DataFrame(tfidf_matrix_25.toarray(), columns=vectorizer.get_feature_names_out())

# Predict probabilities on 2025 prospects
xgb_preds_proba = xgb_final.predict_proba(df_tfidf_25)[:, 1]

# Add predictions to the dataframe
report_25['first_rounder_proba'] = xgb_preds_proba

# Rank prospects by likelihood of being a first rounder
report_25_ranked = report_25.sort_values(by='first_rounder_proba', ascending=False)
```

## Results

Now that the heavy lifting is done from building the model, the results
are ready to be interpreted. The idea here is that this model can be
used to predict the likelihood of a player going in the first round
based on their scouting report. Perhaps unsurprisingly, the results may
not look exactly how we expected them to.

``` python
# Display the ranked prospects
ranked = pd.read_csv('G:/My Drive/Py Projects/nfl-draft-preview/data/scoutingreports/jeremiah_2025_ranked.csv')
print(ranked.to_markdown())
```

    |    | Name               |   first_rounder_proba |
    |---:|:-------------------|----------------------:|
    |  0 | T.J. Sanders       |              0.500306 |
    |  1 | Travis Hunter      |              0.487165 |
    |  2 | Jihaad Campbell    |              0.480641 |
    |  3 | Kenneth Grant      |              0.473475 |
    |  4 | Armand Membou      |              0.4574   |
    |  5 | Benjamin Morrison  |              0.454257 |
    |  6 | Josh Simmons       |              0.447684 |
    |  7 | Carson Schwesinger |              0.447075 |
    |  8 | Will Campbell      |              0.440519 |
    |  9 | Matthew Golden     |              0.437669 |
    | 10 | Shedeur Sanders    |              0.432644 |
    | 11 | James Pearce Jr.   |              0.425554 |
    | 12 | Maxwell Hairston   |              0.425449 |
    | 13 | TreVeyon Henderson |              0.416668 |
    | 14 | Derrick Harmon     |              0.414875 |
    | 15 | Kelvin Banks Jr.   |              0.413913 |
    | 16 | Marcus Mbow        |              0.413502 |
    | 17 | Jalon Walker       |              0.401993 |
    | 18 | Ashton Jeanty      |              0.400626 |
    | 19 | Luther Burden III  |              0.39296  |
    | 20 | Tyler Booker       |              0.387092 |
    | 21 | Josaiah Stewart    |              0.381088 |
    | 22 | Emeka Egbuka       |              0.379159 |
    | 23 | Nick Emmanwori     |              0.377785 |
    | 24 | Will Johnson       |              0.376144 |
    | 25 | Donovan Ezeiruaku  |              0.376083 |
    | 26 | Mike Green         |              0.373362 |
    | 27 | Colston Loveland   |              0.368582 |
    | 28 | Jack Sawyer        |              0.36807  |
    | 29 | Cam Ward           |              0.364048 |
    | 30 | Joshua Farmer      |              0.362807 |
    | 31 | Omarion Hampton    |              0.357396 |
    | 32 | Tetairoa McMillan  |              0.357001 |
    | 33 | Grey Zabel         |              0.356309 |
    | 34 | Azareye'h Thomas   |              0.354896 |
    | 35 | Shemar Stewart     |              0.353331 |
    | 36 | Mykel Williams     |              0.353165 |
    | 37 | Elijah Arroyo      |              0.3512   |
    | 38 | Mason Graham       |              0.350768 |
    | 39 | Josh Conerly Jr.   |              0.350471 |
    | 40 | Donovan Jackson    |              0.350066 |
    | 41 | Malaki Starks      |              0.348301 |
    | 42 | Jaxson Dart        |              0.340907 |
    | 43 | Jahdae Barron      |              0.340761 |
    | 44 | Walter Nolen       |              0.334408 |
    | 45 | Tyler Warren       |              0.33411  |
    | 46 | Alfred Collins     |              0.32853  |
    | 47 | Abdul Carter       |              0.315731 |
    | 48 | Quinshon Judkins   |              0.303585 |
    | 49 | Mason Taylor       |              0.295869 |

Now that we know how the model feels about the 2025 prospects, it is
important to remember that this is just one model and one person’s
opinion. Fully tuned the model was still only 54% accurate. Despite
this, the importance of machine learning interpretability cannot be
understated, so let’s take a look at the SHAP values to see what the
model is thinking.

``` python
# Get SHAP values
explainer = shap.Explainer(xgb_final)
shap_values = explainer(df_tfidf_25)

# Show top 10 SHAP values
shap.summary_plot(shap_values, df_tfidf_25, plot_type="bar", max_display=10)

# Plot beeswarm
shap.plots.beeswarm(shap_values)

# Waterfall for T.J. Sanders
shap.plots.waterfall(shap_values[41])

# Waterfall for Travis Hunter
shap.plots.waterfall(shap_values[1])

# Waterfall for Abdul Carter
shap.plots.waterfall(shap_values[0])
```

<p align="center">

<img src="https://github.com/therealsmithy/nfl-draft-preview/blob/main/vizzes/report-beeswarm-v1.jpeg"/>
<figcaption align="center">

Beeswarm Visualization of SHAP values
</figcaption>

</p>

The model assigns high values of “starter”, “move”, and “overall” with a
negative impact on the likelihood of being a first rounder. These words
may be tough to interpret in this scenario. Moving down, higher values
of slot is also associated with lower first-round likelihood. A “slot”
reciever is a player who lines up in the slot position, which is between
the offensive line and the outside receiver. This position is typically
not as highly valued as the outside receiver position, which may explain
the negative impact on the likelihood of being a first rounder. As
expected, higher values of “size”, “tackle”, and “quickness” are
associated with a higher likelihood of being a first rounder. These are
all attributes that are highly valued in the NFL.

Next, let’s take a look at some interesting player cases and see why the
model attributed the probability that it did to them. First up, T.J.
Sanders.

<p align="center">

<img src="https://github.com/therealsmithy/nfl-draft-preview/blob/main/vizzes/sanders-waterfall-v1.jpeg"/>
<figcaption align="center">

Waterfall Visualization of Important Features influencing T.J. Sanders
</figcaption>

</p>

As previously stated, the words “starter” and “move” negatively impact a
player’s likelihood of being a first rounder. In Sanders’ case, these
words do not appear in his scouting report, leading to them greatly
impacting his first-round likelihood. The words “blocks”, “explosive”,
and “tackle” all appear in his report with a positive impact.

The next player to inspect is Travis Hunter.

<p align="center">

<img src="https://github.com/therealsmithy/nfl-draft-preview/blob/main/vizzes/hunter-waterfall-v1.jpeg"/>
<figcaption align="center">

Waterfall Visualization of Important Features influencing Travis Hunter
</figcaption>

</p>

Just like Sanders, Hunter does not have the words “starter” or “move” in
his report. For anyone who has kept up with college football discourse
in 2024, one of Hunter’s greatest attributes is his ability to play both
sides of the ball. This is reflected in the word “time” having a
positive value and one of the highest impacts on his first-round
likelihood.

Finally, let’s take a look at Abdul Carter. Carter is widely considered
as one of the best prospects in this class, but the model has him very
low on the first-round likelihood list. Let’s inspect why.

<p align="center">

<img src="https://github.com/therealsmithy/nfl-draft-preview/blob/main/vizzes/carter-waterfall-v1.jpeg"/>
<figcaption align="center">

Waterfall Visualization of Important Features influencing Abdul Carter
</figcaption>

</p>

When looking at this graph, the high negative impact of the word “move”
jumps off the screen. Carter’s ability to move between positions on the
defense, whether on the line or as a linebacker, is a key part of his
game. The model interprets “move” as a negative attribute, which is
likely why Carter is so low on the first-round likelihood list.

## Discussion

At the end of the day, Insiders like Jeremiah are paid for their ability
to generate buzz and “clicks”. As far as I know they do not voice their
opinions on players so that people can run Machine Learning models on
them but that is exactly what I did. The low model accuracy and the
negative impact of certain words on a player’s likelihood of being a
first rounder show that the model may not be the best way to predict a
player’s draft position. However, the SHAP values show that the model is
able to pick up on certain attributes that are highly valued in the NFL,
such as size and quickness. The main takeaway from this project is that
while scouting reports may be interesting to read and generate buzz,
they may not be the best way to predict a player’s draft position.
