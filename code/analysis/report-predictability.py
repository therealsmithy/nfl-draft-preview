import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Lemmatize first
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(word) for word in w_tokenizer.tokenize(text)]
reports_past['report_lemmatized'] = reports_past['Report'].apply(lemmatize_text)

# Define dataframe
xgb_df = reports_past[['report_lemmatized', 'first_rounder']].copy()

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(xgb_df['report_lemmatized'].apply(lambda x: ' '.join(x)))

# Split data
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, xgb_df['first_rounder'], test_size=0.2, random_state=102701)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric=['auc', 'error'])
xgb_model.fit(X_train, y_train)

# Evaluate model performance
train_auc = xgb_model.score(X_train, y_train)
accuracy = accuracy_score(y_test, xgb_model.predict(X_test))