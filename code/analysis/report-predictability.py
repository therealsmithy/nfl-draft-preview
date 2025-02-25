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

# Read in data
first_rounders = pd.read_csv('data/2020_to_2024_first_rounders.csv')
report_20 = pd.read_csv('data/scoutingreports/jeremiah_2020.csv')
report_21 = pd.read_csv('data/scoutingreports/jeremiah_2021.csv')
report_22 = pd.read_csv('data/scoutingreports/jeremiah_2022.csv')
report_23 = pd.read_csv('data/scoutingreports/jeremiah_2023.csv')
report_24 = pd.read_csv('data/scoutingreports/jeremiah_2024.csv')
report_25 = pd.read_csv('data/scoutingreports/jeremiah_2025.csv')
stopwords = pd.read_csv('data/stopwords.csv')

# Combine reports
reports_past = pd.concat([report_20, report_21, report_22, report_23, report_24])

# 1ST: BINARY CLASSIFICATION - Predict if a prospect will be a first rounder based on their scouting report

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

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(xgb_df['report_lemmatized'].apply(lambda x: ' '.join(x)))

# Make sure output is interpretable
feature_names = vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# INITIAL XGBoost classifier
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=102701)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df_tfidf, xgb_df['first_rounder'], test_size=0.2, random_state=102701)

# Fit to training data
xgb_clf.fit(X_train, y_train)

# Create basic predictions
y_pred = xgb_clf.predict(X_test)
initial_accuracy = accuracy_score(y_test, y_pred)
print(f'Initial accuracy: {initial_accuracy * 100:.2f}%')

# Check confusion matrix  to see where the model struggles
conf_matrix = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(conf_matrix).plot()
plt.show()

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
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

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

# Display the ranked prospects
print(report_25_ranked[['Name', 'first_rounder_proba']])

# Save the ranked prospects
report_25_ranked[['Name', 'first_rounder_proba']].to_csv('data/scoutingreports/jeremiah_2025_ranked.csv', index=False)

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

# 2ND: REGRESSION - Predict the exact draft position of a top 50 prospect based on their scouting report