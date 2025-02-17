import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
import matplotlib.pyplot as plt

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
plt.plot(x_axis, eval_results['validation_0']['error'], label = f'Learning Rate: {0.1}')
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
    'scale_pos_weight': [0.36],
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
    'max_depth': [5],
    'min_child_weight': [3],
    'learning_rate': [0.1], 
    'n_estimators': [100], 
    'subsample': [0.5, 0.75, 1.0],
    'colsample_bytree': [0.5, 0.75, 1.0], 
    'reg_alpha': [1e-5],
    'early_stopping_rounds':[10],
    'scale_pos_weight': [0.36],
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
    'max_depth': [5],
    'min_child_weight': [3],
    'learning_rate': [0.1], 
    'n_estimators': [100], 
    'subsample': [0.5],
    'colsample_bytree': [1.0], 
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    'early_stopping_rounds':[10],
    'scale_pos_weight': [0.36],
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


# Plot feature importance
# uh oh...
xgb.plot_importance(model, max_num_features = 10)
plt.show()
