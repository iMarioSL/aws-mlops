################################
# Training a simple classifier #
################################

# NOTE: This file and `fit-model.ipynb` produce the same result. The only
# difference is that the notebook has more context (and requires more
# dependencies). You can use both files interchangeably.

# This notebook will fit a gradient boosted ensemble of trees to the infamous
# 1995 breast cancer dataset. The goal is to produce a model that can
# somewhat-accurately predict if a patient has breast cancer or not.
# We will then export the fitted model and deploy it using AWS Lambda.

# Imports
import pickle
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import f1_score

# Load dataset
all_data = load_breast_cancer()

# Features to pandas
X = pd.DataFrame(
    data=all_data['data'],
    columns=all_data['feature_names']
)

# Target to pandas
y = pd.DataFrame(
    data=(1 - all_data['target']), # Flip target labels
    columns=['malignant']
)

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature selection
# -----------------

# We will now instantiate a classifier with some generic hyperparameters. We
# will then recursively fit this model on the data, and in each step, we will
# drop the least important feature. We do this to limit the number of features
# needed to make a good prediction.

# Declare a splitter (used in cross validation)
cv_splitter = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# Instantiate model
clf = GradientBoostingClassifier(
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=3,
    max_features=9,
    random_state=42
)

# Instantiate feature eliminator
rfe = RFECV(
    estimator=clf,
    step=1,
    min_features_to_select=5,
    cv=cv_splitter,
    scoring='f1',
    n_jobs=-1
)

# Fit many models, each with less features
rfe = rfe.fit(
    X=X_train,
    y=y_train['malignant']
)

# Store top-five features
cols = rfe.get_feature_names_out()

# Print results
print(f'The optimal model uses {rfe.n_features_} features:')
print(rfe.get_feature_names_out())

# Model selection
# ---------------

# Now that we have found the best subset of features for the basic model, we
# will play around with its hyperparameters to find the best overall model.

# Hyperparameter candidates
grid = {
    'n_estimators': [500, 1000, 1500, 2000],
    'max_depth': [1, 2, 3]
}

# Instantiate search
search = GridSearchCV(
    estimator=clf,
    param_grid=grid,
    scoring='f1',
    n_jobs=-1,
    refit=True,
    cv=cv_splitter
)

# Find best combination
search = search.fit(
    X_train[cols],
    y_train['malignant']
)

# Final model
# -----------

# Now that we know both the best subset of features and hyperparameters, we can
# persist the model using pickle.

# Instantiate model
clf = GradientBoostingClassifier()

# Set parameters
clf = clf.set_params(
    **search.best_estimator_.get_params()
)

# Fit on whole training dataset
clf.fit(
    X_train[cols],
    y_train['malignant']
)

# Make predictions
pred_train = clf.predict(X_train[cols])
pred_test = clf.predict(X_test[cols])

# Score predictions
f1_train = f1_score(
    y_true=y_train,
    y_pred=pred_train
)
f1_test = f1_score(
    y_true=y_test,
    y_pred=pred_test
)

# Summary
print(f'F1-score on training data: {round(f1_train, 2)}')
print(f'F1-score on testing data: {round(f1_test, 2)}')

# Fit on all data
clf = clf.fit(
    X=X_train[cols].values, # Train without feature names
    y=y_train['malignant']
)

# Export model
pickle.dump(
    obj=clf,
    file=open('../results/clf.sav', 'wb')
)
