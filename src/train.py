from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

def train_logistic_regression(
        X_train,
        y_train):
    model = LogisticRegression(
        max_iter=1000
    )
    model.fit(
        X_train,
        y_train
    )
    return model

def train_random_forest_classifier(
        X_train,
        y_train):
    model = RandomForestClassifier(
        random_state=42
    )
    model.fit(
        X_train,
        y_train
    )
    return model

def train_gradient_boosting_classifier(
        X_train,
        y_train):
    model = GradientBoostingClassifier(
        random_state=42
    )
    model.fit(
        X_train,
        y_train
    )
    return model

def train_random_forest_regressor(
        X_train,
        y_train):
    model = RandomForestRegressor(
        random_state=42
    )
    model.fit(
        X_train,
        y_train
    )
    return model