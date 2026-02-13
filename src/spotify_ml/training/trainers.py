from __future__ import annotations

from sklearn.model_selection import RandomizedSearchCV


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def tune_model(
    model,
    param_distributions: dict,
    x_train,
    y_train,
    scoring: str,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
):
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
