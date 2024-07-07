from typing import Dict
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

# Mini-Spielerei. Das dauert noch bis wir uns hierum kümmern.


def grid_tune_hyperparameters(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int = 5,
    random_state: int = 42,
    verbose: int = 0
) -> GridSearchCV:
    """
    Führt eine Hyperparameter-Tuning mit GridSearchCV durch.

    Parameters:
    - model: Das zu tunende Modell
    - X_train: Trainings-Merkmalsmatrix
    - y_train: Trainings-Zielvariable
    - param_grid: Dictionary mit den Hyperparametern
    - cv_splits: Anzahl der Cross-Validation-Splits
    - random_state: Zufallszustand für die Reproduzierbarkeit
    - verbose: Verbose-Level für GridSearchCV

    Returns:
    - grid_search: Das GridSearchCV-Objekt nach dem Fitting
    """
    param_grid = {
        'n_estimators': [50],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [10, 20, 50, None],
        'min_samples_leaf': [3, 5, 15, None],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                         random_state=random_state)
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=verbose, n_jobs=2)
    grid_search.fit(X_train, y_train)

    return grid_search


def randomize_tune_hyperparameters(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_distributions: Dict[str, list],
    cv_splits: int = 5,
    n_iter: int = 10,
    random_state: int = 42,
    verbose: int = 2
) -> RandomizedSearchCV:
    """
    Führt eine Hyperparameter-Tuning mit RandomizedSearchCV durch.

    Parameters:
    - model: Das zu tunende Modell
    - X_train: Trainings-Merkmalsmatrix
    - y_train: Trainings-Zielvariable
    - param_distributions: Dictionary mit den Hyperparametern und deren Verteilungen
    - cv_splits: Anzahl der Cross-Validation-Splits
    - n_iter: Anzahl der zufälligen Kombinationen, die ausprobiert werden sollen
    - random_state: Zufallszustand für die Reproduzierbarkeit
    - verbose: Verbose-Level für RandomizedSearchCV

    Returns:
    - randomized_search: Das RandomizedSearchCV-Objekt nach dem Fitting
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                         random_state=random_state)
    randomized_search = RandomizedSearchCV(
        estimator=model, param_distributions=param_distributions, n_iter=n_iter,
        cv=cv, scoring='accuracy', verbose=verbose, random_state=random_state, n_jobs=2
    )
    randomized_search.fit(X_train, y_train)

    return randomized_search
