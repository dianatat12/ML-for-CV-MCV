from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def fit_svm(train_X, train_y, C=1.0, kernel="rbf"):
    svc = SVC(C=C, kernel=kernel)
    svc.fit(train_X, train_y)
    return svc


def fit_knn(train_X, train_y, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_X, train_y)
    return knn


def fit_xgboost(train_X, train_y, max_depth=3, learning_rate=0.1, n_estimators=100):
    xgb = XGBClassifier(
        max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators
    )
    xgb.fit(train_X, train_y)
    return xgb


def fit_decision_tree(train_X, train_y, max_depth=None, criterion="gini"):
    dt = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    dt.fit(train_X, train_y)
    return dt


def fit_logistic_regression(train_X, train_y, C=1.0, solver="lbfgs"):
    lr = LogisticRegression(C=C, solver=solver)
    lr.fit(train_X, train_y)
    return lr


def get_ML_model(MODEL_NAME: str, train_X, train_y, **kwargs):
    if MODEL_NAME == "SVM":
        return fit_svm(train_X, train_y, **kwargs)
    elif MODEL_NAME == "KNN":
        return fit_knn(train_X, train_y, **kwargs)
    elif MODEL_NAME == "XGBOOST":
        return fit_xgboost(train_X, train_y, **kwargs)
    elif MODEL_NAME == "DecisionTree":
        return fit_decision_tree(train_X, train_y, **kwargs)
    elif MODEL_NAME == "LogisticRegression":
        return fit_logistic_regression(train_X, train_y, **kwargs)
    else:
        raise ValueError("Unknown model name")
