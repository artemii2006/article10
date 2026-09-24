"""PCR, PLS и штрафные регрессии (панели A и B таблиц 1-2)."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

RIDGE_GRID = np.logspace(-4, 4, 25)
LASSO_GRID = np.logspace(-6, 0, 25)
ENET_GRID = np.logspace(-6, 0, 25)


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def pcr(X_train, y_train, X_test, n_components, squared=False):
    X_train, X_test = standardize(X_train, X_test)
    pca = PCA(n_components=n_components).fit(X_train)
    f_train, f_test = pca.transform(X_train), pca.transform(X_test)
    if squared:
        f_train = np.hstack([f_train, f_train**2])
        f_test = np.hstack([f_test, f_test**2])
    return LinearRegression().fit(f_train, y_train).predict(f_test)


def pls(X_train, y_train, X_test, n_components):
    X_train, X_test = standardize(X_train, X_test)
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_train, y_train)
    return model.predict(X_test).ravel()


def ridge(X_train, y_train, X_test, alpha):
    X_train, X_test = standardize(X_train, X_test)
    return Ridge(alpha=alpha).fit(X_train, y_train).predict(X_test)


def lasso(X_train, y_train, X_test, alpha):
    X_train, X_test = standardize(X_train, X_test)
    model = Lasso(alpha=alpha, max_iter=50_000)
    return model.fit(X_train, y_train).predict(X_test)


def elastic_net(X_train, y_train, X_test, alpha, l1_ratio=0.5):
    X_train, X_test = standardize(X_train, X_test)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50_000)
    return model.fit(X_train, y_train).predict(X_test)
