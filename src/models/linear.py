"""Линейные семейства моделей: PCR, PLS и штрафные регрессии.

Соответствуют панелям A и B таблиц 1-2 статьи. Все модели стандартизуют
предикторы внутри пайплайна, поэтому среднее и дисперсия считаются только
по обучающей части каждого расширяющегося окна.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class AddSquares(BaseEstimator, TransformerMixin):
    """Дополняет компоненты их квадратами (спецификация PCA-Squared)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.hstack([X, X**2])


def make_pcr(n_components: int, squared: bool = False) -> Pipeline:
    """Регрессия на первые n главных компонент предикторов.

    При n_components = 10 и десяти форвардных ставках на входе это, с точностью
    до поворота базиса, исходная спецификация Cochrane & Piazzesi (2005).
    """
    steps = [
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
    ]
    if squared:
        steps.append(("squares", AddSquares()))
    steps.append(("ols", LinearRegression()))
    return Pipeline(steps)


def make_pls(n_components: int) -> Pipeline:
    """Частные наименьшие квадраты: компоненты строятся с учётом цели."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pls", PLSRegression(n_components=n_components, scale=False)),
        ]
    )


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])


def make_lasso(alpha: float) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=50_000))]
    )


def make_elastic_net(alpha: float, l1_ratio: float = 0.5) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50_000),
            ),
        ]
    )


# Сетки штрафов подбираются на validation-части каждого окна.
# Для ridge верхняя граница сетки связывает: на форвардных ставках validation
# почти всегда просит максимальное сжатие, то есть скатывается к модели с одной
# константой. Расширение сетки до 1e8 меняет R2_oos лишь на доли процента
# (-11.4% -> -12.1% для rx_2), поэтому оставлена умеренная сетка.
RIDGE_GRID = np.logspace(-4, 4, 25)
LASSO_GRID = np.logspace(-6, 0, 25)
ENET_GRID = np.logspace(-6, 0, 25)
