"""Model config file."""
import numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

NTREES = 10


class SKModel:
    def __init__(self, metadata, random_seed) -> None:
        self.ord_imp = Imputer(missing_values="NaN", strategy="mean", axis=0,
                               verbose=0, copy=True)
        self.cat_imp = Imputer(missing_values=-1,
                               strategy="most_frequent",
                               axis=0, verbose=0, copy=True)

        psize = (2 * metadata.halfwidth + 1)**2
        if metadata.ncategories:
            n_values = [k for k in metadata.ncategories for _ in range(psize)]
            self.enc = OneHotEncoder(n_values=n_values,
                                     categorical_features="all",
                                     dtype=np.float32, sparse=False)

        self.est = RandomForestClassifier(n_estimators=NTREES,
                                          random_state=random_seed)

    def fit(self, Xo: np.ndarray, Xc: np.ndarray, Y: np.array) -> None:
        X_list = []
        if Xc is not None:
            Xc.data[Xc.mask] = -1
            X_cat_imp = self.cat_imp.fit_transform(Xc.data)
            X_onehot = self.enc.fit_transform(X_cat_imp)
            X_list.append(X_onehot)
        if Xo is not None:
            Xo.data[Xo.mask] = np.nan
            X_imputed = self.ord_imp.fit_transform(Xo.data)
            X_list.append(X_imputed)
        X = np.concatenate(X_list, axis=1)
        self.est.fit(X, Y)


    def predict(self, Xo: np.ma.MaskedArray, Xc: np.ma.MaskedArray,
                percentiles: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        if Xc is not None:
            Xc.data[Xc.mask] = -1
            X_cat_imp = self.cat_imp.transform(Xc.data)
            X_onehot = self.enc.transform(X_cat_imp)
            X_list.append(X_onehot)
        if Xo is not None:
            Xo.data[Xo.mask] = np.nan
            X_imputed = self.ord_imp.transform(Xo)
            X_list.append(X_imputed)
        X = np.concatenate(X_list, axis=1)
        Py = self.est.predict_proba(X)
        Ey = np.argmax(Py, axis=1)
        return Ey, Py
