import sys
import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV


def top_k_error(pred_ys, y, k=2):
    return np.mean([int(y0 in pred_y[:k]) for y0, pred_y in zip(y, pred_ys)])


class OvoLearner:

    def __init__(self, model_gen, n_classes):
        self.model_gen = model_gen
        self.fitted_models = dict()
        self.n_c = n_classes

    def fit(self, X, y):
        n_models = int(self.n_c*(self.n_c-1)/2)
        i = 0
        for k, l in combinations(range(self.n_c), 2):
            filt = np.logical_or(y == k, y == l)
            X_f = X[filt]
            y_f = y[filt]
            y_classif = (y_f == k).astype(int)

            model = self.model_gen()
            model.fit(X_f, y_classif)

            self.fitted_models[(k, l)] = model
            i += 1
            print("Fitted model {} of {} - {}".format(
                i, n_models, datetime.datetime.now()), flush=True)

    def predict(self, X):
        score = np.zeros((X.shape[0], self.n_c))

        for k, l in combinations(range(self.n_c), 2):
            y = self.fitted_models[(k, l)].predict(X)
            score[:, k] += y
            score[:, l] += 1-y
        return np.argsort(score, axis=1)[:, ::-1]  # decreasing scores


def model_gen(max_iter=2000):
    return LogisticRegressionCV(cv=5, max_iter=max_iter, multi_class="auto")


def main():
    if len(sys.argv) >= 2:
        assert sys.argv[1] in {"digits", "fashion"}
        df_train = pd.read_csv("{}-mnist/mnist_train.csv".format(sys.argv[1]),
                               header=None)
        df_test = pd.read_csv("{}-mnist/mnist_test.csv".format(sys.argv[1]),
                              header=None)
        X_train, y_train = df_train.values[:, 1:], df_train[0].values
        X_test, y_test = df_test.values[:, 1:], df_test[0].values

        # No need for scaler

        print("Fitting PCA - {}".format(datetime.datetime.now()), flush=True)
        prop_var = 0.95
        pca = PCA(n_components=200)
        X_train_red = pca.fit_transform(X_train)
        n_components = np.where(pca.explained_variance_ratio_.cumsum() > prop_var)[0][0]
        print("Number of PCA components = {}".format(n_components), flush=True)
        X_train_red = X_train_red[:, :n_components]
        X_test_red = pca.transform(X_test)[:, :n_components]

        print("Fitting model - {}".format(datetime.datetime.now()), flush=True)
        model = model_gen()
        model.fit(X_train_red, y_train)

        print("Fitting OVO - {}".format(datetime.datetime.now()), flush=True)
        ovo_model = OvoLearner(model_gen, 10)
        ovo_model.fit(X_train_red, y_train)

        print("Done fitting - {}".format(datetime.datetime.now()), flush=True)

        print("Predicting model - {}".format(datetime.datetime.now()),
              flush=True)
        y_pred_test = np.argsort(model.predict_proba(X_test_red),
                                 axis=1)[:, ::-1]
        print("Predicting OVO - {}".format(datetime.datetime.now()),
              flush=True)
        y_ovo_pred_test = ovo_model.predict(X_test_red)
        print("Done predicting - {}".format(datetime.datetime.now()),
              flush=True)

        dump(model, "models/model_{}.joblib".format(sys.argv[1]))
        dump(ovo_model, "models/ovo_model_{}.joblib".format(sys.argv[1]))

        for k in range(1, 6):
            acc = top_k_error(y_pred_test, y_test, k=k)
            acc_ovo = top_k_error(y_ovo_pred_test, y_test, k=k)
            print("Top {} accuracy = {:.3f}".format(k, acc), flush=True)
            print("Top {} accuracy for OVO = {:.3f}".format(k, acc_ovo),
                  flush=True)


if __name__ == "__main__":
    main()
