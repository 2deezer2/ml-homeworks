from __future__ import annotations

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_class = None,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False):
        self.base_model_class = DecisionTreeRegressor if base_model_class is None else base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list) #Не понял зачем это

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.feature_importances = []

    def fit_new_base_model(self, x, y, predictions):
        base_model = self.base_model_class(**self.base_model_params)
        #Генерю бутстрап выборку
        idx = np.random.choice(len(y), size=int(self.subsample * len(y)), replace=True)
        x_bootstrap, antigrad = x[idx], -self.loss_derivative(y[idx], predictions[idx])

        # Обучить базовую модель
        base_model.fit(x_bootstrap, antigrad)

        # Найти оптимальную гамму
        new_predictions = base_model.predict(x)
        optimal_gamma = self.find_optimal_gamma(y, predictions, new_predictions)

        # Добавить гамму, модель и предсказания в списки
        self.gammas.append(optimal_gamma)
        self.models.append(base_model)


    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        count_stop = 0

        if self.plot:
            self.history = {'val_score': [],
                            'train_score': []}

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            valid_predictions += self.models[-1].predict(x_valid) * self.gammas[-1] * self.learning_rate
            train_predictions += self.models[-1].predict(x_train) * self.gammas[-1] * self.learning_rate
            self.history['val_score'].append(roc_auc_score(y_valid, valid_predictions))
            self.history['train_score'].append(roc_auc_score(y_train, train_predictions))


            if self.early_stopping_rounds is not None:
                if self.loss_fn(y_valid, valid_predictions) >= self.validation_loss.min():
                    count_stop += 1
                else:
                    count_stop = 0
                    self.validation_loss[count_stop] = self.loss_fn(y_valid, valid_predictions)
                if count_stop >= self.early_stopping_rounds:
                    break
        if self.plot:
            data = self.history
            return sns.lineplot(data=data, palette='flare'), plt.title('Score while fitting'), plt.xlabel('iteration'), plt.ylabel('AUC_ROC')

    def predict_proba(self, x):
        result = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            result += gamma * model.predict(x) * self.learning_rate
        return (np.array([1 - self.sigmoid(result), self.sigmoid(result)]).T)
            
            
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        feature_importances = np.zeros((len(self.models), len(self.models[0].feature_importances_)))
    
        for i, model in enumerate(self.models):
            feature_importances[i] = model.feature_importances_
            
        return np.mean(feature_importances, axis=0)
