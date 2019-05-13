from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


class DoubleMLModel(object):

    def __init__(self, text_features, outcomes, confounders, model_g_a, model_g_b, model_m_a, model_m_b, frac_a=0.5):
        self.A = text_features
        self.Y = outcomes
        self.T = confounders

        self.A_a = self.A_b = self.Y_a = self.Y_b = self.T_a = self.T_b = None

        self.model_g_a = model_g_a
        self.model_m_a = model_m_a
        self.model_g_b = model_g_b
        self.model_m_b = model_m_b

        self.theta_a = None
        self.theta_b = None

        self.split(frac_a=frac_a)


    def split(self, frac_a):
        self.A_a, self.A_b, self.Y_a, self.Y_b, self.T_a, self.T_b = train_test_split(self.A, self.Y, self.T,
                                                                                      test_size=frac_a,
                                                                                      shuffle=True,
                                                                                      random_state=42)

    @staticmethod
    def train_model_cls(x_train, y_train, x_test, y_test, model, y_hat_name=["metric", "y_train_hat", "y_test_hat"]):
        model.fit(x_train, y_train.ravel())
        y_train_hat = model.predict(x_train)
        y_test_hat = model.predict(x_test)

        scores = [
            ["Accuracy", accuracy_score(y_train, y_train_hat), accuracy_score(y_test, y_test_hat)],
            ["F1", f1_score(y_train, y_train_hat), f1_score(y_test, y_test_hat)]
        ]

        print(tabulate(scores, headers=y_hat_name))

        return y_test_hat

    @staticmethod
    def train_model_reg(x_train, y_train, x_test, y_test, model, y_hat_name=["metric", "y_train_hat", "y_test_hat"]):
        model.fit(x_train, y_train.ravel())
        y_train_hat = model.predict(x_train)
        y_test_hat = model.predict(x_test)

        scores = [
            ["MSE", mean_squared_error(y_train, y_train_hat), mean_squared_error(y_test, y_test_hat)]
        ]

        print(tabulate(scores, headers=y_hat_name))

        return y_test_hat


    # @staticmethod
    # def class_labels(y):
    #     labels = np.unique(y)
    #     count = [np.count_nonzero(y == c) for c in labels]
    #     weights = [c / sum(count) for c in count]
    #
    #     print("label_count:\t", count)
    #
    #     return labels, count, weights

    # def normalize(self, with_mean=False):
    #     scaler = StandardScaler(with_mean=with_mean)
    #     scaler.fit(X)
    #     X_train = scaler.transform(X_train)
    #     X_test = scaler.transform(X_test)