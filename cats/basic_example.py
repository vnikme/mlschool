import numpy as np
from catboost import CatBoostClassifier


def generate_data(a, b, sigma, n):
    x = np.random.randn(n, 2)
    # y = a * x0 + b * x1
    s = np.sum(x * [a, b], axis=1)
    y = np.zeros(n, dtype=np.float32)
    y[s >= 0] = 1
    y[s < 0] = -1
    x += sigma * np.random.randn(n, 2)
    return x, y


class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
            der1 = targets[index] - p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


def main():
    np.random.seed(1)
    x, y = generate_data(0.5, 0.3, 0.3, 100)
    x_train, y_train = x[:90], y[:90]
    x_test, y_test = x[90:], y[90:]
    model = CatBoostClassifier(
        loss_function=LoglossObjective(),
        iterations=10,
        learning_rate=0.03,
        depth=2,
        eval_metric='AUC'
    )
    # Fit model
    model.fit(x_train, y_train)
    # Print predictions
    outputs_train = model.predict_proba(x_train)
    outputs_test = model.predict_proba(x_test)
    print(outputs_test)


if __name__ == '__main__':
    main()

