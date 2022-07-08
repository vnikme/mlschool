import torch
import numpy as np
import matplotlib.pyplot as plt


class LinearModel(torch.nn.Module):
    def __init__(self, size):
        super(LinearModel, self).__init__()

        # parameters
        self.size = size

        # model
        self.model = torch.nn.Linear(size, 1)


    def forward(self, data):
        return self.model(data).reshape([-1])


def generate_data(a, b, sigma, n):
    x = np.random.randn(n, 2)
    # y = a * x0 + b * x1
    s = np.sum(x * [a, b], axis=1)
    y = np.zeros(n, dtype=np.float32)
    y[s > 0] = 1
    x += sigma * np.random.randn(n, 2)
    return torch.from_numpy(x.astype(dtype=np.float32)), torch.from_numpy(y)


def plot_data(x, y, a, b):
    plt.scatter(x[:,0], x[:,1], c = [('r' if x > 0.0 else 'b') for x in y])
    if abs(b) > 1e-5:
        plt.plot([-3, 3], [a*3/b, -a*3/b])
    else:
        plt.plot([0, 0], [-3, 3])
    plt.axis([-3, 3, -3, 3])
    plt.show()


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    x, y = generate_data(0.5, 0.3, 0.3, 100)
    x_train, y_train = x[:10], y[:10]
    x_test, y_test = x[10:], y[10:]
    model = LinearModel(2)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.0200117)
    criterion = torch.nn.BCEWithLogitsLoss()
    losses_train, losses_test = [], []
    for i in range(30000):
        optimiser.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        train_loss = float(loss)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            outputs = model(x_test)
            test_loss = float(criterion(outputs, y_test))
        losses_train.append(train_loss)
        losses_test.append(test_loss)
        print('Train: {}, test: {}'.format(train_loss, test_loss))
    print(model.model.weight)
    plt.plot(losses_train)
    plt.plot(losses_test)
    plt.show()
    plot_data(x_train, y_train, model.model.weight[0][0], model.model.weight[0][1])
    plot_data(x_test, y_test, model.model.weight[0][0], model.model.weight[0][1])


if __name__ == '__main__':
    main()

