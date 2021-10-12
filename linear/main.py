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
    x, y = generate_data(-0.5, 0.3, 0.1, 1000)
    model = LinearModel(2)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.0200117)
    criterion = torch.nn.BCEWithLogitsLoss()
    for i in range(30000):
        optimiser.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        print(float(loss))
        loss.backward()
        optimiser.step()
        if i % 10000 == 0:
            plot_data(x, y, model.model.weight[0][0], model.model.weight[0][1])
    print(model.model.weight)


if __name__ == '__main__':
    main()

