import torch
import torch.nn
import torch.optim
import numpy as np


class LinearModel(nn.Module):
    def __init__(self, size):
        super(LinearModel, self).__init__()

        # parameters
        self.size = size

        # model
        self.model = torch.nn.Linear(size, 1)


    def forward(self, data):
        return self.model(data)


def generate_data(a, b, sigma, n):
    print(np.random.randn(n, 2))


def main():
    torch.manual_seed(1)
    generate_data()


if __name__ == '__main__':
    main()

