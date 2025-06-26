from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

import autograd

torch.manual_seed(42)


def load_data(
    batch_size=64,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return (train_loader, test_loader)


class Network(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


def network(
    x: autograd.Tensor,
    w1: autograd.Tensor,
    w2: autograd.Tensor,
    b1: autograd.Tensor,
    b2: autograd.Tensor,
) -> autograd.Tensor:
    x = x @ w1.transpose() + b1
    x = x.relu()
    x = x @ w2.transpose() + b2
    return x.softmax()


def unwrap(t: autograd.Tensor | None) -> autograd.Tensor:
    if t is None:
        raise ValueError("Tensor is None")
    return t


def main(hidden_size: int = 100):
    input_size = 28 * 28
    output_size = 10

    train_loader, test_loader = load_data()

    torch_net = Network(input_size, hidden_size, output_size)
    optimizer = torch.optim.SGD(torch_net.parameters(), lr=0.01)

    w1 = autograd.Tensor.from_torch(torch_net.fc1.weight, requires_grad=True)
    b1 = autograd.Tensor.from_torch(torch_net.fc1.bias, requires_grad=True)
    w2 = autograd.Tensor.from_torch(torch_net.fc2.weight, requires_grad=True)
    b2 = autograd.Tensor.from_torch(torch_net.fc2.bias, requires_grad=True)
    lr = autograd.Tensor([1], [0.01], requires_grad=False, grad=None, graph=None)

    torch_loss = []
    autograd_loss = []

    for k, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch_size = x.shape[0]
        x = x.view(batch_size, input_size)
        y_true = torch.zeros(batch_size, output_size)
        y_true[torch.arange(batch_size), y] = 1.0

        # Torch model
        y_pred = torch_net(x)
        loss = ((y_pred - y_true) ** 2).sum()
        torch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Autograd model
        x = autograd.Tensor.from_torch(x, requires_grad=False)
        y_true = autograd.Tensor.from_torch(y_true, requires_grad=False)
        y_pred = network(x, w1, w2, b1, b2)
        loss = ((y_pred - y_true) * (y_pred - y_true)).reduce_sum()
        loss.backward(None)
        autograd_loss.append(loss.get_data()[0])
        w1 = w1 - lr * unwrap(w1.get_grad())
        w2 = w2 - lr * unwrap(w2.get_grad())
        b1 = b1 - lr * unwrap(b1.get_grad())
        b2 = b2 - lr * unwrap(b2.get_grad())
        for t in [w1, w2, b1, b2]:
            t.set_grad(None)
            t.set_graph(None)

    torch_right, torch_wrong, autograd_right, autograd_wrong = 0, 0, 0, 0
    for x, y in test_loader:
        batch_size = x.shape[0]
        x = x.view(batch_size, input_size)

        # Torch model
        y_pred = torch_net(x)
        y_pred = torch.max(y_pred, dim=1).indices
        torch_right += (y == y_pred).sum()
        torch_wrong += (y != y_pred).sum()

        # Autograd model
        y_pred = network(autograd.Tensor.from_torch(x), w1, w2, b1, b2).to_torch()
        y_pred = torch.max(y_pred, dim=1).indices
        autograd_right += (y == y_pred).sum()
        autograd_wrong += (y != y_pred).sum()

    print(f"Torch accuracy: {torch_right / (torch_right + torch_wrong) * 100}%")
    print(
        f"Autograd accuracy: {autograd_right / (autograd_right + autograd_wrong) * 100}%"
    )

    plt.plot(torch_loss, label="Torch loss")
    plt.plot(autograd_loss, label="Autograd loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
