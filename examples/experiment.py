from pathlib import Path

import torch
import torchvision

from exptorch import create_experiments, Params, Struct

from examples.models import MLP

config_dir = Path("./")


train_params = Params(fixed=Struct(num_epochs=2, batch_size=16))

model = MLP
model_params = Params(
    fixed=Struct(input_dim=28 * 28, output_dim=10),
    channels=[[512, 64], [512, 128, 64], [512, 128, 64, 32]],
    activation=[torch.nn.ReLU, torch.nn.Tanh],
)

losses = Struct(loss=torch.nn.CrossEntropyLoss)

optimizers = Struct(adam=torch.optim.Adam, sgd=torch.optim.SGD)
optimizer_params = Params(lr=[0.001, 0.005, 0.01])


if __name__ == "__main__":
    create_experiments(
        model=model,
        optimizers=optimizers,
        losses=losses,
        dataset=torchvision.datasets.MNIST(root="./", train=True, download=True, transform=torchvision.transforms.ToTensor()),
        train_params=train_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        save_dir=Path("./"),
        run=True,
    )
