from pathlib import Path

import torch
import torchvision

from exptorch import create_experiments, Params, Struct

from examples.models import MLP


config_dir = Path("./")
train_params = Params(fixed=Struct(epochs=5, batch_size=16))

model = MLP
model_params = Params(
    fixed=Struct(input_dim=28 * 28, output_dim=10),
    channels=[[512, 64], [512, 128, 64], [512, 128, 64, 32]],
    activation=[torch.nn.ReLU, torch.nn.Tanh],
)

losses = Struct(ce=torch.nn.CrossEntropyLoss, mse=torch.nn.MSELoss)

optimizers = [torch.optim.Adam, torch.optim.SGD]
optimizer_params = Params(lr=[0.001, 0.005, 0.01])

train_dataset = torchvision.datasets.MNIST
train_dataset_params = Params(
    fixed=Struct(
        root="./",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    ),
)

if __name__ == "__main__":
    create_experiments(
        train_params=train_params,
        model=MLP,
        model_params=model_params,
        optimizers=optimizers,
        optimizer_params=optimizer_params,
        losses=losses,
        train_dataset=train_dataset,
        train_dataset_params=train_dataset_params,
        save_dir=Path("./"),
        run=True,
    )
