import torch 
import math 
import time
import augmentations as aug
import torchvision.datasets as datasets
from optimizers import LARS


def exclude_bias_and_norm(p):
    return p.ndim == 1

def adjust_learning_rate(epochs, base_lr, batch_size, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = base_lr * batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def get_loader( batch_size=64):
    transforms = aug.TrainTransform()
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    #datasets.ImageFolder(data_dir / "train", transforms)
    #sampler = torch.utils.data.Sampler(dataset)

    loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=1,
                    pin_memory=True,
                    shuffle=True,
                    )
    return loader 


def train(model, loader, batch_size=64, epochs=100, wd=1e-6):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        learning_rate = 0.2
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=wd,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
            )
        start_time = time.time()
        for epoch in range(epochs):
            for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
                x = x.to(device)
                y = y.to(device)

                lr = adjust_learning_rate(epochs, learning_rate, batch_size, optimizer, loader, step)

                optimizer.zero_grad()
                loss = model.forward(x, y)
                loss.backward()
                optimizer.step()
                print("step loss:", loss.item())

            current_time = time.time()
            print(f"epoch={epoch}, loss={loss.item()}, time={int(current_time - start_time)},lr={lr}")