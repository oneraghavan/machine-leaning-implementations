import numpy as np
import torch
from torchvision import datasets, transforms


##Pytorch does not have the equivalent flatten layer
class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

#ToDo Add ability to add callbacks to this
model = torch.nn.Sequential(
    Flatten(),
    torch.nn.Linear(784, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10),
    torch.nn.Softmax()
)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

optimizer.zero_grad()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0,), (255,))
                   ])),
    batch_size=2048, shuffle=True)

device = torch.device("cpu")
for epoch in range(1, 20):
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % 100 == 0:
            predictions = [np.argmax(out) for out in output.detach().numpy()]
            correct_local = (target.detach().numpy() == predictions).sum()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAccuracy {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), 100. * correct_local / data.shape[0]))

    print('\nTraining set: , Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
