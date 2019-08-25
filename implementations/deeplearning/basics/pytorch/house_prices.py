import numpy as np
import torch
from torch.autograd import Variable

# Create input and output
xs = np.array([1, 2, 3, 4.0, 5, 6, 8, 9, 10], dtype=np.float32).reshape(-1, 1)
ys = np.array([100, 150, 200, 250, 300, 350, 450, 500, 550], dtype=np.float32).reshape(-1, 1)

# Create model , define parameters
model = torch.nn.Sequential(torch.nn.Linear(1, 1))
learningRate = 0.01

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

optimizer.zero_grad()

# Setup input and labels based on device availablity
if torch.cuda.is_available():
    inputs = Variable(torch.from_numpy(xs).cuda())
    labels = Variable(torch.from_numpy(ys).cuda())
else:
    inputs = Variable(torch.from_numpy(xs))
    labels = Variable(torch.from_numpy(ys))
outputs = model(inputs)
loss = criterion(outputs, labels)
print(loss)

# get gradients w.r.t to parameters
loss.backward()
optimizer.step()

test = torch.from_numpy(np.array([7], dtype=np.float32).reshape(-1, 1))
print(model(test))
