import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(), # Image [0, 255] -> Tensor [0,1]
    transforms.Normalize((0.5,), (0.5,)) #Pixel values will be around 0, thus we obtain -> Tensor [-1, 1]
])

vis_transform = transforms.Compose([transforms.ToTensor()])

# Create training set and define training dataloader
train_dataset = torchvision.datasets.MNIST(root = './images', train=True, download=True, transform = transform)
validation_dataset = torchvision.datasets.MNIST(root = './images', train=True, download=True, transform = transform)
testing_dataset = torchvision.datasets.MNIST(root = './images', train=False, download=True, transform = transform)
vis_dataset = torchvision.datasets.MNIST(root ='./images', train=False, download=True, transform = vis_transform)

# Create test set and define test dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle=True)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = 64, shuffle=False)
vis_loader = torch.utils.data.DataLoader(vis_dataset, batch_size = 64, shuffle = False)


def show5(img_loader):
    dataiter = iter(img_loader)
    
    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        print(int(labels[i].detach()))
    
        image = images[i].numpy()
        plt.imshow(image.T.squeeze().T)
        plt.show()

# Explore data
show5(vis_loader)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)  # First layer
        self.fc2 = nn.Linear(256, 128)     # Second layer
        self.fc3 = nn.Linear(128, 64)      #Third Layer
        self.fc4 = nn.Linear(64, 10)      # Output layer

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function here, as we'll use CrossEntropyLoss
        return x


net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


num_epochs = 5
loss_print = 250
train_loss_history = list()
val_loss_history = list()
val_accuracy_history = list()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        if (i + 1) % loss_print == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / (i + 1):.4f}')
    
    epoch_accuracy = 100 * train_correct / train_total
    epoch_loss = running_loss / len(train_loader)
    train_loss_history.append(epoch_loss)
    val_accuracy_history.append(epoch_accuracy)

    net.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): 
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_accuracy = 100 * val_correct / val_total
    val_epoch_loss = val_running_loss / len(test_loader)
    val_loss_history.append(val_epoch_loss)
    val_accuracy_history.append(val_epoch_accuracy)

    print(f'Epoch {epoch + 1} training accuracy: {epoch_accuracy:.2f}% training loss: {epoch_loss:.5f}')
    print(f'Epoch {epoch + 1} validation accuracy: {val_epoch_accuracy:.2f}% validation loss: {val_epoch_loss:.5f}')

print('Done!')

plt.plot(train_loss_history, label = "Training Loss")
plt.plot(val_loss_history, label = "Validation Loss")
plt.legend()
plt.show()



# %%
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass to get outputs
        outputs = net(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum().item()

# Calculate the accuracy
accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {accuracy:.2f}%')


num_epochs = 10
loss_print = 250
train_loss_history = list()
val_loss_history = list()
val_accuracy_history = list()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        if (i + 1) % loss_print == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / (i + 1):.4f}')
    
    epoch_accuracy = 100 * train_correct / train_total
    epoch_loss = running_loss / len(train_loader)
    train_loss_history.append(epoch_loss)
    val_accuracy_history.append(epoch_accuracy)

    net.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): 
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_accuracy = 100 * val_correct / val_total
    val_epoch_loss = val_running_loss / len(test_loader)
    val_loss_history.append(val_epoch_loss)
    val_accuracy_history.append(val_epoch_accuracy)

    print(f'Epoch {epoch + 1} training accuracy: {epoch_accuracy:.2f}% training loss: {epoch_loss:.5f}')
    print(f'Epoch {epoch + 1} validation accuracy: {val_epoch_accuracy:.2f}% validation loss: {val_epoch_loss:.5f}')

print('Done!')
#We achieved 97.20% accuracy!
torch.save(net, './badr_MNIST_model.pt')





