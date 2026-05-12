import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Method_CNN(nn.Module):
    def __init__(self, method_name, dummy):
        super(Method_CNN, self).__init__()
        self.method_name = method_name
        self.loss_history = []
        self.acc_history = []

        # First conv layer will be initialized dynamically
        self.conv1 = None
        self.bn1 = None

        # Deeper CNN
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fully connected layers initialized dynamically
        self.fc1 = None
        self.fc2 = None

        # Send fixed conv layers to device
        self.conv2.to(self.device)
        self.conv3.to(self.device)
        self.bn2.to(self.device)
        self.bn3.to(self.device)

    def reshape_input(self, X):
        # Normalize inputs to [0,1] and convert to torch
        X = X.astype(np.float32)
        if X.max() > 1.0:
            X = X / 255.0
        return torch.tensor(X, dtype=torch.float32).to(self.device)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(self, X_train, y_train, X_test, y_test, epochs=30, batch_size=64):
        X_train = self.reshape_input(X_train)

        # Convert labels to 0-based
        y_train = np.array(y_train) - np.min(y_train)
        y_test = np.array(y_test) - np.min(y_test)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        # Initialize conv1/bn1 and fc layers dynamically
        if self.conv1 is None:
            in_channels = X_train.shape[1]  # 1 for MNIST, 3 for CIFAR/ORL
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1).to(self.device)
            self.bn1 = nn.BatchNorm2d(32).to(self.device)

            with torch.no_grad():
                sample_X = X_train[:1]
                x = self.relu(self.bn1(self.conv1(sample_X)))
                x = self.pool(x)
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.pool(x)
                x = self.relu(self.bn3(self.conv3(x)))
                x = self.pool(x)
                n_features = x.numel()

            self.fc1 = nn.Linear(n_features, 192).to(self.device)
            self.fc2 = nn.Linear(192, len(np.unique(y_train))).to(self.device)

            self.optimizer = optim.Adam(
                list(self.conv1.parameters()) +
                list(self.bn1.parameters()) +
                list(self.conv2.parameters()) +
                list(self.bn2.parameters()) +
                list(self.conv3.parameters()) +
                list(self.bn3.parameters()) +
                list(self.fc1.parameters()) +
                list(self.fc2.parameters()),
                lr=0.001
            )

        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            correct = 0

            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train_tensor[indices]

                self.optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                self.loss_history.append(loss.item())
                epoch_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()

            acc_epoch = correct / n_samples
            self.acc_history.append(acc_epoch)

            if epoch % 5 == 0:
                print(f"Epoch {epoch} Loss: {epoch_loss / (n_samples // batch_size):.4f} Acc: {acc_epoch:.4f}")

        # Testing
        X_test = self.reshape_input(X_test)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.forward(X_test)
            _, predicted = torch.max(outputs.data, 1)

        # Metrics
        accuracy = accuracy_score(y_test, predicted.cpu().numpy())
        f1 = f1_score(y_test, predicted.cpu().numpy(), average='macro')
        precision = precision_score(y_test, predicted.cpu().numpy(), average='macro', zero_division=0)
        recall = recall_score(y_test, predicted.cpu().numpy(), average='macro', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'true_y': y_test,
            'pred_y': predicted.cpu().numpy()
        }

        return metrics

    def run(self, X_train, y_train, X_test, y_test):
        return self.fit(X_train, y_train, X_test, y_test)