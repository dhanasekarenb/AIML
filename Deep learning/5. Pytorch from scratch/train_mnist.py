import numpy as np
from torchvision import datasets

from Torch import (
    Sequential,
    Linear,
    Activation_ReLU,
    Softmax,
    CrossEntropyLoss,
    SGD
)


# -----------------------------
# One Hot Encoding
# -----------------------------
def one_hot(labels, num_classes=10):
    y = np.zeros((len(labels), num_classes))
    y[np.arange(len(labels)), labels] = 1
    return y


# -----------------------------
# Load MNIST
# -----------------------------
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True
)


# -----------------------------
# Convert to NumPy
# -----------------------------
X_train = train_dataset.data.numpy().astype(np.float32)
Y_train = train_dataset.targets.numpy()

X_test = test_dataset.data.numpy().astype(np.float32)
Y_test = test_dataset.targets.numpy()


# -----------------------------
# Normalize
# -----------------------------
X_train /= 255.0
X_test /= 255.0


# -----------------------------
# Flatten
# -----------------------------
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)


# -----------------------------
# One Hot Labels
# -----------------------------
Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)


# -----------------------------
# Model
# -----------------------------
model = Sequential(
    Linear(784, 128),
    Activation_ReLU(),
    Linear(128, 10),
    Softmax()
)


# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)


# -----------------------------
# Training Settings
# -----------------------------
epochs = 20
batch_size = 64


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(epochs):

    epoch_loss = 0

    for start in range(0, len(X_train), batch_size):

        end = start + batch_size

        x = X_train[start:end]
        y = Y_train[start:end]

        optimizer.zero_grad()

        prediction = model(x)

        loss = criterion(prediction, y)

        grad = criterion.backward()

        model.backward(grad)

        optimizer.step()

        epoch_loss += loss

    print(
        f"Epoch {epoch+1}/{epochs} | Loss : {epoch_loss:.4f}"
    )


# -----------------------------
# Testing
# -----------------------------
prediction = model(X_test)

predicted_labels = np.argmax(prediction, axis=1)
true_labels = np.argmax(Y_test, axis=1)

accuracy = np.mean(predicted_labels == true_labels)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")