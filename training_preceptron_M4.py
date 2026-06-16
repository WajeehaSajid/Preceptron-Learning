# ============================================================
# AL-2002 PROJECT - MODULE 4: Learning in AI
# Roll No: 24F-0806
# Description: This program implements two learning algorithms:
#   1. Perceptron Learning Rule
#   2. Gradient Descent Delta Rule
# Both are used to classify Iris flower species.
# Dataset: UCI Iris Dataset (150 samples, 3 classes)
# We do binary classification: Setosa vs Non-Setosa
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# STEP 1: Load and Prepare Dataset
# ─────────────────────────────────────────────

# Load iris dataset (built-in sklearn)
iris = load_iris()
X = iris.data        # 4 features: sepal length, sepal width, petal length, petal width
y = iris.target      # 0=Setosa, 1=Versicolor, 2=Virginica

# Binary classification: Setosa(0) vs Not-Setosa(1)
# Perceptron works best with binary labels
y_binary = np.where(y == 0, 1, -1)   # Setosa = +1, Others = -1

# Normalize features (very important for gradient descent)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
print(f"Features:         {X_train.shape[1]}")


# ─────────────────────────────────────────────
# STEP 2: Activation Functions
# ─────────────────────────────────────────────

def step_function(net_input):
    """
    Step activation function.
    Returns +1 if input >= 0, else -1.
    Used in Perceptron.
    """
    return np.where(net_input >= 0, 1, -1)


def sigmoid_function(net_input):
    """
    Sigmoid activation function.
    Returns value between 0 and 1.
    Used in Gradient Descent.
    """
    return 1 / (1 + np.exp(-net_input))


def sigmoid_derivative(net_input):
    """
    Derivative of sigmoid — needed for gradient descent weight update.
    """
    sig = sigmoid_function(net_input)
    return sig * (1 - sig)


def linear_function(net_input):
    """
    Linear activation function.
    Just returns the input as-is.
    Used in Gradient Descent (Delta Rule uses linear output for error).
    """
    return net_input


# ─────────────────────────────────────────────
# STEP 3: Perceptron Learning Rule
# ─────────────────────────────────────────────

class Perceptron:
    """
    Perceptron classifier using the Perceptron Learning Rule.
    Update rule: w = w + learning_rate * (y - y_hat) * x
    Only updates weights when prediction is WRONG.
    Uses step function as activation.
    """

    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []   # track how many mistakes per epoch

    def fit(self, X, y):
        """Train the perceptron on training data."""
        num_samples, num_features = X.shape

        # Initialize weights and bias to zero
        self.weights = np.zeros(num_features)
        self.bias = 0

        for epoch in range(self.epochs):
            total_errors = 0

            for i in range(num_samples):
                # Calculate net input (weighted sum)
                net_input = np.dot(X[i], self.weights) + self.bias

                # Apply step activation
                prediction = step_function(net_input)

                # Update weights ONLY if prediction is wrong
                if prediction != y[i]:
                    update = self.learning_rate * y[i]
                    self.weights += update * X[i]
                    self.bias    += update
                    total_errors += 1

            self.errors_per_epoch.append(total_errors)

            # Stop early if no errors
            if total_errors == 0:
                print(f"Perceptron converged at epoch {epoch + 1}")
                break

    def predict(self, X):
        """Predict class labels for input X."""
        net_input = np.dot(X, self.weights) + self.bias
        return step_function(net_input)

    def accuracy(self, X, y):
        """Calculate accuracy percentage."""
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return (correct / len(y)) * 100


# ─────────────────────────────────────────────
# STEP 4: Gradient Descent Delta Rule
# ─────────────────────────────────────────────

class GradientDescentDeltaRule:
    """
    Gradient Descent classifier using the Delta (Widrow-Hoff) Rule.
    Update rule: w = w + learning_rate * (y - output) * activation_derivative * x
    Updates weights on EVERY sample (not just wrong ones).
    Can use different activation functions.
    """

    def __init__(self, learning_rate=0.01, epochs=100, activation='linear'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_name = activation
        self.weights = None
        self.bias = None
        self.loss_per_epoch = []    # track Mean Squared Error per epoch

    def activate(self, net_input):
        """Apply selected activation function."""
        if self.activation_name == 'sigmoid':
            return sigmoid_function(net_input)
        else:  # linear (default for delta rule)
            return linear_function(net_input)

    def activate_derivative(self, net_input):
        """Derivative of activation function for weight update."""
        if self.activation_name == 'sigmoid':
            return sigmoid_derivative(net_input)
        else:  # linear derivative = 1
            return 1

    def fit(self, X, y):
        """Train using gradient descent delta rule."""
        num_samples, num_features = X.shape

        # Initialize weights and bias to small random values
        self.weights = np.random.randn(num_features) * 0.01
        self.bias = 0

        for epoch in range(self.epochs):
            epoch_loss = 0

            for i in range(num_samples):
                # Calculate net input
                net_input = np.dot(X[i], self.weights) + self.bias

                # Get output with activation
                output = self.activate(net_input)

                # Calculate error
                error = y[i] - output

                # Gradient descent weight update (Delta Rule)
                delta = error * self.activate_derivative(net_input)
                self.weights += self.learning_rate * delta * X[i]
                self.bias    += self.learning_rate * delta

                # Accumulate squared error for loss tracking
                epoch_loss += error ** 2

            # Store average loss
            self.loss_per_epoch.append(epoch_loss / num_samples)

    def predict(self, X):
        """Predict class labels."""
        net_input = np.dot(X, self.weights) + self.bias
        output = self.activate(net_input)

        # Convert output to +1 or -1
        return np.where(output >= 0.5 if self.activation_name == 'sigmoid'
                        else output >= 0, 1, -1)

    def accuracy(self, X, y):
        """Calculate accuracy percentage."""
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return (correct / len(y)) * 100


# ─────────────────────────────────────────────
# STEP 5: Train Both Models
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("   TRAINING PERCEPTRON")
print("="*50)
perceptron = Perceptron(learning_rate=0.1, epochs=100)
perceptron.fit(X_train, y_train)

train_acc_p = perceptron.accuracy(X_train, y_train)
test_acc_p  = perceptron.accuracy(X_test,  y_test)

print(f"Perceptron Train Accuracy: {train_acc_p:.2f}%")
print(f"Perceptron Test  Accuracy: {test_acc_p:.2f}%")


print("\n" + "="*50)
print("   TRAINING GRADIENT DESCENT (Linear)")
print("="*50)
gd_linear = GradientDescentDeltaRule(learning_rate=0.01, epochs=100, activation='linear')
gd_linear.fit(X_train, y_train)

train_acc_gl = gd_linear.accuracy(X_train, y_train)
test_acc_gl  = gd_linear.accuracy(X_test,  y_test)

print(f"GD (Linear) Train Accuracy: {train_acc_gl:.2f}%")
print(f"GD (Linear) Test  Accuracy: {test_acc_gl:.2f}%")


print("\n" + "="*50)
print("   TRAINING GRADIENT DESCENT (Sigmoid)")
print("="*50)
gd_sigmoid = GradientDescentDeltaRule(learning_rate=0.1, epochs=100, activation='sigmoid')
gd_sigmoid.fit(X_train, y_train)

train_acc_gs = gd_sigmoid.accuracy(X_train, y_train)
test_acc_gs  = gd_sigmoid.accuracy(X_test,  y_test)

print(f"GD (Sigmoid) Train Accuracy: {train_acc_gs:.2f}%")
print(f"GD (Sigmoid) Test  Accuracy: {test_acc_gs:.2f}%")


# ─────────────────────────────────────────────
# STEP 6: Compare Results in a Table
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("   FINAL COMPARISON TABLE")
print("="*60)
print(f"{'Model':<30} {'Train Acc':>10} {'Test Acc':>10}")
print("-"*60)
print(f"{'Perceptron (Step)':<30} {train_acc_p:>9.2f}% {test_acc_p:>9.2f}%")
print(f"{'GD Delta (Linear)':<30} {train_acc_gl:>9.2f}% {test_acc_gl:>9.2f}%")
print(f"{'GD Delta (Sigmoid)':<30} {train_acc_gs:>9.2f}% {test_acc_gs:>9.2f}%")
print("="*60)


# ─────────────────────────────────────────────
# STEP 7: Plotting Results
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Module 4 - Learning Algorithms Comparison", fontsize=14, fontweight='bold')

# Plot 1: Perceptron errors per epoch
axes[0].plot(perceptron.errors_per_epoch, color='red', linewidth=2)
axes[0].set_title("Perceptron — Errors per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Number of Misclassifications")
axes[0].grid(True)

# Plot 2: GD Linear loss per epoch
axes[1].plot(gd_linear.loss_per_epoch, color='blue', linewidth=2)
axes[1].set_title("GD Delta (Linear) — Loss per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Mean Squared Error")
axes[1].grid(True)

# Plot 3: GD Sigmoid loss per epoch
axes[2].plot(gd_sigmoid.loss_per_epoch, color='green', linewidth=2)
axes[2].set_title("GD Delta (Sigmoid) — Loss per Epoch")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Mean Squared Error")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("module4_results.png", dpi=150)
plt.show()
print("\nGraph saved as module4_results.png")