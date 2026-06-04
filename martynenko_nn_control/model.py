import numpy as np

def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def tanh_derivative(a):
    return 1 - a**2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(a):
    return (a > 0).astype(float)

class SimpleMLP:
    def __init__(self, input_size=2, hidden_size=10, activation="tanh", seed=42):
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def _activation(self, z):
        if self.activation_name == "relu":
            return relu(z)
        return tanh(z)

    def _activation_derivative(self, a):
        if self.activation_name == "relu":
            return relu_derivative(a)
        return tanh_derivative(a)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._activation(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = sigmoid(z2)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}
        return y_hat, cache

    @staticmethod
    def binary_cross_entropy(y, y_hat):
        eps = 1e-9
        return float(-np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))

    @staticmethod
    def accuracy(y, y_hat):
        return float(np.mean((y_hat >= 0.5) == y))

    def backward(self, cache, y):
        X = cache["X"]
        a1 = cache["a1"]
        y_hat = cache["y_hat"]
        n = X.shape[0]
        dz2 = (y_hat - y) / n
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._activation_derivative(a1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        grad_norm = float(np.sqrt(sum(np.sum(g**2) for g in grads.values())))
        return grads, grad_norm

    def step(self, grads, lr):
        update_norm = float(np.sqrt(np.sum((lr * grads["dW1"])**2) + np.sum((lr * grads["dW2"])**2)))
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]
        return update_norm

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
