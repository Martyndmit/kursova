import pandas as pd
from model import SimpleMLP

def train_model(X_train, y_train, X_test, y_test, experiment, seed=42):
    model = SimpleMLP(hidden_size=experiment["hidden_size"], activation=experiment["activation"], seed=seed)
    lr = experiment["learning_rate"]
    epochs = experiment["epochs"]
    history = []
    for epoch in range(1, epochs + 1):
        y_hat, cache = model.forward(X_train)
        train_loss = model.binary_cross_entropy(y_train, y_hat)
        train_acc = model.accuracy(y_train, y_hat)
        grads, grad_norm = model.backward(cache, y_train)
        update_norm = model.step(grads, lr)
        test_hat = model.predict_proba(X_test)
        test_loss = model.binary_cross_entropy(y_test, test_hat)
        test_acc = model.accuracy(y_test, test_hat)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append({
                "experiment": experiment["name"], "epoch": epoch, "learning_rate": lr,
                "hidden_size": experiment["hidden_size"], "activation": experiment["activation"],
                "train_loss": train_loss, "test_loss": test_loss,
                "train_accuracy": train_acc, "test_accuracy": test_acc,
                "gradient_norm": grad_norm, "update_norm": update_norm
            })
    return model, pd.DataFrame(history)
