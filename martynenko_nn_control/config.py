RANDOM_SEED = 42
N_SAMPLES = 700
TEST_RATIO = 0.25
NOISE = 0.15

EXPERIMENTS = [
    {"name": "lr_too_small", "learning_rate": 0.005, "hidden_size": 10, "activation": "tanh", "epochs": 800, "label": "Занадто малий крок"},
    {"name": "lr_balanced", "learning_rate": 0.08, "hidden_size": 10, "activation": "tanh", "epochs": 800, "label": "Збалансований крок"},
    {"name": "lr_too_large", "learning_rate": 50.0, "hidden_size": 10, "activation": "tanh", "epochs": 800, "label": "Надмірний крок"},
    {"name": "small_network", "learning_rate": 0.08, "hidden_size": 3, "activation": "tanh", "epochs": 800, "label": "Мала мережа"},
    {"name": "larger_network", "learning_rate": 0.08, "hidden_size": 18, "activation": "tanh", "epochs": 800, "label": "Більша мережа"},
    {"name": "relu_balanced", "learning_rate": 0.04, "hidden_size": 10, "activation": "relu", "epochs": 800, "label": "ReLU-конфігурація"},
]
