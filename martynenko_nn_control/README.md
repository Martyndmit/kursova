# Neural Backpropagation Cybernetic Control Model

Python implementation for a coursework project on cybernetic principles of control in neural networks and backpropagation analysis.

## Idea

The project implements a small neural network from scratch using NumPy. The network is trained on a synthetic binary classification dataset. The code explicitly demonstrates the cybernetic control loop:

1. forward propagation produces a prediction;
2. the loss function calculates the error;
3. backpropagation transfers the error signal backward;
4. gradients define the control action;
5. weights are updated;
6. a new iteration begins with lower error.

No TensorFlow or PyTorch is used, so all core mechanisms are visible in the source code.

## Experiments

The project compares several training configurations:

- too small learning rate;
- balanced learning rate;
- too large learning rate;
- small hidden layer;
- larger hidden layer;
- ReLU activation configuration.

## Outputs

After running `python main.py`, the `results/` directory contains:

- `dataset.csv`;
- `training_history.csv`;
- `summary_results.csv`;
- loss and accuracy curves;
- gradient norm and weight update plots;
- decision boundary plots;
- neural network architecture diagram;
- cybernetic feedback loop diagram.

## Run

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Author

Dmytro Martynenko — coursework practical implementation.
