# evolutionary-reinforcement-learner
Evolutionary algorithm for reinforcement learning

# How to Run

To run:

```
docker build .
docker run -it <name_of_container> /bin/bash
```

To run tests from the root directory:

```
python -m pytest tests
```

To run examples from the root directory:

```
python examples.py
```

# Folder Structure

### environments

Contains wrappers for Pygame Learning Environment (PLE) and OpenAI Gym environments.  The wrappers standardize the way the environments work.

### models

Contains a custom multi-layer perceptron (MLP) and a wrapper for models created with Tensorflow.  The Tensorflow wrapper uses Tensorflow's model architecture as a shell of sorts.  Models created with Tensorflow are not optimized with TF, but with the optimizers defined in this module instead.

### Optimizers

Contains optimization algorithms:
* Evolutionary Strategy (ES)
* Covariance Matrix Adaptive Evolutionary Strategy (CMAES)
