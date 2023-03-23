import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm


def initialisation(dim):

    params = {}
    C = len(dim)

    np.random.seed(1)

    for c in range(1, C):
        params['W' +
                   str(c)] = np.random.randn(dim[c], dim[c - 1])
        params['b' + str(c)] = np.random.randn(dim[c], 1)

    return params


def forward_propagation(X, params):

    activations = {'A0': X}

    C = len(params) // 2

    for c in range(1, C + 1):

        Z = params['W' + str(c)].dot(activations['A' +
                                                     str(c - 1)]) + params['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    return activations


def back_propagation(y, params, activations):

    m = y.shape[1]
    C = len(params) // 2

    dZ = activations['A' + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1/m * np.dot(dZ,
                                                activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(params['W' + str(c)].T, dZ) * activations['A' +
                                                                      str(c - 1)] * (1 - activations['A' + str(c - 1)])

    return gradients


def update(gradients, params, learning_rate):

    C = len(params) // 2

    for c in range(1, C + 1):
        params['W' + str(c)] = params['W' + str(c)] - \
            learning_rate * gradients['dW' + str(c)]
        params['b' + str(c)] = params['b' + str(c)] - \
            learning_rate * gradients['db' + str(c)]

    return params


def predict(X, params):
    activations = forward_propagation(X, params)
    C = len(params) // 2
    Af = activations['A' + str(C)]
    return Af >= 0.5


def deep_neural_network(X, y, hidden_layers=(16, 16, 16), learning_rate=0.001, n_iter=3000):

    # initialisation params
    dim = list(hidden_layers)
    dim.insert(0, X.shape[0])
    dim.append(y.shape[0])
    np.random.seed(1)
    params = initialisation(dim)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_iter), 2))

    C = len(params) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X, params)
        gradients = back_propagation(y, params, activations)
        params = update(gradients, params, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
        y_pred = predict(X, params)
        training_history[i, 1] = (
            accuracy_score(y.flatten(), y_pred.flatten()))

    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='Train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='Train acc')
    plt.legend()
    plt.show()

    return training_history

#### TEST ######

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))

print('Dim de X:', X.shape)
print('Dim de y:', y.shape)

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
plt.show()

# appel fonction
deep_neural_network(X, y, hidden_layers=(16, 16, 16),
                    learning_rate=0.1, n_iter=3000)
