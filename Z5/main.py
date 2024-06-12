import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Neuron:
    def __init__(self, n_inputs, activation='leaky_relu', bias=0., weights=None):
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

        if activation == 'leaky_relu':
            self.activation = self._leaky_relu
            self.activation_prime = self._leaky_relu_prime
        else:
            raise ValueError(f"Unknown activation function '{activation}'")

    def _leaky_relu(self, x):
        return np.maximum(x * 0.1, x)

    def _leaky_relu_prime(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = 0.1
        return dx

    def __call__(self, xs):
        return self.activation(np.dot(xs, self.ws) + self.b)

    def update_weights(self, delta, lr):
        self.ws -= lr * delta


class NeuralNetwork:
    def __init__(self, layer_sizes, activation='leaky_relu'):
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = [Neuron(layer_sizes[i - 1], activation) for _ in range(layer_sizes[i])]
            self.layers.append(layer)

    def _forward(self, x):
        for layer in self.layers:
            x = np.array([neuron(x) for neuron in layer])
        return x

    def _backward(self, x, y, lr):
        outputs = self._forward(x)


        deltas = []
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            if layer_idx == len(self.layers) - 1:  # Ostatnia warstwa
                delta = (outputs - y) * layer[0].activation_prime(outputs)
            else:
                next_layer = self.layers[layer_idx + 1]
                delta = np.dot(deltas[-1], [neuron.ws for neuron in next_layer])
                delta *= layer[0].activation_prime(outputs)

            deltas.append(delta)

        deltas.reverse()


        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            input_to_use = x if layer_idx == 0 else self._forward(
                x)  # Wejście dla pierwszej warstwy, wyjście dla pozostałych
            for neuron, d in zip(layer, deltas[layer_idx]):
                neuron.update_weights(input_to_use, lr * d)

    def train(self, X, y, epochs=100, lr=0.01):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                self._backward(xi, yi, lr)

    def visualize(self):
        G = nx.DiGraph()
        positions = {}


        input_layer_size = self.layer_sizes[0]
        input_positions = [(0, i - input_layer_size / 2) for i in range(input_layer_size)]
        for i, pos in enumerate(input_positions):
            positions[f'Input_{i}'] = pos
            G.add_node(f'Input_{i}')


        layer_positions = [1]
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer):
                pos = (layer_positions[-1], neuron_index - len(layer) / 2)
                node_label = f'L{layer_index + 1}_N{neuron_index + 1}'
                positions[node_label] = pos
                G.add_node(node_label)


                if layer_index == 0:
                    for input_index in range(input_layer_size):
                        G.add_edge(f'Input_{input_index}', node_label)
                else:
                    prev_layer_size = self.layer_sizes[layer_index]
                    for prev_neuron_index in range(prev_layer_size):
                        prev_node_label = f'L{layer_index}_N{prev_neuron_index + 1}'
                        G.add_edge(prev_node_label, node_label)

            layer_positions.append(layer_positions[-1] + 1)

        fig, ax = plt.subplots(figsize=(12, 6))
        nx.draw(G, pos=positions, with_labels=False, arrows=True, node_size=1000, node_color='lightblue', ax=ax)

        plt.title("")
        plt.show()



X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])


network = NeuralNetwork([2, 2, 1])


network.train(X, y, epochs=1000, lr=0.1)


network.visualize()
