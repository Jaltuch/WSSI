import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

    def _f(self, x):
        return max(x * .1, x)

    def __call__(self, xs):
        return self._f(xs @ self.ws + self.b)


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = [Neuron(layer_sizes[i - 1]) for _ in range(layer_sizes[i])]
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = np.array([neuron(x) for neuron in layer])
        return x

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

        plt.title("Wizualizacja Struktury Sieci Neuronowej")
        plt.show()



network = NeuralNetwork([3, 4, 4, 1])


network.visualize()
