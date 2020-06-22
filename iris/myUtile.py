import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def get_enabled_connection(genome):
    L = []
    for key in genome.connections:
        if (genome.connections[key].enabled):
            L.append(key)
    return L

def average_deep(connections, inputs, outputs):
    deep = []
    stack = []
    for i in inputs:
        stack.append((i,1))
    while(stack != []):
        in_node = stack.pop()
        if(in_node[0] in outputs):
            deep.append(in_node[1])
        for connection in connections:
            if(connection[0] == in_node[0]):
                stack.append((connection[1], in_node[1]+1))

    if(deep == []):
        return 0
    return sigmoid(sum(deep)/len(deep))


def average_clustering(connections, nodes):
    coefficients = []
    for node in nodes:
        coefficient = 0
        relate_nodes = []
        for connection in connections:
            if (connection[0] == node):
                relate_nodes.append(connection[1])

        if (relate_nodes == []):
            continue

        for connection in connections:
            if (connection[0] in relate_nodes and connection[1] in relate_nodes):
                coefficient += 1

        Kv = len(relate_nodes)
        if (Kv == 1):
            coefficients.append(0)
        else:
            coefficients.append((2 * coefficient) / (Kv * (Kv - 1)))

    if(connections == []):
        return 0
    return sum(coefficients) / len(coefficients)


def evaluate(predict, test_target):
    correct = 0
    for pre, test in zip(predict, test_target):
        max_num = max(pre)
        index = pre.index(max_num)
        if(test[index] == 1):
            correct += 1

    return correct / len(test_target)


# def eval_genomes(genomes, config):
#     for genome_id, genome in genomes:
#         genome.fitness = 4.0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         for xi, xo in zip(xor_inputs, xor_outputs):
#             output = net.activate(xi)
#             genome.fitness -= (output[0] - xo[0]) ** 2
#
#
#             connections = []
#             for key in genome.connections:
#                 if (genome.connections[key].enabled):
#                     connections.append(key)
#             nodes_not_input = list(genome.nodes)
#             nodes_input = config.genome_config.input_keys
#             nodes_output = config.genome_config.output_keys
#             print(connections, nodes_not_input, nodes_input, nodes_output)
#             ave_deep = myUtile.average_deep(connections, nodes_input, nodes_output)
#             ave_clusting = myUtile.average_clusting(connections, nodes_not_input + nodes_input)
