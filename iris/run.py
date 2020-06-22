"""
2-input XOR example -- this is most likely the simplest possible example.
"""

# 2-input XOR inputs and expected outputs.
from __future__ import print_function
import os
import neat
import visualize
import pickle
import myUtile
import matplotlib.pyplot as plt


with open("train_feature_pickle", 'rb') as f:
    train_feature = pickle.load(f)

with open("train_target_pickle", 'rb') as f:
    train_target = pickle.load(f)

with open("test_feature_pickle", 'rb') as f:
    test_feature = pickle.load(f)

with open("test_target_pickle", 'rb') as f:
    test_target = pickle.load(f)




result = {}

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        difference = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(train_feature, train_target):
            output = net.activate(xi)
            difference += (output[0] - xo[0]) ** 2
            difference += (output[1] - xo[1]) ** 2
            difference += (output[2] - xo[2]) ** 2

        connections = []
        for key in genome.connections:
            if (genome.connections[key].enabled):
                connections.append(key)
        nodes_not_input = list(genome.nodes)
        nodes_input = config.genome_config.input_keys
        nodes_output = config.genome_config.output_keys
        ave_deep = myUtile.average_deep(connections, nodes_input, nodes_output)
        ave_clustering = myUtile.average_clustering(connections, nodes_not_input + nodes_input)

        punish = CONNECT_RATE * myUtile.sigmoid(len(connections))
        punish += DEEP_RATE * ave_deep
        punish += CLUSTERING_RATE * ave_clustering

        genome.fitness = 1 - difference / (3 * len(train_target)) - punish


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    X = []
    Y = []
    for i in range(300):
        winner = p.run(eval_genomes, 1)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        predict = []
        for xi in test_feature:
            pre = winner_net.activate(xi)
            predict.append(pre)

        X.append(i)
        Y.append(myUtile.evaluate(predict, test_target))

    # plt.figure(figsize=(8, 4))
    # plt.plot(X, Y, "b--", linewidth=1)
    # plt.xlabel("iterations")
    # plt.ylabel("accuracy")
    name = str(CONNECT_RATE) + '_' + str(DEEP_RATE) + '_' + str(CLUSTERING_RATE) + '_' + str(RUN_NUMBER)
    # plt.title(name)
    # plt.savefig(name + ".jpg")
    # print(name + '    ' + str(Y[-1]))
    result[name] = [X, Y, format(winner)]

    #visualize.draw_net(config, winner, False, name)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    CONNECT_RATE = 0.0
    DEEP_RATE = 0.0
    CLUSTERING_RATE = 0.0

    RUN_NUMBER = 1

    for RUN_NUMBER in range(30):
        run(config_path)

    CONNECT_RATE = 0.05
    for RUN_NUMBER in range(30):
        run(config_path)
    CONNECT_RATE = 0.0

    with open("result_pickle", 'wb') as f:
        pickle.dump(result, f)

    DEEP_RATE = 0.15
    for RUN_NUMBER in range(30):
        run(config_path)
    DEEP_RATE = 0.0

    CLUSTERING_RATE = 0.15
    for RUN_NUMBER in range(30):
        run(config_path)


    with open("result_pickle", 'wb') as f:
        pickle.dump(result, f)