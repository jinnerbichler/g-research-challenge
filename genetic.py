import logging
import random
import main as g_research
import tensorflow as tf
from typing import List

logging.basicConfig(
    # format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
)

logger = logging.getLogger('genetic')


def main():
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.

    param_choices = {
        'hidden_units': [[64], [128], [256], [512], [768], [1024], [2048]],
        'learning_rate': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        'shuffle': [True, False],
        'batch_size': [1, 10, 100],
        'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5, 0.6],
        'l2_reg_scale': [1e-3, 1e-4, 1e-5, 1e-7, 1e-9, 1e-10],
        'activation': [tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh, tf.nn.elu],
    }

    logger.info("***Evolving %d generations with population %d***" % (generations, population))

    generate(generations, population, param_choices)


def generate(generations, num_population, param_choices):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        num_population (int): Number of networks in each generation
        param_choices (dict): Parameter choices for networks

    """
    optimizer = Optimizer(param_choices)
    population = optimizer.create_population(num_population)

    # Evolve the generation.
    for i in range(generations):
        logger.info("***Doing generation %d of %d***" % (i + 1, generations))

        # evaluate population
        evaluate_population(population)

        # print out the average accuracy each generation.
        average_accuracy = get_average_fitness(population)
        logger.info("Generation average: %.2f" % average_accuracy)
        logger.info('-' * 80)

        # evolve, except on the last iteration.
        if i != generations - 1:
            population = optimizer.evolve(population)

    # print out the top 5 networks.
    population = sorted(population, key=lambda x: x['fitness'], reverse=True)
    for chromosome in population[:5]:
        logger.info('{}: {}'.format(chromosome['fitness'], chromosome['params']))


def evaluate_population(population):
    for count, chromosome in enumerate(population):
        logger.info('Evaluating network %d of %d' % (count, len(population)))
        parameter = chromosome['params']

        # chromosome['fitness'] = 1 / parameter['learning_rate'] * parameter['batch_size']
        # chromosome['fitness'] *= parameter['l2_reg_scale'] * parameter['dropout_rate']
        # chromosome['fitness'] *= parameter['shuffle'] * 100
        # chromosome['fitness'] *= parameter['hidden_units'][0]

        try:
            error = g_research.evaluate(max_steps=1.2e5, **parameter)
            chromosome['fitness'] = 1.0 / error
        except:
            logger.exception('Error while evaluating {}'.format(parameter))
            chromosome['fitness'] = 0.0


def get_average_fitness(population):
    """Get the average accuracy for a group of networks."""
    total_accuracy = 0
    for chromosome in population:
        total_accuracy += chromosome['fitness']
    return total_accuracy / len(population)


class Optimizer:
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.4):
        """Create an optimizer.

        Args:
            param_choices (dict): Possible network paremeters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.param_choices = param_choices

    def create_population(self, num_population):
        # type: (int) -> List[dict]
        populations = []  # type: List[dict]
        for _ in range(num_population):
            params = {key: random.choice(self.param_choices[key]) for key in self.param_choices}
            chromosome = {'params': params, 'fitness': None}
            populations.append(chromosome)
        return populations

    def breed(self, mother, father):
        children = []
        for _ in range(2):

            # loop through the parameters and pick params for the kid.
            child_params = {}
            for param in self.param_choices:
                child_params[param] = random.choice([mother['params'][param], father['params'][param]])

            # now create a chromosome object.
            new_chromosome = {'params': child_params, 'error': None}

            # randomly mutate some of the children.
            if self.mutate_chance > random.random():
                new_chromosome = self.mutate(new_chromosome)

            children.append(new_chromosome)

        return children

    def mutate(self, chromosome):
        """
        Randomly mutate one part of the network.
        """
        # choose a random key.
        mutation = random.choice(list(self.param_choices.keys()))

        # mutate one of the params.
        chromosome['params'][mutation] = random.choice(self.param_choices[mutation])

        return chromosome

    def evolve(self, population):
        # sort on the scores.
        graded = sorted(population, key=lambda x: x['fitness'], reverse=True)

        # get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # the parents are every network we want to keep.
        parents = graded[:retain_length]

        # for those we aren't keeping, randomly add some anyway.
        for _ in graded[retain_length:]:
            if self.random_select > random.random():
                individual = self.create_population(1)[0]
                parents.append(individual)

        # add children, which are bred from two remaining networks.
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []
        while len(children) < desired_length:

            # get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # breed them.
                babies = self.breed(male, female)

                # add the children one at a time.
                for baby in babies:
                    # don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents


if __name__ == '__main__':
    main()
