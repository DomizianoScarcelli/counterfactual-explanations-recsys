import random
from operator import attrgetter

from deap.algorithms import tools
from deap.tools import selRandom
from tqdm import tqdm

from config import ConfigParams
from utils import set_seed


# Taken from deap.algorithms.eaSimple
# Solution taken from https://github.com/DEAP/deap/issues/508
def eaSimpleBatched(population, toolbox, cxpb, mutpb, ngen, stats=None,
                    halloffame=None, verbose=__debug__, pbar=True):
    """
    This extends the deap.eaSimple method in order for the sequences in the
    population to be evaluated in batch, instead of one-by-one. This is useful
    in order to improve the speed when the evaluation involved a deep learning
    model
    """
    set_seed()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.evaluate(invalid_ind)
    for i,ind in enumerate(invalid_ind):
          ind.fitness.values = fitnesses[i]

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in tqdm(range(1, ngen + 1), "Running generation algorithm...", disable=not pbar, leave=False):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = indexedVarAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.evaluate(invalid_ind)
        for i,ind in enumerate(invalid_ind):
              ind.fitness.values = fitnesses[i]


        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def indexedVarAnd(population, toolbox, cxpb, mutpb):
    """
    Extends the `deap.algorithms.varAnd` method in order to also inject the
    offspring index into the mutation and crossover functions, which will be
    used to generate a random seed. This makes sure that for the same sequence
    at the same index, the same mutation will always be applied, but avoids to
    apply the same mutation to the same sequence at all indices, which will
    result in all the sequences in the population to always be equal.
    """
    set_seed()
    offspring = [toolbox.clone(ind) for ind in population]
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i], i)
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i], i)
            del offspring[i].fitness.values

    return offspring

def indexedSelTournament(individuals, k, tournsize, fit_attr="fitness"):
    chosen = []
    if ConfigParams.DETERMINISM:
        hash_key = hash(tuple([ind.fitness.values for ind in individuals])) ^ hash(tuple(hash(tuple(ind)) for ind in individuals))
        curr_seed = hash_key
    for i in range(k):
        if ConfigParams.DETERMINISM:
            set_seed(hash(curr_seed)) #type: ignore
        aspirants = [random.choice(individuals) for _ in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        # set the seed based on individuals, index and current chosen ones
        if ConfigParams.DETERMINISM:
            curr_seed ^= i ^ hash(tuple(chosen[-1])) #type: ignore

    return chosen


def indexedCxTwoPoint(ind1, ind2, index, return_indices: bool=False):
    # print(f"[DEBUG] len ind1", len(ind1))
    # print(f"[DEBUG] len ind2", len(ind2))
    if ConfigParams.DETERMINISM:
        set_seed(dependencies=[ind1, ind2, index])
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    
    set_seed()
    if return_indices:
        return (ind1, ind2), (cxpoint1, cxpoint2)
    return ind1, ind2
