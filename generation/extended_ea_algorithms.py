import random
from operator import attrgetter
from typing import Optional

from deap.algorithms import tools
from deap.tools.support import deepcopy
from tqdm import tqdm

from constants import MAX_LENGTH
from generation.utils import clone
from utils_classes.Split import Split


def split_population(population: list, split: Optional[Split]):
    if split is None:
        return
    for ind in population:
        parsed_split = split.parse_nan(ind)
        start, middle, end = parsed_split.split
        assert isinstance(start, int)
        assert isinstance(middle, int)
        assert isinstance(end, int)

        # Modify the individual in place by trimming it to the desired range
        # print(f"[DEBUG] start and middle are: ", start, middle)
        middle_clone = ind[start : start + middle].copy()
        assert (
            len(middle_clone) > 0
        ), f"Wrong middle clone: {middle_clone} for seq: {ind} of length: {len(ind)}. {start, middle, end}"
        del ind[:start]
        del ind[middle:]
        assert ind == middle_clone, f"Wrong split: {ind} != {middle_clone}"
        assert len(ind) == middle, f"Wrong split: {len(ind)} != {middle}"


def reconstruct_population(
    population: list, og_population: list, split: Optional[Split]
) -> None:
    if split is None:
        return

    assert og_population is not None

    og_population = clone(og_population)
    for ind, og_ind in zip(population, og_population):
        parsed_split = split.parse_nan(og_ind)
        start, middle, _ = parsed_split.split

        assert isinstance(start, int)
        assert isinstance(middle, int)
        og_ind[start : start + len(ind)] = ind
    population[:] = og_population


# Taken from deap.algorithms.eaSimple
# Solution taken from https://github.com/DEAP/deap/issues/508
def eaSimpleBatched(
    population,
    toolbox,
    cxpb: float,
    mutpb: float,
    ngen: int,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    pbar=True,
    split: Optional[Split] = None,
):
    """
    This extends the deap.eaSimple method in order for the sequences in the
    population to be evaluated in batch, instead of one-by-one. This is useful
    in order to improve the speed when the evaluation involved a deep learning
    model
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    assert all(
        len(ind) <= MAX_LENGTH for ind in population
    ), f"There are individuals with length > max: {[len(ind) for ind in population if len(ind) > MAX_LENGTH ]}"
    fitnesses = toolbox.evaluate(invalid_ind)
    for i, ind in enumerate(invalid_ind):
        ind.fitness.values = fitnesses[i]

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Split each individual to force the edits only on a part of the sequence
    og_population = None if split is None else deepcopy(population)
    # Begin the generational process
    for gen in tqdm(
        range(1, ngen + 1),
        "Running generation algorithm...",
        disable=not pbar,
        leave=False,
    ):
        # print(f"Length before split: {[len(ind) for ind in population]}")

        assert all(
            len(ind) <= MAX_LENGTH for ind in population
        ), f"At gen {gen} there are individuals with length > max: {[len(ind) for ind in population if len(ind) > MAX_LENGTH ]}"

        split_population(population, split)

        # assert all(
        #     len(ind) > 0 for ind in population
        # ), f"[DEBUG gen: {gen}] AFTER POP SPLIT: population must not contains empty sequences!"

        # print(f"Length after split: {[len(ind) for ind in population]}")
        # print(f"Ind len:", [len(ind) for ind in population])
        # Select and vary the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = customVarAnd(offspring, toolbox, cxpb, mutpb)

        # TODO: [DEBUG], remove
        # assert all(
        #     len(offspring[i]) <= MAX_LENGTH - og_lengths[i]
        #     for i in range(len(offspring))
        # ), f"{[(len(offspring[i]), og_lengths[i]) for i in range(len(offspring)) if not len(offspring[i]) <= MAX_LENGTH - og_lengths[i]]}"

        reconstruct_population(offspring, og_population, split)  # type: ignore

        # TODO: [DEBUG], remove
        # assert all(
        #     MIN_LENGTH <= len(ind) <= MAX_LENGTH for ind in offspring
        # ), f"{[len(ind) for ind in offspring if not MIN_LENGTH <= len(ind) <= MAX_LENGTH ]}"
        # print(f"off len:", [len(ind) for ind in offspring])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.evaluate(invalid_ind)
        for i, ind in enumerate(invalid_ind):
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


def customVarAnd(population: list, toolbox, cxpb: float, mutpb: float):
    """
    Extends the `deap.algorithms.varAnd` method in order to also inject the
    offspring index into the mutation and crossover functions, which will be
    used to generate a random seed. This makes sure that for the same sequence
    at the same index, the same mutation will always be applied, but avoids to
    apply the same mutation to the same sequence at all indices, which will
    result in all the sequences in the population to always be equal.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    # Apply crossover and mutation on the offspring
    # assert all(
    #     len(ind) > 0 for ind in offspring
    # ), f"[DEBUG] BEFORE CROSSOVER: population must not contains empty sequences!"

    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            # print(
            #     f"[DEBUG] Individuals chosen for crossover are: {offspring[i-1]} and {offspring[i]}"
            # )
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1], offspring[i]
            )
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    # assert all(
    #     len(ind) > 0 for ind in offspring
    # ), f"[DEBUG] BETWEEN CROSSOVER and MUTATIONS: population must not contains empty sequences!"

    for i in range(len(offspring)):
        if random.random() < mutpb:
            # print(f"[DEBUG] i: {i}")
            # print(f"[DEBUG] ind: {offspring[i]}")
            (offspring[i],) = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    # assert all(
    #     len(ind) > 0 for ind in offspring
    # ), f"[DEBUG] AFTER MUTATIONS: population must not contains empty sequences!"

    return offspring


def customSelTournament(individuals, k, tournsize, fit_attr="fitness"):
    chosen = []

    all_choices = random.choices(individuals, k=k * tournsize)
    for i in range(k):
        aspirants = all_choices[i * tournsize : (i + 1) * tournsize]
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))

    return chosen


def customCxTwoPoint(ind1, ind2, return_indices: bool = False):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1) if size > 2 else 0
    if cxpoint2 >= cxpoint1 and not cxpoint1 == cxpoint2 == 0:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # print(f"[DEBUG] ind1, ind2 BEFORE CROSSOVER: {ind1, ind2}")
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
        ind2[cxpoint1:cxpoint2],
        ind1[cxpoint1:cxpoint2],
    )
    # print(f"[DEBUG] ind1, ind2 AFTER CROSSOVER: {ind1, ind2}")

    if return_indices:
        return (ind1, ind2), (cxpoint1, cxpoint2)
    return ind1, ind2
