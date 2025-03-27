In CS, Genetic Algorithms are metaheuristic inspired by the process of natural selection that belongs to the larger class of Evolutionary Algorithms. They are used in optimizations and search problems to generate high quality solutions, through process of Natural Selection like mutation, crossover and selection.

![[St_5-xband-antenna.jpg]]
A 2006 NASA ST5 evolved antenna, discovered from evolutionary algorithm to create best radiation pattern.
## How it works

In genetic algorithms, a population of **==phenotypes==** (called individuals), are used to solve a optimization problem towards a better solution. Each phenotype has a set of properties called **==genotype==** (also chromosomes), which can be mutated and altered.

The evolution usually starts from a randomly generated population and is an iterative process, and the population in each iteration is called a **==generation==**. In each generation, the **==fitness==** of every individual in evaluated in the population. The fitness is usually the value of the objective function in the optimization problem being solved. Fit Individuals are stochastically selected from the current population, and each individual's genome is modified (possibly recombined or randomly mutated) to form a new generation.

The new generation of Individual solutions is then used in the next iteration of the algorithm.
The Algorithm terminates when either a maximum number of generations has been produced, or a satisfactory fitness level has been reached for the population.

A typical genetic algorithm requires:

1. a [genetic representation](https://en.wikipedia.org/wiki/Genetic_representation "Genetic representation") of the solution domain,
2. a [fitness function](https://en.wikipedia.org/wiki/Fitness_function "Fitness function") to evaluate the solution domain.