import random
import matplotlib.pyplot as plt
import numpy as np

class Chromosome:
    def __init__(self, genes=None):
        if genes is None:
            genes = self.generate_genes()
        self.genes = genes
        self.fitness = 0

    def calculate_fitness(self, obstacle_map, goal_position, start_position):
        x, y = start_position

        path_x, path_y = [x], [y]  # Store the path for plotting
        for move in self.genes:
            temp_x, temp_y = x, y
            if move == "UP" and y < len(obstacle_map) - 1:
                temp_y += 1
            elif move == "DOWN" and y > 0:
                temp_y -= 1
            elif move == "LEFT" and x > 0:
                temp_x -= 1
            elif move == "RIGHT" and x < len(obstacle_map[0]) - 1:
                temp_x += 1

            if self.is_valid_move(obstacle_map, temp_x, temp_y):
                x, y = temp_x, temp_y

            path_x.append(x)
            path_y.append(y)

        distance_to_goal = ((x - goal_position[0]) ** 2 + (y - goal_position[1]) ** 2) ** 0.5 # Euclidean Distanc
        
        self.fitness = 1 / (1 + distance_to_goal)

        return path_x, path_y, x, y

    def is_valid_move(self, obstacle_map, x, y):
        return obstacle_map[y][x] == 0

    def generate_genes(self, length=10):
        return [random.choice(["UP", "DOWN", "LEFT", "RIGHT"]) for g in range(length)]

class Population:
    def __init__(self, size):
        self.size = size
        self.chromosomes = [Chromosome() for c in range(size)]

    def evolve(self, elite_percent=0.1, crossover_prob=0.8, mutation_prob=0.1):
        for chromosome in self.chromosomes:
            chromosome.calculate_fitness(obstacle_map, goal_position, start_position)

        elite_count = int(elite_percent * self.size)
        elite = sorted(self.chromosomes, key=lambda x: x.fitness, reverse=True)[:elite_count]

        # Crossover 
        offspring = []
        while len(offspring) < self.size - elite_count:
            parent1, parent2 = random.sample(elite, 2)
            if random.random() < crossover_prob:
                crossover_point = random.randint(0, len(parent1.genes) - 1)
                child_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
            else:
                child_genes = parent1.genes[:]
            offspring.append(Chromosome(child_genes))

        # Mutation
        for chromosome in offspring:
            if random.random() < mutation_prob:
                mutation_point = random.randint(0, len(chromosome.genes) - 1)
                chromosome.genes[mutation_point] = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

        self.chromosomes = elite + offspring

# Example usage:
if __name__ == "__main__":

    color_mapping = {
    0: 0,        # Empty cell
    "RED": 1,
    "BLUE": 2,
    "GREEN": 3,
    "YELLOW": 4,
    "ORANGE": 5,
    "CYAN": 6}

    obstacle_map = [
        [0, 0, 'ORANGE', 0, 0, 0],
        ['GREEN', 0, 'RED', 0, 0, 0],
        [0, 0, 0, 'CYAN', 0, 'GREEN'],
        [0, 'BLUE', 0, 0, 0, 'RED'],
        [0, 0, 'ORANGE', 0, 'YELLOW', 0],
        ['CYAN', 0, 0, 0, 0, 0]
    ]


    numerical_map = np.array([[color_mapping[cell] for cell in row] for row in obstacle_map])

    goal_position  = (5, 5)   # Correct the goal position  try (5, 4) - (1, 4) - (5, 1)
    start_position = (0, 0)   # Set the starting position  try (3, 1) - (3, 1) - (0, 0)

    population_size = 8000
    generations = 15

    population = Population(population_size)

    fitness_over_generations = []  

    for generation in range(generations):
        print(f"Generation {generation + 1}")
        population.evolve()
        
        average_fitness = sum(chromosome.fitness for chromosome in population.chromosomes) / population_size
        fitness_over_generations.append(average_fitness)

    best_chromosome = max(population.chromosomes, key=lambda x: x.fitness)

    path_x, path_y, final_x, final_y = best_chromosome.calculate_fitness(obstacle_map, goal_position, start_position)
    
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, generations + 1), fitness_over_generations, marker='o')
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')

    plt.subplot(1, 2, 2)
    plt.imshow(numerical_map, cmap="GnBu", origin="upper", vmin=0, vmax=len(color_mapping) - 1)
    plt.plot(path_x, path_y, marker=".", color="red", label="Path")
    plt.scatter(start_position[0], start_position[1], marker="x", color="brown", label="Start Point")
    plt.scatter(goal_position[0], goal_position[1], marker="x", color="green", label="End Point")
    plt.title("Robot's Best Path")
    plt.legend()

    plt.tight_layout()
    plt.show()