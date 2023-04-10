import csv
import numpy as np
from settings import STARTTIME, HEADER1, HEADER2, DNA_SIZE


def isfloat(value):
    """ returns true if value can be a float """
    try:
        float(value)
        return True
    except ValueError:
        return False


def column(matrix, i):
    """ returns the column of index i removing headers """
    return [row[i] for row in matrix if isfloat(row[i])]


def print_info(c, timestamp):
    """ print creature info on console """
    print(f"\n[{timestamp}] [{id(c)}] [Fitness: {c.fitness()}]\n " +
          f"Age: {c.age} seconds, F.Eaten: {c.food_eaten}, P.Eaten: {c.poison_eaten}\n" +
          f"currHP: {c.health}, Gen: {c.gen}, Childs: {c.childs}\n" +
          f"DNA: {c.dna}\n" +
          f"FoodAttr: {c.food_attraction}, PoisonAttr: {c.poison_attraction}\n" +
          f"FoodDist: {c.food_dist}, PoisonDist: {c.poison_dist}\n" +
          f"MaxHealth: {c.max_health}, MaxVel: {c.max_vel}, Size: {c.size}\n" +
          f"MaxSteer: {c.max_steer_force}, DirAngleMult: {c.dir_angle_mult}\n")


class Datastats:
    """ stores statistics, history and other data from the game """

    def __init__(self):
        self.fittest = None
        self.current_fittest = None
        self.oldest = None
        self.fitness_record = 0
        self.oldest_age = 0

        # stores creatures and its fitness value from 0 to 1 compared
        # to the other creatures of the same generation, used in ByGen mode
        self.temp_hist_by_gen = {}

        self.temp_history = []
        self.history = ([])  # will be np.array
        self.temp_stats_history = []
        self.stats_history = ([])  # will be np.array
        self.last_save = 0
        self.header_saved = [False, False]

        self.csv_name1 = STARTTIME + "_history.csv"
        self.csv_name2 = STARTTIME + "_stats.csv"

        # see HEADER1 & 2 for order (excluding time)
        self.means = [0 for _ in range(len(HEADER1) - 1)]
        self.medians = [0 for _ in range(len(HEADER1) - 1)]

    def append_to_hist(self, c, timestamp):
        row = [timestamp, c.fitness(), c.age, c.gen, c.childs,
               c.food_eaten, c.poison_eaten]

        for i in range(DNA_SIZE):
            row.append(c.dna[i])
        self.temp_history.append(row)
        # using numpy for history
        row = np.array(row)
        # if history is still a list, it's empty
        if isinstance(self.history, list):
            self.history = row
        else:
            self.history = np.vstack((self.history, row))

    def save_csv(self):
        with open(self.csv_name1, mode='a', newline='') as data_file:
            data_writer = csv.writer(
                data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # check if we saved the header (not saved = new file)
            if not self.header_saved[0]:
                data_writer.writerow(HEADER1)
                self.header_saved[0] = True
            for line in self.temp_history:
                data_writer.writerow(line)
        self.temp_history.clear()
        with open(self.csv_name2, mode='a', newline='') as data_file:
            data_writer = csv.writer(
                data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # check if we saved the header (not saved = new file)
            if not self.header_saved[1]:
                data_writer.writerow(HEADER2)
                self.header_saved[1] = True
            for line in self.temp_stats_history:
                data_writer.writerow(line)
        self.temp_stats_history.clear()

    def calc_fitness_by_gen(self):
        """ gives each creatures in the dict a chance being it higher
        the higher fitness that creatures has """
        f_sum = 0
        # first loop gives us the sum of the fitness
        for c, _ in self.temp_hist_by_gen.items():
            f_sum += c.fitness()
        # now we calc the chances by fitness of each one
        for c, _ in self.temp_hist_by_gen.items():
            self.temp_hist_by_gen[c] = c.fitness() / f_sum

    def calc_stats(self, timestamp):
        # we have data if history is already an np.array
        if isinstance(self.history, np.ndarray):
            row = []
            row.append(timestamp)
            for i in range(len(self.means)):

                # Slower:
                # self.means[i] = np.mean(column(self.history, i+1))
                # self.medians[i] = np.median(column(self.history, i+1))

                self.means[i] = np.mean(self.history[:, i+1])
                self.medians[i] = np.median(self.history[:, i+1])
                row.append(self.means[i])
                row.append(self.medians[i])

            self.temp_stats_history.append(row)
            row = np.array(row)
            if isinstance(self.stats_history, np.ndarray):
                np.vstack((self.stats_history, row))
            else:
                self.stats_history = row

        print("~~~~~~~~~~")
        print(f"Mean Fitness:\t{self.means[0]}")
        print(f"Median Fitness:\t{self.medians[0]}")
        print("~~~~~~~~~~")

    def print_stats(self):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for i in range(len(HEADER1)):
            if i == 0:
                # skip Time
                continue
            print(f"Mean {HEADER1[i]}:\t{self.means[i-1]}")
            print(f"Median {HEADER1[i]}:\t{self.medians[i-1]}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
