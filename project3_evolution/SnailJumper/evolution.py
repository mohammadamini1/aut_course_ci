import copy

from player import Player
import random
import numpy as np
import json

class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        jfn = random.randint(10000, 99999)
        self.learning_curve_path = "./learning_curve_{}.json".format(jfn)
        with open(self.learning_curve_path, 'w') as r:
            json.dump({}, r)
            r.close()
            print("learning curve save in: " + str(self.learning_curve_path))

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        players.sort(reverse=True, key=lambda x : x.fitness)

        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)
        # TODO (Additional: Learning curve)
        try:
            fitnesses = [p.fitness for p in players]
            min_fitness = min(fitnesses)
            max_fitness = max(fitnesses)
            avr_fitness = sum(fitnesses) // fitnesses.__len__()

            with open(self.learning_curve_path, 'r') as r:
                f = dict(json.load(r))
                r.close()

            try:
                m = int(max([int(k) for k in f.keys()]))
            except:
                m = 0
            f.update({
                str(m + 1): {
                    "min": min_fitness,
                    "max": max_fitness,
                    "avr": avr_fitness,
                }
            })
            with open(self.learning_curve_path, 'w') as r:
                json.dump(f, r)
                r.close()

            # print("\n")
            # print("num players: " + str(num_players))
            # print("min fitness: " + str(min_fitness))
            # print("max fitness: " + str(max_fitness))
            # print("avr fitness: " + str(avr_fitness))
        except Exception as e:
            print("error: " + str(e))
            pass

        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            random.shuffle(prev_players)
            new_players = []
            for i in range(prev_players.__len__() // 2):
                child1 = self.clone_player(prev_players[i * 2])
                child2 = self.clone_player(prev_players[i * 2 + 1])

                ## crossover
                child1.nn.wh = child1.nn.wh.T
                child1.nn.wo = child1.nn.wo.T
                child2.nn.wh = child2.nn.wh.T
                child2.nn.wo = child2.nn.wo.T
                p = random.randint(1, child1.nn.wh.shape[0])
                tmp = child2.nn.wh[:p].copy()
                child2.nn.wh[:p], child1.nn.wh[:p]  = child1.nn.wh[:p], tmp

                p = random.randint(1, child1.nn.wo.shape[0])
                tmp = child2.nn.wo[:p].copy()
                child2.nn.wo[:p], child1.nn.wo[:p]  = child1.nn.wo[:p], tmp
                child1.nn.wh = child1.nn.wh.T
                child1.nn.wo = child1.nn.wo.T
                child2.nn.wh = child2.nn.wh.T
                child2.nn.wo = child2.nn.wo.T

                ## 
                p = random.randint(1, 100)
                l = random.randint(1, 3)
                if p < 4 * l:
                    child1.nn.wh = child1.nn.wh.T
                    i = random.randint(0, child1.nn.wh.shape[0] - 1)
                    # child1.nn.wh[i] = child1.nn.wh[i] * -1
                    child1.nn.wh[i] = 1 - child1.nn.wh[i]
                    child1.nn.wh = child1.nn.wh.T
                elif p < 7 * l:
                    child1.nn.wo = child1.nn.wo.T
                    i = random.randint(0, child1.nn.wo.shape[0] - 1)
                    # child1.nn.wo[i] = child1.nn.wo[i] * -1
                    child1.nn.wo[i] = 1 - child1.nn.wo[i]
                    child1.nn.wo = child1.nn.wo.T
                elif p < 10 * l:
                    child2.nn.wh = child2.nn.wh.T
                    i = random.randint(0, child2.nn.wh.shape[0] - 1)
                    # child2.nn.wh[i] = child2.nn.wh[i] * -1
                    child2.nn.wh[i] = 1 - child2.nn.wh[i]
                    child2.nn.wh = child2.nn.wh.T
                elif p < 13 * l:
                    child2.nn.wo = child2.nn.wo.T
                    i = random.randint(0, child2.nn.wo.shape[0] - 1)
                    # child2.nn.wo[i] = child2.nn.wo[i] * -1
                    child2.nn.wo[i] = 1 - child2.nn.wo[i]
                    child2.nn.wo = child2.nn.wo.T

                new_players.append(child1)
                new_players.append(child2)

            if prev_players.__len__() % 2 != 0:
                new_players.append(prev_players[-1])

            return new_players[:num_players]

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
