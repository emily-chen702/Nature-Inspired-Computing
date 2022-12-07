
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import seaborn as sns


SIZE = 250  # The dimensions of the field
OFFSPRING = 2 # Max offspring offspring when a rabbit reproduces
GRASS_RATE = 0.025 # Probability that grass grows back at any location in the next season.
WRAP = False # Does the field wrap around on itself when rabbits move?

class Organism:
    """ A furry creature roaming a field in search of grass to eat.
    Mr. Rabbit must eat enough to reproduce, otherwise he will starve. """

    def __init__(self, name:str, prey:str, speed:int = 1):
        self.name = name
        self.prey = prey
        self.speed = speed
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0

    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, prey, value):
        """ Feed the rabbit some grass """
        survived = value.copy()
        if prey == 'grass':
            self.eaten += value
        else:
            for o in value:
                if o.x == self.x and o.y == self.y:
                    survived.remove(o)
                    self.eaten += 1
            return survived



    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.choice([-1,0,1])) % SIZE
            self.y = (self.y + rnd.choice([-1,0,1])) % SIZE
        else:
            self.x = min(SIZE-1, max(0, (self.x + rnd.choice([-1, 0, 1]))))
            self.y = min(SIZE-1, max(0, (self.y + rnd.choice([-1, 0, 1]))))


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass """

    def __init__(self):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no rabbits """
        self.organisms = {}
        self.field = np.ones(shape=(SIZE,SIZE), dtype=int)
        self.ngen = []
        self.norganisms = {}
        self.gen = 0


    def add_organism(self, rabbit):
        name = rabbit.name
        if name in self.organisms.keys():
            self.organisms[name] += [rabbit]
        else:
            self.organisms[name] = [rabbit]
        # """ A new rabbit is added to the field """
        # self.rabbits.append(rabbit)

    def move(self):
        """ Rabbits move """
        for group in self.organisms.values():
            for o in group:
                o.move()

    def eat(self):
        """ Rabbits eat (if they find grass where they are) """
        for name, group in self.organisms.items():
            prey = group[0].prey
            if prey == 'grass':
                for o in group:
                    o.eat('grass', self.field[o.x,o.y])
                    self.field[o.x,o.y] = 0
            else:
                for o in group:
                    survived = o.eat(prey, group)
                    self.organisms[name] = survived


    def survive(self):
        """ Rabbits who eat some grass live to eat another day """
        for name, group in self.organisms.items():
            self.organisms[name] = [o for o in group if o.eaten > 0]

    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        for name, group in self.organisms.items():
            born = []
            for o in group:
                for _ in range(rnd.randint(1,OFFSPRING)):
                    born.append(o.reproduce())
            self.organisms[name] += born
        self.update_history_counts()
        # Capture field state for historical tracking
        # self.nrabbits.append(self.num_organisms('rabbits'))
        # self.ngrass.append(self.amount_of_grass())
    def update_history_counts(self):
        self.ngen += [self.gen]
        
        for name, group in self.organisms.items():
            if name in self.norganisms.keys():
                self.norganisms[name] += [len(group)]
            else:
                self.norganisms[name] = [len(group)]
        
        if 'grass' in self.norganisms.keys():
            self.norganisms['grass'] += [self.amount_of_grass()]
        else:
            self.norganisms['grass'] = [self.amount_of_grass()]

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_organisms(self, name: str):
        o_status = np.zeros(shape=(SIZE,SIZE), dtype=int)
        for o in self.organisms[name]:
            o_status[o.x, o.y] = 1
        return o_status

    def num_organisms(self, name:str):
        """ How many rabbits are there in the field ? """
        return len(self.organisms[name])

    def amount_of_grass(self):
        return self.field.sum()

    def generation(self):
        """ Run one generation of rabbits """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()
        self.gen += 1

    def history(self, showTrack=True, showPercentage=True, marker='.'):


        plt.figure(figsize=(6,6))
        plt.xlabel("# Orgamisms")
        plt.ylabel("# Grass")

        grass = self.norganisms['grass'][:]

        if showTrack:
            for organism, value in self.norganisms.items():
                if organism != 'grass':
                    plt.plot(grass[:], value[:], marker=marker)
        else:
            for organism, value in self.norganisms.items():
                if organism != 'grass':
                    plt.scatter(grass[:], value[:], marker=marker)

        plt.grid()

        plt.title("Organisms vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history.png", bbox_inches='tight')
        plt.show()

    def history(self, showTrack=True, showPercentage=True, marker='.'):
            plt.figure(figsize=(6,6))
            plt.xlabel("# Orgamisms")
            plt.ylabel("# Grass")

            # grass = self.norganisms['grass'][:]
            gens = self.ngen[:]

            if showTrack:
                for value in self.norganisms.values():
                    plt.plot(gens[:], value[:], marker=marker)
            else:
                for value in self.norganisms.values():
                    plt.scatter(gens[:], value[:], marker=marker)

            plt.grid()

            plt.title("Organisms over Time: GROW_RATE =" + str(GRASS_RATE))
            plt.savefig("history2.png", bbox_inches='tight')
            plt.show()

    # def history2(self):
    #     xs = self.nrabbits[:]
    #     ys = self.ngrass[:]

    #     sns.set_style('dark')
    #     f, ax = plt.subplots(figsize=(7, 6))

    #     sns.scatterplot(x=xs, y=ys, s=5, color=".15")
    #     sns.histplot(x=xs, y=ys, bins=50, pthresh=.1, cmap="mako")
    #     sns.kdeplot(x=xs, y=ys, levels=5, color="r", linewidths=1)
    #     plt.grid()
    #     plt.xlim(0, max(xs)*1.2)

    #     plt.xlabel("# Rabbits")
    #     plt.ylabel("# Grass")
    #     plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
    #     plt.savefig("history2.png", bbox_inches='tight')
    #     plt.show()


def animate(i, field, im):
    field.generation()
    # print("AFTER: ", i, np.sum(field.field), len(field.rabbits))
    im.set_array(field.field)
    plt.title("generation = " + str(i))
    return im,


def main():

    # Create the ecosystem
    field = Field()

    for _ in range(1):
        field.add_organism(Organism('rabbit', 'grass'))
    
    for _ in range(10):
        field.add_organism(Organism('fox', 'rabbit'))

    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5,5))
    im = plt.imshow(array, cmap='PiYG', interpolation='hamming', aspect='auto', vmin=0, vmax=2)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=10000, interval=1, repeat=True)
    plt.show()

    field.history()
    # field.history2()


if __name__ == '__main__':
    main()