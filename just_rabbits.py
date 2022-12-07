
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import seaborn as sns


SIZE = 500  # The dimensions of the field
OFFSPRING = 2 # Max offspring offspring when a rabbit reproduces
GRASS_RATE = 0.025 # Probability that grass grows back at any location in the next season.
WRAP = False # Does the field wrap around on itself when rabbits move?

class Rabbit:
    """ A furry creature roaming a field in search of grass to eat.
    Mr. Rabbit must eat enough to reproduce, otherwise he will starve. """

    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0

    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the rabbit some grass """
        self.eaten += amount

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
        self.rabbits = []
        self.field = np.ones(shape=(SIZE,SIZE), dtype=int)
        self.nrabbits = []
        self.ngrass = []


    def add_rabbit(self, rabbit):
        """ A new rabbit is added to the field """
        self.rabbits.append(rabbit)

    def move(self):
        """ Rabbits move """
        for r in self.rabbits:
            r.move()

    def eat(self):
        """ Rabbits eat (if they find grass where they are) """

        for rabbit in self.rabbits:
            rabbit.eat(self.field[rabbit.x,rabbit.y])
            self.field[rabbit.x,rabbit.y] = 0

    def survive(self):
        """ Rabbits who eat some grass live to eat another day """
        self.rabbits = [r for r in self.rabbits if r.eaten > 0]

    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        born = []
        for rabbit in self.rabbits:
            for _ in range(rnd.randint(1,OFFSPRING)):
                born.append(rabbit.reproduce())
        self.rabbits += born

        # Capture field state for historical tracking
        self.nrabbits.append(self.num_rabbits())
        self.ngrass.append(self.amount_of_grass())

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_rabbits(self):
        rabbits = np.zeros(shape=(SIZE,SIZE), dtype=int)
        for r in self.rabbits:
            rabbits[r.x, r.y] = 1
        return rabbits

    def num_rabbits(self):
        """ How many rabbits are there in the field ? """
        return len(self.rabbits)

    def amount_of_grass(self):
        return self.field.sum()

    def generation(self):
        """ Run one generation of rabbits """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()

    def history(self, showTrack=True, showPercentage=True, marker='.'):


        plt.figure(figsize=(6,6))
        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")

        xs = self.nrabbits[:]
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x/maxrabbit for x in xs]
            plt.xlabel("% Rabbits")

        ys = self.ngrass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [y/maxgrass for y in ys]
            plt.ylabel("% Rabbits")

        if showTrack:
            plt.plot(xs, ys, marker=marker)
        else:
            plt.scatter(xs, ys, marker=marker)

        plt.grid()

        plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history.png", bbox_inches='tight')
        plt.show()

    def history2(self):
        xs = self.nrabbits[:]
        ys = self.ngrass[:]

        sns.set_style('dark')
        f, ax = plt.subplots(figsize=(7, 6))

        sns.scatterplot(x=xs, y=ys, s=5, color=".15")
        sns.histplot(x=xs, y=ys, bins=50, pthresh=.1, cmap="mako")
        sns.kdeplot(x=xs, y=ys, levels=5, color="r", linewidths=1)
        plt.grid()
        plt.xlim(0, max(xs)*1.2)

        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")
        plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history2.png", bbox_inches='tight')
        plt.show()


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
        field.add_rabbit(Rabbit())

    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5,5))
    im = plt.imshow(array, cmap='PiYG', interpolation='hamming', aspect='auto', vmin=0, vmax=1)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000000, interval=1, repeat=True)
    plt.show()

    field.history()
    field.history2()


if __name__ == '__main__':
    main()



"""
def main():

    # Create the ecosystem
    field = Field()
    for _ in range(10):
        field.add_rabbit(Rabbit())


    # Run the world
    gen = 0
    
    while gen < 500:
        field.display(gen)
        if gen % 100 == 0:
            print(gen, field.num_rabbits(), field.amount_of_grass())
        field.move()
        field.eat()
        field.survive()
        field.reproduce()
        field.grow()
        gen += 1

    plt.show()
    field.plot()

if __name__ == '__main__':
    main()


"""





