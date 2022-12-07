
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np
import copy
import seaborn as sns


SIZE = 500  # The dimensions of the field
OFFSPRING = 2 # Max offspring offspring when a rabbit reproduces
WINTER_GRASS_RATE = 0.001 # Probability that grass grows back at any location in the next season.
SUMMER_GRASS_RATE = 0.03
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

    def move(self, choices:list):
        """ Move up, down, left, right randomly """
        # print('choices', choices)
        if not choices or len(choices) == 4:
            if WRAP:
                self.x = (self.x + rnd.choice([-1,0,1])) % SIZE
                self.y = (self.y + rnd.choice([-1,0,1])) % SIZE
            else:
                self.x = min(SIZE-1, max(0, (self.x + rnd.choice([-1, 0, 1]))))
                self.y = min(SIZE-1, max(0, (self.y + rnd.choice([-1, 0, 1]))))
        else:
            choice = rnd.choice(choices)
            if choice == 'right':
                self.x = min(SIZE-1, max(0, (self.x + 1)))
                self.y = min(SIZE-1, max(0, (self.y)))
            elif choice =='left':
                self.x = min(SIZE-1, max(0, (self.x - 1)))
                self.y = min(SIZE-1, max(0, (self.y)))
            elif choice =='up':
                self.x = min(SIZE-1, max(0, (self.x)))
                self.y = min(SIZE-1, max(0, (self.y - 1)))
            elif choice =='down':
                self.x = min(SIZE-1, max(0, (self.x)))
                self.y = min(SIZE-1, max(0, (self.y + 1)))


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
        # print(len(self.rabbits))
        for r in self.rabbits:
            move_choices = []
            if self.field[(r.x-1)% SIZE, r.y] == 1:
                move_choices += ['left']
            if self.field[(r.x+1)% SIZE, r.y] == 1:
                move_choices += ['right']
            if self.field[r.x, (r.y - 1) % SIZE] == 1:
                move_choices += ['up']
            if self.field[r.x, (r.y + 1) % SIZE] == 1:
                move_choices += ['down']
            r.move(move_choices)

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
        # print(len(born))

        # Capture field state for historical tracking
        self.nrabbits.append(self.num_rabbits())
        self.ngrass.append(self.amount_of_grass())

    def grow(self):
        """ Grass grows back with some probability """
        # growloc = (np.random.rand(SIZE, SIZE) < WINTER_GRASS_RATE) * 1
        rate_increment = (SUMMER_GRASS_RATE - WINTER_GRASS_RATE) / SIZE
        new_field = np.zeros((SIZE, SIZE))
        for i in range(SIZE):
            growloc = (np.random.rand(SIZE, ) < (WINTER_GRASS_RATE + (rate_increment * i))) * 1
            # growloc = (np.random.rand(SIZE, ) < 0.025) * 1
            new_field[i] = growloc

        self.field = np.maximum(self.field, new_field)

    def get_rabbits(self):
        rabbits = np.zeros(shape=(SIZE,SIZE), dtype=int)
        for r in self.rabbits:
            rabbits[r.x, r.y] = 2
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
    
    def get_final_field(self):
        rabbits = self.get_rabbits()
        final_field = np.maximum(self.field, rabbits)
        # print('num_rabbits:', self.num_rabbits())
        # print('num_foxes:', self.num_foxes())
        return final_field

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

        plt.title("Rabbits vs. Grass")
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
        plt.title("Rabbits vs. Grass:")
        plt.savefig("history2.png", bbox_inches='tight')
        plt.show()


def animate(i, field, im):
    field.generation()
    # print("AFTER: ", i, np.sum(field.field), len(field.rabbits))
    im.set_array(field.get_final_field())
    plt.title("generation = " + str(i))
    return im,


def main():

    # Create the ecosystem
    field = Field()

    for _ in range(1):
        field.add_rabbit(Rabbit())
    
    colors = ['yellow', 'green', 'red']  # use strings but can use rgb values 
    cmap = ListedColormap(colors)

    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5,5))
    im = plt.imshow(array, cmap=cmap, interpolation='hamming', aspect='auto', vmin=0, vmax=2)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000000, interval=1, repeat=True)
    plt.show()

    field.history()
    field.history2()


if __name__ == '__main__':
    main()
