
import random as rnd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import seaborn as sns


SIZE = 500  # The dimensions of the field
OFFSPRING = 2 # Max offspring offspring when a rabbit reproduces
GRASS_RATE = 0.025 # Probability that grass grows back at any location in the next season.
WRAP = False # Does the field wrap around on itself when rabbits move?
STARTING_RABBIT_POP=100
STARTING_FOX_POP=50


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

class Fox:
    def __init__(self, ):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
    
    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """
        child = Fox()
        child.x = self.x
        child.y = self.y
        child.eaten = 0
        self.eaten = 0
        return child

    def eat(self, amount):
        """ Feed the fox some grass """
        self.eaten += amount
        

    def move(self):
        """ Move up, down, left, right randomly """
        # pick = rnd.choice(['Left', 'Right', 'Up', 'Down'])
        # if pick == 'Right':
        #     print('right')
        #     self.x = (self.x + 1) % SIZE
        # elif pick == 'Left':
        #     print('left')
        #     self.x = (self.x - 1) % SIZE
        # elif pick == 'Up':
        #     print('up')
        #     self.y = (self.y + 1) % SIZE
        # else:
        #     print('down')
        #     self.y = (self.y - 1) % SIZE
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
        self.foxes = []
        self.field = np.ones(shape=(SIZE,SIZE), dtype=int)
        self.nfoxes = []
        self.nrabbits = []
        self.ngrass = []
        self.ngen = []
        self.rabbit_status = np.zeros(shape=(SIZE, SIZE), dtype=int) # maybe make into a set
        self.gen = 0


    def add_rabbit(self, rabbit):
        """ A new rabbit is added to the field """
        self.rabbits.append(rabbit)
    
    def add_fox(self, fox):
        """ A new rabbit is added to the field """
        self.foxes.append(fox)

    def move(self):
        """ Rabbits move """
        for rabbit in self.rabbits:
            self.rabbit_status[rabbit.x, rabbit.y] -= 1
            rabbit.move()
            self.rabbit_status[rabbit.x, rabbit.y] += 1

        for fox in self.foxes:
            fox.move()

    def eat(self):
        """ Rabbits eat (if they find grass where they are) """
        rabbits_eaten = 0
        rabbits_eaten_coord = []
        for fox in self.foxes:
            rabbit_count = self.rabbit_status[fox.x, fox.y]

            if rabbit_count >= 1:
                rabbits_eaten += 1
                self.rabbit_status[fox.x, fox.y] -= 0

            fox.eat(rabbit_count)
            rabbits_eaten_coord.append((fox.x, fox.y))

        survived = []
        for rabbit in self.rabbits:
            # TODO look up how to remove an object from a numpy list
            # if self.rabbit_status[rabbit.x, rabbit.y] >= 1:
            #     survived += [rabbit]
            # else:
            #     print('eaten')
            if (rabbit.x, rabbit.y) not in rabbits_eaten_coord:
                survived += [rabbit]


            rabbit.eat(self.field[rabbit.x,rabbit.y])
            self.field[rabbit.x,rabbit.y] = 0
        self.rabbits = survived
        # print('rabbits eaten:', rabbits_eaten)

    def survive(self):
        """ Rabbits who eat some grass live to eat another day """
        self.rabbits = [r for r in self.rabbits if r.eaten > 0]

        # if generation is a multiple of 10, kill off the foxes that 
        # have not eaten enough
        if self.gen % 10 == 0:
            self.foxes = [f for f in self.foxes if f.eaten > 0]

    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        born = []
        for rabbit in self.rabbits:
            for _ in range(rnd.randint(1,OFFSPRING)):
                born.append(rabbit.reproduce())
        self.rabbits += born

        if self.gen % 10 == 0:
            born = []
            for fox in self.foxes:
                for _ in range(rnd.randint(1,OFFSPRING)):
                    born.append(fox.reproduce())
            self.foxes += born


        # Capture field state for historical tracking
        self.nrabbits.append(self.num_rabbits())
        self.nfoxes.append(self.num_foxes())
        self.ngrass.append(self.amount_of_grass())
        self.ngen.append(self.gen)

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_rabbits(self):
        rabbits = np.zeros(shape=(SIZE,SIZE), dtype=int)
        for r in self.rabbits:
            rabbits[r.x, r.y] = 2
        return rabbits
    
    def get_foxes(self):
        foxes = np.zeros(shape=(SIZE, SIZE), dtype=int)
        for f in self.foxes:
            foxes[f.x, f.y] = 3
        return foxes

    def num_rabbits(self):
        """ How many rabbits are there in the field ? """
        return len(self.rabbits)
    
    def num_foxes(self):
        """ How many rabbits are there in the field ? """
        return len(self.foxes)

    def amount_of_grass(self):
        return self.field.sum()

    def generation(self):
        """ Run one generation of rabbits """
        self.gen += 1
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()
    
    def get_final_field(self):
        rabbits = self.get_rabbits()
        foxes = self.get_foxes()
        updated_field = np.maximum(self.field, rabbits)
        final_field = np.maximum(updated_field, foxes)
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

    def history3(self, showTrack=True, showPercentage=True, marker='.'):
        
        plt.figure(figsize=(6,6))
        plt.xlabel("Generation")
        plt.ylabel("#")

        grass = self.ngrass[:]
        rabbits = self.nrabbits[:]
        foxes = self.nfoxes[:]
        gens = self.ngen[:]
        print(foxes)
        # print(gens)

        # xf = self.nfoxes
        # if showPercentage:
        #     maxfox = max(xf)
        #     xf = [x/maxfox for x in xf]

        # xs = self.nrabbits[:]
        # if showPercentage:
        #     maxrabbit = max(xs)
        #     xs = [x/maxrabbit for x in xs]
        #     plt.xlabel("% Rabbits")

        # ys = self.ngrass[:]
        # if showPercentage:
        #     maxgrass = max(ys)
        #     ys = [y/maxgrass for y in ys]
        #     plt.ylabel("% Rabbits")
        if showTrack:
            # plt.plot(gens, grass, marker=marker, label='grass')
            plt.plot(gens, foxes, marker=marker, label = 'fox')
            plt.plot(gens, rabbits, marker=marker, label = 'rabbit')
        else:
            # plt.scatter(gens, grass, marker=marker)
            plt.scatter(gens, foxes, marker=marker)
            plt.scatter(gens, rabbits, marker=marker)

        plt.grid()

        plt.title("Foxes vs. Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history3.png", bbox_inches='tight')
        plt.legend()
        plt.show()

def animate(i, field, im):
    field.generation()
    # print("AFTER: ", i, np.sum(field.field), len(field.rabbits))
    im.set_array(field.get_final_field())
    # print(field.field)
    # print(np.max(field.field))
    # im.set_array(field.field)
    plt.title("generation = " + str(i))
    return im,

def create_colormap():
    colors = [(255,255,224),(0, 0, 1), (1, 0, 0), (0, 1, 0)]  # use basic rgb values + yellow for dead grass
    cmap_name = 'rabbit_fox_colormap'
    fig, axs = plt.subplots(2, 2, figsize=(6, 9))
    # fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=4)
    # Fewer bins will result in "coarser" colomap interpolation
    # im = ax.imshow(Z, origin='lower', cmap=cmap)
    # ax.set_title("N bins: %s" % n_bin)
    # fig.colorbar(im, ax=ax)


def main():

    # Create the ecosystem
    field = Field()

    for _ in range(STARTING_RABBIT_POP):
        field.add_rabbit(Rabbit())
    
    for _ in range(STARTING_FOX_POP):
        field.add_fox(Fox())

    colors = ['yellow', 'green', 'red', 'blue']  # use strings but can use rgb values 
    # cmap_name = 'rabbit_fox_colormap'
    # cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=4)
    cmap = ListedColormap(colors)

    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5,5))
    im = plt.imshow(array, cmap=cmap, interpolation='hamming', aspect='auto', vmin=-0.5, vmax=3.5)
    fig.colorbar(im)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000000, interval=1, repeat=True)
    plt.show()

    field.history()
    field.history2()
    field.history3()


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





