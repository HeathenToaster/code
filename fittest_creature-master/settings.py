from time import time

### CONFIGURATION ###
WIN_WIDTH = 1280
WIN_HEIGHT = 860
FPS = 40
BACKGROUND_COLOR = (7, 7, 7)
FOOD_COLOR = (50, 50, 255)
POISON_COLOR = (255, 50, 50)

SAVE_TO_CSV = False
STARTTIME = str(int(time()))  # used to save csv with unique name
SAVE_DELAY = 20 * 1000  # in milliseconds

HEADER1 = ["Time", "Fitness", "Age", "Gen", "Childs", "FoodEaten", "PoisonEaten",
           "MaxVel_MaxHP", "FoodAttraction", "PoisonAttraction",
           "FoodDistance", "PoisonDistance", "MaxSteerForce", "DirAngleMult"]

HEADER2 = []
for header in HEADER1:
    if header == "Time":
        HEADER2.append(header)
    else:
        HEADER2.append('Mean' + header)
        HEADER2.append('Median' + header)

# Switches spawn mode, between Continuous: False, and By Gen: True
SPAWN_MODE = False

TOTAL_CREATURES = 17
MIN_CREATURE_SIZE = 7
MAX_CREATURE_SIZE = 53

# chance to spawn a new creature to add variation to the simulation
# each frame. Only for continuous mode
# keep it low to favor breeding
# if 0 creatures are alive, they spawn in bulk
NEW_CREATURE_CHANCE = 0.003

DNA_SIZE = 7  # number of values in the dna.

# below values are affected by the fitness of the creature
# breed_chance = x / (BREED_CHANCE_VALUE + x);  x --> fitness
BREED_CHANCE_VALUE = 850
MAX_MUTATION_VALUE = 0.2  # how much a property will change when mutated

MUTATION_CHANCE = 0.1  # chance to mutate each property, not affected by fitness

# this should avoid that a new creature spawns directly eating poison or food
# but with (MAX_CREATURE_SIZE // 2) + 1 we won't avoid bigger creatures to pass between
# food/posion that are too close
# biggest gaps make it look ugly and unreal...
DISTANCE_BETWEEN_SPRITES = (MAX_CREATURE_SIZE // 2) + 1

TOTAL_POISON = 76
TOTAL_FOOD = 67
HEALTH_DEGENERATION = 12.3  # creatures will lose hp per second
POISON_VALUE = -52  # negative value as poison is bad!
FOOD_VALUE = 20

# Values that will vary according to DNA changes, but have a max value
MAX_STEER_FORCE = 4
MAX_PERCEPTION_DIST = 300  # max dist at which creatures can evolve to see food & poison
# the highter dir_angle_mult a creature has, the higher priority for targets in front of it
MIN_DIR_ANGLE_MULT = 1
MAX_DIR_ANGLE_MULT = 5
# Creatures have a constraint, they evolve choosing between maxvel and maxhealth
# having more maxhealth means bigger size and less maxvel
TOTAL_MAXVEL_MAXHP_POINTS = 220
# we don't want creatures to spawn with HP values lower than this
# very low values make no sense because they die with health degeneration too fast
MIN_HP = 30

# When the creature finds no food or poison, it wanders
# wander ring properties:
WANDER_RING_DISTANCE = (WIN_WIDTH + WIN_HEIGHT) // 8
WANDER_RING_RADIUS = (WIN_WIDTH + WIN_HEIGHT) // 4
WANDER_RING_WAIT = 2000
