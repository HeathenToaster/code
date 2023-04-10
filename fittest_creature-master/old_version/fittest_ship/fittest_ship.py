import pygame as pg
import pygame.gfxdraw
from pygame.math import Vector2 as vec
from random import randint, uniform
import os

# change to current dir
os.chdir(os.path.abspath(os.path.dirname(__file__)))

### CONFIGURATION ###
WIN_WIDTH = 1280
WIN_HEIGHT = 800
FPS = 60
BACKGROUND_COLOR = (16, 22, 33)

# ship images
SHIP_DICT = {}  # initialized in run function, after pg.display initialized
USE_IMAGES = True  # display ship with the png images or as polygons

MAX_FOOD = 40
MAX_POISON = 50
MAX_VEHICLES = 7
VEHICLE_SIZE = 32

FOOD_VALUE = 12
POISON_VALUE = -50
HEALTH_DEGENERATION = 3.5  # hp per second
BREEDING_WAIT = 15000  # in milliseconds
BREEDING_CHANCE = 0.005
BREED_AGE = 80
MUTATION_CHANCE = 0.1

TOTAL_CORE_POINTS = 340
MAX_DISTANCE_POINTS = 400
MIN_DISTANCE_POINTS = 20
MAX_ATTRACTION_POINTS = 30
MIN_MAX_ATTRACTION_RANGE = 20
MAX_STEER_FORCE = 3

MIN_WANDER_RING_DISTANCE = 5
MAX_WANDER_RING_DISTANCE = 100
MIN_WANDER_RING_RADIUS = 50
MAX_WANDER_RING_RADIUS = 500
MIN_WANDER_RING_WAIT = 300
MAX_WANDER_RING_WAIT = 4000




class Vehicle(pg.sprite.Sprite):
    def __init__(self, pos, size, DNA=None):
        super().__init__()  # init pg.sprite.Sprite

        if not USE_IMAGES:
            # calc the size
            self.size = size
            y2 = self.size / 3   # pointing vertice
            y3 = self.size - y2  # max height

            self.image = pg.Surface((self.size + 1, y3 + 1), pg.SRCALPHA)
            self.poly_coords = [(0, 0), (self.size, y2), (0, y3)]
            pg.gfxdraw.filled_polygon(
                self.image, self.poly_coords, pg.Color('green'))
            # Reference to the original. Allows rotating without losing quality
            self.orig_image = self.image
        else:
            self.poly_coords = [(18, 14), (33, 14), (33, 12), (35, 11),
                                (40, 14), (40, 15), (37, 17), (35, 18), (33, 17), (33, 15), (18, 15)]
            self.image = SHIP_DICT["ship_0"]
            self.orig_image = self.image

        self.rect = self.image.get_rect(center=pos)

        # Non-DNA properties
        self.position = pos  # current position
        self.velocity = vec(0.0, 0.0)  # current velocity vector
        self.acceleration = vec(0.0, 0.0)  # current acceleration vector
        self.health = 100  # current health
        self.age = 0.0  # current age in seconds
        # current generation (if this is a child it has inherited DNA, and will be increased by 1)
        self.generation = 0
        self.childs = 0  # obvious.. number of childs or number of times it bred :D
        # current color (health dependant with function: change_color_by_health)
        self.color = (0, 255, 0)
        self.last_bred_time = pg.time.get_ticks() + \
            BREEDING_WAIT + randint(5, 15)  # last time this vehicle bred

        # the DNA:
        self.DNA = DNA

        # Wander-Ring Non-DNA values:
        self.last_target_time = 0  # last time a target was adquired in milliseconds
        # current position of the center of the ring
        self.wander_ring_pos = self.position
        # point in the circle radius that the vehicle will seek when wandering
        self.wander_target = self.position

        # if no DNA, set initial values:
        if DNA is None:
            self.set_initial_values_noDNA()
        else:
            # this is a child:
            self.max_velocity = self.DNA[0]
            self.max_health = self.DNA[1]
            self.max_steer_force = self.DNA[2]
            self.food_distance = self.DNA[3]
            self.poison_distance = self.DNA[4]
            self.food_attraction = self.DNA[5]
            self.poison_attraction = self.DNA[6]
            self.wander_ring_distance = self.DNA[7]
            self.wander_ring_radius = self.DNA[8]
            self.wander_ring_wait = self.DNA[9]

    def set_initial_values_noDNA(self):
        # max_velocity, max_health and max_steer_force are related in this example
        # and have limited points to distribute
        self.max_velocity = 0
        self.max_health = 0
        self.max_steer_force = 0.0
        total_core_points = TOTAL_CORE_POINTS
        while total_core_points:
            seed = uniform(0, 1)

            if seed > 0.66:
                self.max_velocity += 1
            elif seed > 0.33:
                self.max_health += 1
            elif self.max_steer_force < MAX_STEER_FORCE:
                self.max_steer_force += 0.01

            total_core_points -= 1
        ############################################

        # distance at which the vehicle can see food
        self.food_distance = randint(10, 500)
        self.poison_distance = randint(10, 500)
        while not (MIN_DISTANCE_POINTS < (self.food_distance + self.poison_distance) < MAX_DISTANCE_POINTS):
            self.food_distance = randint(10, 500)
            self.poison_distance = randint(10, 500)

        # attraction to food and poison
        self.food_attraction = uniform(-MIN_MAX_ATTRACTION_RANGE,
                                       MIN_MAX_ATTRACTION_RANGE)
        self.poison_attraction = uniform(-MIN_MAX_ATTRACTION_RANGE,
                                         MIN_MAX_ATTRACTION_RANGE)
        while (abs(self.food_attraction) + abs(self.poison_attraction)) > MAX_ATTRACTION_POINTS:
            self.food_attraction = uniform(-MIN_MAX_ATTRACTION_RANGE,
                                           MIN_MAX_ATTRACTION_RANGE)
            self.poison_attraction = uniform(-MIN_MAX_ATTRACTION_RANGE,
                                             MIN_MAX_ATTRACTION_RANGE)

        self.wander_ring_distance = randint(
            MIN_WANDER_RING_DISTANCE, MAX_WANDER_RING_DISTANCE)  # 50
        self.wander_ring_radius = randint(
            MIN_WANDER_RING_RADIUS, MAX_WANDER_RING_RADIUS)  # 200
        self.wander_ring_wait = randint(
            MIN_WANDER_RING_WAIT, MAX_WANDER_RING_WAIT)

        # now set the values to the DNA
        self.DNA = []
        self.DNA.append(self.max_velocity)
        self.DNA.append(self.max_health)
        self.DNA.append(self.max_steer_force)
        self.DNA.append(self.food_distance)
        self.DNA.append(self.poison_distance)
        self.DNA.append(self.food_attraction)
        self.DNA.append(self.poison_attraction)
        self.DNA.append(self.wander_ring_distance)
        self.DNA.append(self.wander_ring_radius)
        self.DNA.append(self.wander_ring_wait)

    def breed(self):
        if self.age > BREED_AGE and self.health > self.max_health / 2:
            if uniform(0, 1) < BREEDING_CHANCE:
                now = pg.time.get_ticks()
                if now - self.last_bred_time > BREEDING_WAIT:
                    self.last_bred_time = now
                    print(
                        f"[{now}] id:{id(self)} age:{self.age} childs:{self.childs} gen:{self.generation}  ... breeding!")
                    new_DNA = self.DNA.copy()
                    # MUTATE DNA
                    new_DNA = self.mutate_DNA(new_DNA)
                    return new_DNA

    def mutate_DNA(self, DNA):
        now = pg.time.get_ticks()
        if uniform(0.0, 1.0) < MUTATION_CHANCE:
            print(
                f"[{now}] id:{id(self)} mutating max_velocity, max_health and max_steer_force!")

            # first we substract points at random
            mutation_points = 5
            while mutation_points:
                seed = uniform(0, 1)

                if seed > 0.66:
                    # max_velocity
                    DNA[0] -= 1
                elif seed > 0.33:
                    # max_health
                    DNA[1] -= 1
                else:
                    # max_steer_force
                    DNA[2] -= 0.01

                mutation_points -= 1

            # then we add the same amount at random again
            mutation_points = 5
            while mutation_points:
                seed = uniform(0, 1)

                if seed > 0.66:
                    # max_velocity
                    DNA[0] += 1
                elif seed > 0.33:
                    # max_health
                    DNA[1] += 1
                elif DNA[2] < MAX_STEER_FORCE:
                    # max_steer_force
                    DNA[2] += 0.01

                mutation_points -= 1

        if uniform(0.0, 1.0) < MUTATION_CHANCE:
            print(f"[{now}] id:{id(self)} mutating food and poison distances!")
            # food distance
            DNA[3] += randint(-1, 1)
            # poison distance
            DNA[4] += randint(-1, 1)
            while not (MIN_DISTANCE_POINTS < (DNA[3] + DNA[4]) < MAX_DISTANCE_POINTS):
                DNA[3] += randint(-1, 1)
                DNA[4] += randint(-1, 1)

        if uniform(0.0, 1.0) < MUTATION_CHANCE:
            print(f"[{now}] id:{id(self)} mutating food and poison attraction!")

            # food_attraction
            DNA[5] += uniform(-0.1, 0.1)
            # poison_attraction
            DNA[6] += uniform(-0.1, 0.1)
            while (abs(DNA[5]) + abs(DNA[6])) > MAX_ATTRACTION_POINTS:
                DNA[5] -= uniform(0, 0.1)
                DNA[6] -= uniform(0, 0.1)

        if uniform(0.0, 1.0) < MUTATION_CHANCE:
            print(f"[{now}] id:{id(self)} mutating wander ring values!")
            # wander_ring_distance
            DNA[7] += randint(-1, 1)
            while not (MIN_WANDER_RING_DISTANCE < DNA[7] < MAX_WANDER_RING_DISTANCE):
                DNA[7] += randint(-1, 1)

            # wander_ring_radius
            DNA[8] += randint(-1, 1)
            while not (MIN_WANDER_RING_RADIUS < DNA[8] < MAX_WANDER_RING_RADIUS):
                DNA[8] += randint(-1, 1)
            # wander_ring_wait
            DNA[9] += randint(-1, 1)
            while not (MIN_WANDER_RING_WAIT < DNA[9] < MAX_WANDER_RING_WAIT):
                DNA[9] += randint(-1, 1)

        return DNA

    def change_color_by_health(self):
        percentage = max(0, self.health) / self.max_health  # max() avoids negative values
        green = min(int(percentage * 255), 255)  # min avoids going over 255, which is an invalid value
        red = int(255 - green)
        return (red, green, 0)

    def seek(self, target):
        # calculate the desired vector:
        self.desired = (target - self.position).normalize() * self.max_velocity
        # the steer force, a vector that points from velocity to desired:
        steer = (self.desired - self.velocity)
        if steer.length() > self.max_steer_force:
            steer.scale_to_length(self.max_steer_force)

        # return the steer force
        return steer

    # the same seek function without changing self.desired, because it will be set in search_food()
    def seek2(self, target):
        desired = (target - self.position).normalize() * self.max_velocity
        steer = (desired - self.velocity)
        if steer.length() > self.max_steer_force:
            steer.scale_to_length(self.max_steer_force)
        return steer

    def wander(self):
        now = pg.time.get_ticks()
        if now - self.last_target_time > 500:
            self.last_target_time = now
            target = vec((randint(0, WIN_WIDTH), randint(0, WIN_HEIGHT)))
        return self.seek(target)

    def wander_ring(self):
        now = pg.time.get_ticks()
        if now - self.last_target_time > self.wander_ring_wait:
            self.last_target_time = now

            new_pos = vec((randint(0, int(WIN_WIDTH)),
                           randint(0, int(WIN_HEIGHT))))
            if self.velocity.length():
                self.wander_ring_pos = new_pos + self.velocity.normalize() * \
                    self.wander_ring_distance
            else:
                self.wander_ring_pos = new_pos * self.wander_ring_distance

        target = self.wander_ring_pos + \
            vec(self.wander_ring_radius, 0).rotate(uniform(0, 360))

        self.wander_target = target  # only for drawing its vector

        return self.seek(target)

    def search_food(self, foods, poisons):
        targets = {}

        for food in foods:
            food_distance = (self.position - food.position).length()
            if food_distance < self.food_distance:
                targets[food] = food_distance

        for poison in poisons:
            poison_distance = (self.position - poison.position).length()
            if poison_distance < self.poison_distance:
                targets[poison] = poison_distance

        steer = vec(0.0, 0.0)
        if len(targets):
            # get the closest
            min_dist = min(targets.values())
            for fp, dist in targets.items():
                # get the steer force for the current food
                steer_force = (self.seek2(fp.position))

                # apply the force according to the attraction
                if fp.poison:
                    steer_force *= self.poison_attraction
                else:
                    steer_force *= self.food_attraction

                # the closer, the more steer_force
                if dist:
                    steer_force = steer_force / dist

                # apply some more force to the closest target:
                # WE NEED TO USE ABSOLUTE VALUES NOT TO TURN THE SIGN AGAIN
                if dist == min_dist:
                    if fp.poison:
                        steer_force *= abs(self.poison_attraction)
                    else:
                        steer_force *= abs(self.food_attraction)

                # sum all the steer_force
                steer += steer_force

            # if we didn't reach max velocity:
            steer *= self.max_velocity

            # calculate the desired vector
            desired = steer + self.velocity
            if desired.length() > self.max_velocity:
                desired.scale_to_length(self.max_velocity)
            self.desired = desired

            if steer.length() > self.max_steer_force:
                steer.scale_to_length(self.max_steer_force)

            self.acceleration = steer
        else:
            self.acceleration = self.wander_ring()

    def update(self, dt):

        self.velocity += self.acceleration
        if self.velocity.length() > self.max_velocity:
            self.velocity.scale_to_length(self.max_velocity)

        # update position according to velocity, pixels/second
        self.position += self.velocity * dt
        # update rect center to correctly draw on screen
        self.rect.center = self.position

        # rotate according to angle
        self.rotate()

        # We need to reset acceleration to 0 !!
        self.acceleration *= 0

        # calculate how much time in seconds the vehicle has been alive
        self.age += dt

        # we slowly lose health
        self.health -= HEALTH_DEGENERATION * dt
        if self.health <= 0:
            # then, the vehicle is dead, kill it to remove from sprites lists
            # and del the instanciated object
            self.kill()
            del self

    def rotate(self):
        # change color according to current health
        self.color = self.change_color_by_health()
        # change original image color

        if not USE_IMAGES:
            pg.gfxdraw.filled_polygon(
                self.orig_image, self.poly_coords, self.color)
            pg.gfxdraw.aapolygon(
                self.orig_image, self.poly_coords, pg.Color('black'))
        else:
            img_index = (self.velocity.length() *
                         (len(SHIP_DICT) - 1)) // self.max_velocity
            img_name = "ship_" + str(int(img_index))
            self.orig_image = SHIP_DICT[img_name]
            pg.gfxdraw.filled_polygon(
                self.orig_image, self.poly_coords, self.color)

        # The vector to the target.
        direction = self.velocity
        # radius (distance to the target) and the angle
        _, angle = direction.as_polar()
        # Rotate the image by the negative angle (y-axis in pygame is flipped).
        self.image = pg.transform.rotate(self.orig_image, -angle)
        # Create a new rect with the center of the old rect.
        self.rect = self.image.get_rect(center=self.rect.center)

    def draw_vectors(self, screen, opt1=False, opt2=False, opt3=False):
        scale = 1

        if opt2:
            # vel
            pg.draw.line(screen, pg.Color('green'), self.position,
                         (self.position + self.velocity * scale), 4)
            # desired
            pg.draw.line(screen, pg.Color('orange'), self.position,
                         (self.position + self.desired * scale), 4)

        if opt3:
            # wander ring representation
            pg.draw.circle(screen, pg.Color('white'),
                           (int(self.wander_ring_pos.x), int(self.wander_ring_pos.y)), self.wander_ring_radius, 1)
            pg.draw.line(screen, pg.Color('cyan'),
                         self.wander_ring_pos, self.wander_target, 5)

        if opt1:
            # food / poison attraction
            if self.velocity.length():
                direction = self.velocity.normalize()
            else:
                direction = self.velocity
            pg.draw.line(screen, pg.Color('darkgreen'), self.position,
                         (self.position + (direction * self.food_attraction) * scale * 15), 4)
            pg.draw.line(screen, pg.Color('red'), self.position,
                         (self.position + (direction * self.poison_attraction) * scale * 15), 2)
            # food / poison view distance
            pg.draw.circle(screen, pg.Color('darkgreen'),
                           (int(self.position.x), int(self.position.y)), self.food_distance, 1)
            pg.draw.circle(screen, pg.Color('darkred'),
                           (int(self.position.x), int(self.position.y)), self.poison_distance, 1)


class Food(pg.sprite.Sprite):
    def __init__(self, pos, is_poison=False):
        super().__init__()  # init pg.sprite.Sprite
        self.image = pg.Surface((7, 7), pg.SRCALPHA)
        self.rect = self.image.get_rect(center=pos)
        self.position = pos
        self.poison = is_poison
        self.color = pg.Color('green')
        if self.poison:
            self.color = pg.Color('red')
        #pg.draw.circle(self.image, self.color, (3, 3), 3)
        pg.gfxdraw.filled_circle(self.image, 3, 3, 3, self.color)


class Game:
    def __init__(self):
        self.width = WIN_WIDTH
        self.height = WIN_HEIGHT
        self.fps = FPS
        self.running = True
        pg.init()
        self.screen = screen
        self.clock = clock

        self.draw_vectors = False
        self.draw_vectors2 = False
        self.draw_vectors3 = False

        # to easily use update on all the sprites
        self.all_sprites = pg.sprite.Group()

        # vehicle sprites
        self.vehicle_sprites = pg.sprite.Group()
        self.max_vehicles = MAX_VEHICLES
        # food sprites
        self.food_sprites = pg.sprite.Group()
        self.max_food = MAX_FOOD
        # poison sprites
        self.poison_sprites = pg.sprite.Group()
        self.max_poison = MAX_POISON

        # most ancient vehicle
        self.age_record = 0
        self.vr = None  # instance of the most ancient vehicle
        self.cvr = None  # instance of the current most ancient vehicle

    def events(self):
        # Events here, without pg.event.get() or pg.event.wait() window becomes irresponsibe
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False

            # this will process they keystroke once, without repeating if holded down
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.running = False
                elif event.key == pg.K_v:
                    self.draw_vectors = not self.draw_vectors
                elif event.key == pg.K_b:
                    self.draw_vectors2 = not self.draw_vectors2
                elif event.key == pg.K_n:
                    self.draw_vectors3 = not self.draw_vectors3
                elif event.key == pg.K_p:
                    print(
                        f"\n[{pg.time.get_ticks()}] total: {len(self.vehicle_sprites)}\n")

                    def print_record(v):
                        print(f"max hp: {v.max_health}  max vel: {v.max_velocity}  max steer: {v.max_steer_force}\n" +
                              f"food attr: {v.food_attraction} poison attr: {v.poison_attraction}\n" +
                              f"food dist: {v.food_distance}  poison dist: {v.poison_distance}  current HP: {v.health}\n" +
                              f"id: {id(v)}  gen: {v.generation}  childs: {v.childs}  age: {v.age} seconds.\n")
                    # print info
                    if self.cvr is not None:
                        print_record(self.cvr)
                    if self.vr is not None:
                        print_record(self.vr)

                elif event.key == pg.K_m:
                    for sprite in self.all_sprites:
                        sprite.kill()
                        del sprite

    def key_events(self):
        pass
        # this will trigger the action meanwhile the key is pressed down
        # pressed = pg.key.get_pressed()
        # if pressed[pg.K_LEFT]:
        # move left

    def check_newpos_isvalid(self, newpos, dist=32):
        valid = True
        for s in self.all_sprites:
            if (newpos - s.position).length() < dist:
                valid = False
                break

        return valid

    def spawn_food(self):
        # food_distance = (self.position - food.position).length()
        # spawns one kind of food at a time
        if len(self.food_sprites) <= self.max_food + len(self.vehicle_sprites):
            newpos = vec(randint(0, WIN_WIDTH), randint(0, WIN_HEIGHT))

            if self.check_newpos_isvalid(newpos):
                f = Food(newpos)
                self.food_sprites.add(f)
                self.all_sprites.add(f)

        if len(self.poison_sprites) <= self.max_poison + len(self.vehicle_sprites):
            newpos = vec(randint(0, WIN_WIDTH), randint(0, WIN_HEIGHT))

            if self.check_newpos_isvalid(newpos):
                f = Food((newpos), is_poison=True)
                self.poison_sprites.add(f)
                self.all_sprites.add(f)

    def spawn_vehicles(self, child_DNA=None, parent=None):
        if child_DNA is None:
            # create a new vehicle if we are under the max
            if len(self.vehicle_sprites) < self.max_vehicles:
                newpos = vec(randint(0, WIN_WIDTH), randint(0, WIN_HEIGHT))

                if self.check_newpos_isvalid(newpos):
                    p = Vehicle(newpos, VEHICLE_SIZE)
                    self.all_sprites.add(p)
                    self.vehicle_sprites.add(p)
        else:
            if len(self.vehicle_sprites) < self.max_vehicles * 2:
                # create a new child vehicle
                # we move it a bit away from parents position
                newpos = vec(parent.position.x +
                             randint(-40, 40), parent.position.y)
                p = Vehicle(newpos, VEHICLE_SIZE, child_DNA)
                p.generation += parent.generation + 1  # increase childs generation
                parent.childs += 1  # increase parent child counter
                self.all_sprites.add(p)
                self.vehicle_sprites.add(p)
            else:
                print(
                    f"[{pg.time.get_ticks()}] id:{id(parent)} pop:{len(self.vehicle_sprites)}  Can't Breed! Max Population!")

    def process_collisions(self):
        # check for food collisions
        hits = pg.sprite.groupcollide(self.vehicle_sprites, self.food_sprites, False, True,
                                      collided=pg.sprite.collide_circle_ratio(0.60))
        if hits:
            for vehicle in hits:
                for food in hits[vehicle]:
                    # we did eat food, increase health
                    vehicle.health += FOOD_VALUE
                    # cut to max health if we surpassed it
                    vehicle.health = min(vehicle.max_health, vehicle.health)
                    # kill and remove the food
                    food.kill()  # should be killed already as its set to True in groupcollide
                    del food  # delete instanciated object

        # check for poison collisions
        hits = pg.sprite.groupcollide(self.vehicle_sprites, self.poison_sprites, False, True,
                                      collided=pg.sprite.collide_circle_ratio(0.60))
        if hits:
            for vehicle in hits:
                for poison in hits[vehicle]:
                    # we did eat some poison, lower the health
                    vehicle.health += POISON_VALUE
                    # kill and delete the poison object
                    poison.kill()
                    del poison

    def game_loop(self):
        while self.running:
            # get delta time in seconds (default is miliseconds)
            dt = self.clock.get_time() / 1000

            # spawn food & poison
            self.spawn_food()

            # spawn more population if needed, with a chance, to allow the possible childs to take its place
            if uniform(0, 1) < 0.005:
                self.spawn_vehicles()

            self.events()
            self.key_events()

            # try to breed:
            for vehicle in self.vehicle_sprites:
                # each vehicle has a chance to breed if it meets some requeriments
                child_DNA = vehicle.breed()
                if child_DNA is not None:
                    self.spawn_vehicles(child_DNA=child_DNA, parent=vehicle)

            # search for food:
            for vehicle in self.vehicle_sprites:
                vehicle.search_food(self.food_sprites, self.poison_sprites)

            # update all the sprites
            self.all_sprites.update(dt)

            # we check if we ate food or poison with groupcollision
            self.process_collisions()

            # fill the screen then draw sprites
            self.screen.fill(BACKGROUND_COLOR)
            self.all_sprites.draw(self.screen)

            # check for the current and record of vehicle age
            self.cvr = None
            current_age_record = 0
            for v in self.vehicle_sprites:
                # this loop also draws vectors:
                v.draw_vectors(self.screen, self.draw_vectors,
                               self.draw_vectors2, self.draw_vectors3)

                # age record vehicle:
                if v.age > self.age_record:
                    self.age_record = v.age
                    self.vr = v

                # current age record vehicle:
                if v.age > current_age_record:
                    current_age_record = v.age
                    self.cvr = v

            if self.cvr is not None:
                pg.gfxdraw.filled_circle(screen, int(self.cvr.position.x), int(self.cvr.position.y),
                                         7, pg.Color('steelblue2'))
                # pg.draw.circle(screen, (255, 255, 120),
                #        (int(self.cvr.position.x), int(self.cvr.position.y)), 7)

            # switch "frame"
            pg.display.flip()

            pg.display.set_caption("{:.2f}".format(clock.get_fps()))
            self.clock.tick(self.fps)

    def run(self):
        # initial population
        while len(self.vehicle_sprites) < self.max_vehicles:
            newpos = vec(randint(0, WIN_WIDTH), randint(0, WIN_HEIGHT))

            if self.check_newpos_isvalid(newpos):
                p = Vehicle(newpos, VEHICLE_SIZE)
                self.all_sprites.add(p)
                self.vehicle_sprites.add(p)

        self.game_loop()
        pg.quit()


if __name__ == "__main__":
    screen = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pg.time.Clock()
    game = Game()

    # initialize ship images dict
    for f in os.listdir('res'):
        if f.endswith('.png'):
            path = os.path.join('res', f)
            key = f[:-4]  # ship_0.png --> ship_0
            SHIP_DICT[key] = pg.image.load(path).convert_alpha()

    game.run()
