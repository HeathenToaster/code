# fittest_creature
Evolutionary Genetic Algorithm for Steering Behaviours written in python with pygame

![](data_examples/example1.png)

## Disclaimer

Firstly thanks to Daniel Shiffman for the idea, the nature of code, and his awesome videos.

If someone reads this know that you're welcome to point code mistakes (there will be!!) :)

This is more of a remainder for myself if I forget how does it work in the future, but I hope it helps anyone interested in python and genetic algorithms.

## Details

The **old_version/fittest_ship.py** script was just me learning pygame and experimenting with genetic algorithms.

In **fittest_creature.py** the spawned creatures try to survive, the longer they live the more childs they will have.

All the creatures have an array that stores its DNA, each item in the array corresponds to a number from 0 to 1 and is linked to a property or a "phenotype" of the creature.


The environment is simple, there is food, and poison (green vs red circles), if a creature eats food it will gain health, if it eats poison it will lose health. Also they lose a health amount overtime.
The values can be changed in settings.py.

Creatures spawn with random DNA values, here is an example:

```
dna[0] = 0.2774876569919621  --> MaxHealth: 145.60197488128608, MaxVel: 74.39802511871393, Size: 41
dna[1] = 0.7841259541028673  --> FoodAttr: 11.365038164114694
dna[2] = 0.3977295324539377  --> PoisonAttr: -4.09081870184249
dna[3] = 0.7463319410665977  --> FoodDist: 154.3397493919876
dna[4] = 0.6083194845586946  --> PoisonDist: 129.497507220565
```

Here, dna[1] is mapped to Food attraction, which in this example has a minimum of -20, and a maximum of 20, so 0.78 translates to 11.36.
The rest of the properties work in the same way, with the exception of dna[0] which is mapped to three properties, max health, max velocity and size, I think that the bigger a creature is, the slower it is, and the more health it has.

So for example:
- dna[0] == 0.8  --> Bigger creature, big health pool, Low max velocity
- dna[0] == 0.2  --> Smaller creatures, small health pool, high max velocity

That's with the hope that the smaller ones catch more food and are able to sustain the health degeneration, but with the current settings I find that the creatures with high fitness have a somewhat big size.

The creature above has some nice properties to survive, its attracted to food (positive value) and avoids poison (negative value), can see food at a range of 154 and poison at a range of 129 (poison distance lower than food distance is a good idea as you will see running a simulation).


### How does the genetic algorithm work here

Creatures have a fitness function that takes into account its age, food eaten and poison eaten, the more fitness, the more likely they are to breed.

The breed function of the creature returns a mutated version of its own DNA, there is a chance to mutate each property, and a total amount that property can mutate which **decreases** as the creature has more fitness.
So, the less fitness a creature has, the higher a property can mutate, that way when we have a bad population (bad = low fitness) creatures will mutate more.

There are two modes that can be toggled by pressing **w**:
- Continuous: Creatures have a chance to breed while they're alive, based on its fitness of course, if all of them die a new set of creatures will spawn, the idea is that when some good creatures are in play they will keep breeding and so will their childs.
- By Gen: We spawn a full set of creatures, let them play and when all die the fittest have a greater chance to breed the next generation, but all of them have a chance.

_NOTE: in both modes variation is added with a chance to spawn a completely new creature_

There are a few more variables in play, the code has a lot of comments and python it self is pretty undertandable =)

### Hotkeys

- Press **v** to turn on/off a visual representation of food/poison properties of the creatures.
- Press **n** to turn on/off a visual representation of desired and vel vectors of the creatures.
- Press **w** switches spawn mode, between Continuous and By Gen, current status is shown in window title.w
- Press **s** to turn on/off save to csv file. (current mode can be seen on status bar).
 This will save two .csv files, *history.csv with data from all dead creatures and *stats.csv with a bunch of statistics.
- Press **p** to print to the console information about the current records.
- Press **i** to print to the console statistical information.
