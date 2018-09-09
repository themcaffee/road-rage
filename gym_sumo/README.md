# Sumo Gym Environment

A basic environment for the [sumo](http://sumo.dlr.de/wiki/Simulation_of_Urban_MObility_-_Wiki) traffic simulation program. The goal
of the environment is to optimize a single 4 way red light to create
the fastest road network possible.


## Setup

To use you will need to [install sumo](http://sumo.dlr.de/wiki/Installing) and set $SUMO_HOME to the
appropriate location. If you installed using a package manager this is
probably `/usr/share/sumo`.


## Configuration

There are multiple variables on the environment that can be set that
effect how it works.

```
# Display a GUI or not
env.nogui
# Type of prediction to use (DQN, random, or timed)
env.prediction_type
# To display verbose debug information
env.debug
# To write the tripinfo after every simulation
env.write_tripinfo
```