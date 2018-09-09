import optparse

import gym
import sumoenv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = "SumoEnv-v0"


def main(options):
    env = gym.make(ENV_NAME)
    if options.gui:
        env.nogui = False
    options.prediction_type = options.type
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    model = make_model(env, nb_actions)

    # Configure and compile the agent
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Begin training
    print("=================== Starting training.. ==============================")
    dqn.fit(env, nb_steps=10000, visualize=False, verbose=2, nb_max_episode_steps=1000)

    # After training is done, save the weights
    print("=================== Finished training, saving weights.. ==============")
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Evaluate the model
    print("=================== Finished saving weights, evaluating model ========")
    dqn.test(env, nb_episodes=3, visualize=False, nb_max_episode_steps=1000, verbose=1)


def make_model(env, nb_actions):
    # Build a simple model
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--gui", action="store_true", default=False, help="Run the GUI version of sumo")
    optParser.add_option("--type", default="DQN", help="The type of prediction to use")
    options, args = optParser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    main(options)
