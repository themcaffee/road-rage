import optparse
from pprint import pprint

import gym
from keras import Input
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess

import gym_sumo
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = "SumoEnv-v0"

# Default training configuration
TRAINING_STEPS = 10000
TRAINING_MAX_STEPS = 3000
TRAINING_WARMUP = 200

# Default evaluation configuration
EVAL_EPISODES = 3
EVAL_MAX_STEPS = 3000


def main(options):
    env = gym.make(ENV_NAME)
    if options.gui:
        env.nogui = False
    options.prediction_type = options.type
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.shape[0]

    actor = make_actor(env, nb_actions)

    critic, action_input, observation_input = make_critic(env, nb_actions)

    # Configure and compile the agent
    memory = SequentialMemory(limit=50000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    dqn = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                    memory=memory, nb_steps_warmup_critic=options.training_warmup, nb_steps_warmup_actor=options.training_warmup,
                    random_process=random_process, gamma=.99, target_model_update=1e-3)
    dqn.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    # Begin training
    print("=================== Starting training.. ==============================")
    dqn.fit(env, nb_steps=options.training_steps, visualize=False, verbose=2, nb_max_episode_steps=options.training_max_steps)

    # After training is done, save the weights
    print("=================== Finished training, saving weights.. ==============")
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Evaluate the model
    print("=================== Finished saving weights, evaluating model ========")
    res = dqn.test(env, nb_episodes=options.eval_episodes, visualize=False, nb_max_episode_steps=options.eval_max_steps, verbose=1)
    pprint(res.history)


def make_actor(env, nb_actions):
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


def make_critic(env, nb_actions):
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return critic, action_input, observation_input


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--gui", action="store_true", default=False, help="Run the GUI version of sumo")
    optParser.add_option("--type", default="DQN", help="The type of prediction to use")
    optParser.add_option("--training-steps", default=TRAINING_STEPS, type="int",
                         help="The number of simulation steps to train for")
    optParser.add_option("--training-max-steps", default=TRAINING_MAX_STEPS, type="int",
                         help="The maximum number of steps during training per episode")
    optParser.add_option("--training-warmup", default=TRAINING_WARMUP, type="int",
                         help="Steps to take randomly before prediction")
    optParser.add_option("--eval-episodes", default=EVAL_EPISODES, type="int",
                         help="Number of episodes to evaluate for")
    optParser.add_option("--eval-max-steps", default=EVAL_MAX_STEPS, type="int",
                         help="Max simulation steps per episode during training")
    options, args = optParser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    main(options)
