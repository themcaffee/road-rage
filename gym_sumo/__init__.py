from gym.envs.registration import register

register(
    id='SumoEnv-v0',
    entry_point="gym_sumo.gym_sumo:SumoEnv",
)
