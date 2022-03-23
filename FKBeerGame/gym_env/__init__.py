from gym.envs.registration import register

register(
    id='beer-game-v0',
    entry_point='gym_env.envs:BeerGame',
)
