import numpy as np
from gym_env.envs import BeerGame

if __name__ == '__main__':
    beer_game = BeerGame(n_agents=4)
    start_state = beer_game.reset()
    # beer_game.render()
    done = False
    while not done:
        actions = np.random.randint(0, 16, size=4)
        print(actions)
        step_state, step_rewards, done = beer_game.step(list(actions))
        beer_game.render()
    print(beer_game.states)
