import numpy as np

from environment.beer_game import BeerGame

if __name__ == '__main__':
    beer_game = BeerGame(n_agents=2)
    start_state = beer_game.reset()
    # beer_game.render()
    done = False
    while not done:
        actions = np.random.randint(0, 16, size=2)
        print(actions)
        step_state, step_rewards, done = beer_game.step(list(actions))
        beer_game.render()
