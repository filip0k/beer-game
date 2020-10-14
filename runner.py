import numpy as np

from environment.beer_game import BeerGame

if __name__ == '__main__':
    beer_game = BeerGame()
    start_state = beer_game.reset()
    beer_game.render()
    done = False
    while not done:
        actions = np.random.randint(0, 16, size=4)
        done = beer_game.step(list(actions))
        beer_game.render()
