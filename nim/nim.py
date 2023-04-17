from manager import MiniMaxAgent, MCTSAgent
from game_state import NimGameState


def nim_game(piles):
    print("Current piles: {}".format(piles))

    state = NimGameState(piles)

    minimax = MiniMaxAgent(state=state)
    mcts = MCTSAgent(state=state, c=1, rollouts=20000)

    agents = [('minimax', minimax), ('mcts', mcts)]
    while state.winner == 0:
        name, agent = agents[state.current_player-1]
        move = agent.get_move()
        print(f"{name} CHOSE: ", move)
        state.move(move)
        print(f"Current piles: {piles}")

    print(f"Winner = agent{agents[state.winner-1][0]}")


nim_game([1, 3, 5, 2, 1])

# 1
# 1 2
# 1 4
# 1 2 4


class GameStateData:
    def __init__(self, piles):
        self.piles = piles
        self._finished = False



class GameState:
    def __init__(self, piles=None):
        self.data = GameStateData(piles)

    # [10, 10, 10], 10
    #
    def isFinished(self):
        return self.data._finished


    def getLegalPiles(self):
        if self.isFinished():
            return []
        else:
            return filter(lambda x: x > 0, self.data.piles)


    def getLegalPicks(self, pileIndex):
        if self.isFinished():
            return []
        else:
            return self.data.piles[pileIndex]


    def checkWin(self):
        return all(x <= 0 for x in self.data.piles)
