from manager import MiniMaxAgent


def nim_game(piles):
    print("Current piles: {}".format(piles))

    agent1 = MiniMaxAgent()
    while any(piles):
        print("Player 1 turn")

        # CHOICE 1
        pile = int(input("Choose pile (1-3): "))
        while pile < 1 or pile > 3 or piles[pile-1] == 0:
            # CHOICE 1
            pile = int(input("Invalid pile, choose pile (1-3): "))
        # CHOICE 1
        sticks_taken = int(input("Choose sticks (1-{}): ".format(min(piles[pile-1], 4))))
        while sticks_taken < 1 or sticks_taken > piles[pile-1]:
            sticks_taken = int(input("Invalid choice, choose sticks (1-{}): ".format(min(piles[pile-1]))))
        piles[pile-1] -= sticks_taken
        print("Current piles: {}".format(piles))
        if not any(piles):
            print("Agent 2 wins!")
            return

        # CHOICE 1
        score, new_state = agent1.best_move(piles)
        print("AGENT CHOSE: ", new_state)

        piles = new_state

        print("Current piles: {}".format(piles))
        if not any(piles):
            print("Player 1 wins!")
            return


nim_game([1, 3, 5, 7])

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
