def nim_game(piles, max_sticks):
    print("Current piles: {}".format(piles))
    while any(piles):
        print("Player 1 turn")
        pile = int(input("Choose pile (1-3): "))
        while pile < 1 or pile > 3 or piles[pile-1] == 0:
            pile = int(input("Invalid pile, choose pile (1-3): "))
        sticks_taken = int(input("Choose sticks (1-{}): ".format(min(piles[pile-1], max_sticks))))
        while sticks_taken < 1 or sticks_taken > max_sticks or sticks_taken > piles[pile-1]:
            sticks_taken = int(input("Invalid choice, choose sticks (1-{}): ".format(min(piles[pile-1], max_sticks))))
        piles[pile-1] -= sticks_taken
        print("Current piles: {}".format(piles))
        if not any(piles):
            print("Player 1 wins!")
            return
        print("Player 2 turn")
        pile = int(input("Choose pile (1-3): "))
        while pile < 1 or pile > 3 or piles[pile-1] == 0:
            pile = int(input("Invalid pile, choose pile (1-3): "))
        sticks_taken = int(input("Choose sticks (1-{}): ".format(min(piles[pile-1], max_sticks))))
        while sticks_taken < 1 or sticks_taken > max_sticks or sticks_taken > piles[pile-1]:
            sticks_taken = int(input("Invalid choice, choose sticks (1-{}): ".format(min(piles[pile-1], max_sticks))))
        piles[pile-1] -= sticks_taken
        print("Current piles: {}".format(piles))
        if not any(piles):
            print("Player 2 wins!")
            return

nim_game([10, 10, 10], 10)


class GameStateData:
    def __init__(self, piles, maxPicks):
        self.piles = piles
        self.maxPicks = maxPicks
        self._finished = False



class GameState:
    def __init__(self, piles=None, maxPicks=None):
        self.data = GameStateData(piles, maxPicks)

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







