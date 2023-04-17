class NimGameState:
    def __init__(self, piles):
        self.piles = piles
        self.current_player = 1
        self.winner = 0

    @property
    def legal_moves(self):
        for pile, remaining_counters in enumerate(self.piles):
            for takes in range(1, remaining_counters + 1):
                yield pile, takes

    def move(self, move):
        pile, takes = move
        self.piles[pile] -= takes

        self.current_player = 1 if self.current_player == 2 else 2

        if not any(self.piles):
            self.winner = self.current_player
