from abc import ABC, abstractmethod

class PlayerInterface(ABC):
    @abstractmethod
    def get_move(self):
        pass


class Player(PlayerInterface):
    def __init__(self, name, board_size):
        self.name = name
        self.board_size = board_size

    def get_move(self):
        y, x = input(f'Player {self.name}s turn. Enter y and x between 0-{self.board_size - 1}: ').split()

        while not (0 <= int(x) <= self.board_size - 1 and 0 <= int(y) <= self.board_size - 1):
            y, x = input(f'Player {self.name}s turn. Enter y and x between 0-{self.board_size - 1}: ').split()

        y = int(y)
        x = int(x)

        return y * self.board_size + x