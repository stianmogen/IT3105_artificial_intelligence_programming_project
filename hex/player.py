from abc import ABC, abstractmethod


"""
Player interface class general purpose abstraction
"""
class PlayerInterface(ABC):
    @abstractmethod
    def get_move(self):
        pass

"""
A hex player implements  a the player interface
and includes game specific logic 
"""
class Player(PlayerInterface):
    def __init__(self, name, board_size):
        """
        :param name: player name
        :param board_size: size of hex boara
        """
        self.name = name
        self.board_size = board_size

    def get_move(self):
        """
        asks player to input a move, only acceps legal moves
        :return: chosen move
        """
        y, x = input(f'Player {self.name}s turn. Enter y and x between 0-{self.board_size - 1}: ').split()

        while not (0 <= int(x) <= self.board_size - 1 and 0 <= int(y) <= self.board_size - 1):
            y, x = input(f'Player {self.name}s turn. Enter y and x between 0-{self.board_size - 1}: ').split()

        y = int(y)
        x = int(x)

        return y * self.board_size + x