import numpy as np

from ActorClient import ActorClient
from agents.mcts_agent import MCTSAgent
from hex.game_state import HexGameState
from nn.anet import Anet2


class MyClient(ActorClient):
 def __init__(self):
  super().__init__()
  f = "hex/7x7/game231.h5"
  self.model = Anet2(board_size=7, load_path=f)
  self.state = HexGameState(size=7)
  self.mcts = MCTSAgent(self.state, self.model, 1, 1, 1, 1, rollouts=200)

 def handle_game_start(self, start_player):
  self.state.reset_board()
  self.state.current_player = 1 if start_player == 2 else 2

 def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
  self.player = series_id
  print("Player id", self.player)

 def handle_get_action(self, state):
  for i, cell in enumerate(state[1:]):
   if cell != 0 and self.state.board[i] == 0:
    self.state.place_piece(i)
    self.mcts.move(i)
  board_size = int((len(state) - 1) ** 0.5)
  move, _, _ = self.mcts.get_move()
  self.mcts.move(move)
  self.state.place_piece(move)
  y = move // board_size
  x = move % board_size

  return int(y), int(x)


if __name__ == '__main__':
 client = MyClient()
 client.run()