import copy

from hex.game_state import HexGameState


def possible_new_states(state):
    for y, x in state.empty_spaces:
        new_state = copy.deepcopy(state)
        new_state.place_piece(y, x)
        yield new_state


class MiniMaxAgent:

    def __init__(self, state: HexGameState):
        self.state = state

    def minimax(self, state, is_maximizing):
        if (score := self.evaluate(state)) is not None:
            return score
        return (max if is_maximizing else min)(
            self.minimax(new_state, is_maximizing=not is_maximizing)
            for new_state in possible_new_states(state)
        )

    def best_move(self):
        for y, x in self.state.empty_spaces:
            new_state = copy.deepcopy(self.state)
            new_state.place_piece(y, x)
            score = self.minimax(new_state, is_maximizing=False)
            if score > 0:
                break
        return y, x

    def evaluate(self, state):
        if state.winner is not None:
            if state.winner == -1:
                return 0
            return 1 if self.state.current_player.value == state.winner else -1
        return None