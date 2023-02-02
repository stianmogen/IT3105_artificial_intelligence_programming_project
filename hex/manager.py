class AgentState:

    def __init__(self, playerNum):
        self.playerNum = playerNum

    def __str__(self):
        return "Player " + self.playerNum


class Actions:
    def __init__(self, piles, maxPicks):
        pass


class MiniMaxAgent:

    def minimax(self, state, is_maximizing):
        if (score := self.evaluate(state, is_maximizing)) is not None:
            return score

        return (max if is_maximizing else min)(
            self.minimax(new_state, is_maximizing=not is_maximizing)
            for new_state in self.possible_new_states(state)
        )

    def best_move(self, state):
        for new_state in self.possible_new_states(state):
            score = self.minimax(new_state, is_maximizing=False)
            if score > 0:
                break
        return score, new_state

    def evaluate(self, state, is_maximizing):
        if all(counters == 0 for counters in state):
            return 1 if is_maximizing else -1
        return None

    def possible_new_states(self, state):
        for pile, counters in enumerate(state):
            for remain in range(counters):
                yield state[:pile] + [remain] + state[pile + 1:]