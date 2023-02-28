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
        """
        For each pile there is a possibility of taking 1 to all remaining counters
        :param state: current game state
        :return: generator with all possible new states
        """
        for pile, counters in enumerate(state):
            for remain in range(counters):
                yield state[:pile] + [remain] + state[pile + 1:]
