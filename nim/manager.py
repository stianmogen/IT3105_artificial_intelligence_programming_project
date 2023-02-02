

class Manager:
    def choose_move(self):
        pass

    def win_state(self):
        pass

class Agent:
    def __init__(self, index=0):
        self.index = index


    def getAction(self, state):
        pass


class AgentState:
    def __init__(self, playerNum):
        self.playerNum = playerNum

    def __str__(self):
        return "Player " + self.playerNum

class Actions:
    def __init__(self, piles, maxPicks):
        pass


def minimax(state, is_maximizing):
    if (score := evaluate(state, is_maximizing)) is not None:
        return score

    if is_maximizing:
        scores = [
            minimax(new_state, is_maximizing=False)
            for new_state in possible_new_states(state)
        ]
        return max(scores)
    else:
        scores = [
            minimax(new_state, is_maximizing=True)
            for new_state in possible_new_states(state)
        ]
        return min(scores)


def best_move(state, is_maximizing):
    for new_state in possible_new_states(state):
        score = minimax(new_state, is_maximizing=is_maximizing)
        if score > 0:
            break
    return score, new_state


def evaluate(state, is_maximizing):
    if state == 0:
        return 1 if is_maximizing else -1


def possible_new_states(state):
    for pile, counters in enumerate(state):
        for remain in range(counters):
            yield state[:pile] + (remain,) + state[pile + 1:]
