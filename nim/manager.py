import copy
import random

from node import Node


class MiniMaxAgent:
    def __init__(self, state):
        self.rootstate = state

    def minimax(self, state, is_maximizing, alpha=float('-inf'), beta=float('inf')):
        if (score := self.evaluate(state, is_maximizing)) is not None:
            return score

        scores = []
        for move in state.legal_moves:
            new_state = copy.deepcopy(state)
            new_state.move(move)
            score = self.minimax(new_state, is_maximizing=not is_maximizing, alpha=alpha, beta=beta)
            scores.append(score)

            if is_maximizing:
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            else:
                beta = min(beta, score)
                if beta <= alpha:
                    break

        return max(scores) if is_maximizing else min(scores)

    def best_move(self, state):
        for move in state.legal_moves:
            new_state = copy.deepcopy(state)
            new_state.move(move)
            score = self.minimax(new_state, is_maximizing=False)
            if score > 0:
                break
        return move

    def get_move(self):
        state = copy.deepcopy(self.rootstate)
        return self.best_move(state)

    def evaluate(self, state, is_maximizing):
        if all(counters == 0 for counters in state.piles):
            return 1 if is_maximizing else -1
        return None


class MCTSAgent:
    def __init__(self, state, c, rollouts=500):
        self.rootstate = state
        self.root = Node()
        self.c = c
        self.rollouts = rollouts

    def search(self):
        self.root = Node()

        for _ in range(self.rollouts):
            node, state = self.select_node()
            if self.expand(node, state):
                node = random.choice(node.children)
                state.move(node.move)
            turn = state.current_player
            winner = self.rollout(state)
            reward = 0 if winner == turn else 1
            self.backup(node, reward)

    def best_move(self):
        max_value = max(self.root.children, key=lambda edge: edge.N).N
        max_nodes = [n for n in self.root.children if n.N == max_value]
        best_node = random.choice(max_nodes)

        return best_node.move

    def get_move(self):
        self.search()
        return self.best_move()

    def select_node(self):
        node = self.root
        state = copy.deepcopy(self.rootstate)

        while len(node.children) != 0:
            max_value = max(node.children, key=lambda edge: edge.value(self.c)).value(self.c)
            max_nodes = [n for n in node.children if n.value(self.c) == max_value]
            node = random.choice(max_nodes)
            state.move(node.move)
            if node.N == 0:
                return node, state

        return node, state

    def expand(self, node, state):
        children = []
        if state.winner != 0:
            return False  # game is over at this node so nothing to expand

        for move in state.legal_moves:
            children.append(Node(move, node))

        node.add_children(children)
        return True

    def rollout(self, state):
        while state.winner == 0:
            move = random.choice(list(state.legal_moves))
            state.move(move)
        return state.winner

    def backup(self, node, reward):
        while node is not None:
            node.N += 1
            node.score += reward
            node = node.parent
            reward = 1 - reward