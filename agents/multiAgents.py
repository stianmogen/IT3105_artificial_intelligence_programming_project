from hex.game_state import HexGameState
from hex.manager import Node
from nn.neuralNet import NeuralNet


class MonteCarlo:
    """
    https://www.youtube.com/watch?v=UXW2yZndl7U
    1. tree traversal UCB1
    2. node expansion
    3. rollout (random simulation)
    4. backpropagation
    """

    def __init__(self, game: HexGameState, root: Node, c=1, num_simulations=10):
        self.game = game
        self.root = root
        self.state = {}
        self.action_states = {}
        self.c = c

    """
    Q(s,a) = value (expected final result) of doing action a from state s.
    These appear on the edges between tree nodes.
    Over time (many expansions, rollouts and backpropagations), these Q values approach those found by Minimax.
    """

    def get_q_value(self, action: Node):
        if action == None:
            return self.root.get_Q()
        return action.get_Q()

    """
    with N(s,a) = number of times branch (s,a) has been traversed, and
    P(s,a) = prior probability p(a | s). In AlphaGo, these P(s,a) come from
    the Actor-SL neural network (trained on a database of expert moves).
    """

    def get_n_value(self, state, action=None):
        # n value is depended on action value definition
        if action:
            if action in self.root.children:
                return action.get_N()
        return self.root.get_N()

    """
    random component. should probably move possible_new_states to game class
    potentially 
    """

    def get_random_action(self):
        pass

    def evaluate(self, state):
        # something along these lines
        return NeuralNet.predict(state)

    def rollout(self, state):
        # best action chosen for the rollout, based on predictions from evaluate
        pred = self.evaluate(state)
        return NeuralNet.best_action(pred)

    """
    we will use Upper Confidence Bound (UCT)
    u(s,a) = c * sqrt ( log(N(s)) / ( 1 + N(s,a) ) )
    """

    def exploration_bonus(self, state, action):
        if self.get_n_value(state) == 0:
            return np.inf
        return self.c * np.sqrt(np.log(self.get_n_value(state) / (1 + self.get_n_value(state, action))))

    def choose_branch_max(self, state, action):
        return self.get_q_value() + self.exploration_bonus(state, action)

    def choose_branch_min(self, state, action):
        return self.get_q_value() - self.exploration_bonus(state, action)

    def traverse(self):
        # TODO how tf to traverse
        pass

    def expand(self, state, parent: Node):
        if self.game.winner:
            return 0
        children = []
        for move in self.game.check_win(0, 0, state):
            children.appen(Node(move, parent))
        parent.add_children(children)
        return True

