import numpy as np


class TestClass:
    """
    Test classe with methods for simple evaluation of certain states, when one move in particular is desirable
    """
    def test_5x5(self, actor):
        a = actor.eval_state(np.array([1, 2, 0, 0, 0,
                                       1, 2, 0, 0, 0,
                                       1, 2, 0, 0, 0,
                                       1, 2, 0, 0, 0,
                                       0, 0, 0, 0, 0]), 1)
        b = actor.eval_state(np.array([2, 2, 2, 2, 0,
                                       1, 1, 1, 1, 0,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0]), 2)
        c = actor.eval_state(np.array([0, 0, 0, 0, 0,
                                       0, 2, 1, 1, 0,
                                       0, 1, 2, 2, 0,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0]), 2)
        d = actor.eval_state(np.array([0, 0, 0, 0, 0,
                                       0, 2, 2, 1, 1,
                                       1, 1, 2, 2, 0,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0]), 1)

        # High
        e = actor.eval_state(np.array([1, 2, 0, 0, 0,
                                       1, 2, 0, 0, 0,
                                       1, 2, 0, 0, 0,
                                       1, 2, 0, 0, 0,
                                       0, 0, 0, 0, 0]), 2)
        f = actor.eval_state(np.array([2, 2, 2, 2, 0,
                                       1, 1, 1, 1, 0,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0]), 1)
        g = actor.eval_state(np.array([0, 0, 0, 0, 0,
                                       0, 2, 1, 1, 0,
                                       0, 1, 2, 2, 0,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0]), 1)
        h = actor.eval_state(np.array([0, 0, 0, 0, 0,
                                       0, 2, 2, 1, 1,
                                       1, 1, 2, 2, 0,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0]), 2)

        print(a + b + c + d + (1 - e) + (1 - f) + (1 - g) + (1 - h))

    def test_7x7(self, actor):
        a = actor.eval_state(np.array([1, 2, 0, 0, 0, 0, 0,
                                       1, 2, 0, 0, 0, 0, 0,
                                       1, 2, 0, 0, 0, 0, 0,
                                       1, 2, 0, 0, 0, 0, 0,
                                       1, 2, 0, 0, 0, 0, 0,
                                       1, 2, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0]), 1)
        b = actor.eval_state(np.array([2, 2, 2, 2, 2, 2, 0,
                                       1, 1, 1, 1, 1, 1, 0,
                                       0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0]), 2)
        print(a + b)

    def get_test_acc(self, actor, n):
        if n == 5:
            self.test_5x5(actor)
        elif n == 7:
            self.test_7x7(actor)
        else:
            pass
            print("No test data for n =", n)
