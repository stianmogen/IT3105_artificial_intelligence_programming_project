"""
DisjointSet is a class for handling the relationship between nodes
in a Hex game
"""
class DisjointSet:
    def __init__(self, n):
        """
        defines the parent and rank dictionaries corresponding to the size n
        :param n: size of set
        """
        self.parent = {i: i for i in range(n)}
        self.rank = {i: 0 for i in range(n)}

    def find(self, x):
        """
        recursive lookup of parents nodes until root of x is found
        :param x: node, from which we will find the root
        :return: root of tree containing x
        """
        if self.parent[x] != x:
            # updates the parent to find to be self.parent[x]
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """
        Attaches the roots of x and y to each other
        :param x: node x
        :param y: node y
        :return:
        """
        # finds root nodes of x and y
        x_root = self.find(x)
        y_root = self.find(y)

        # returns if x and y have the same root
        if x_root == y_root:
            return

        # attaches trees together,
        # the tree of smaller rank is attached to tree of greater rank
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            # if tie y is attached to x and rank is increased
            self.parent[y_root] = x_root
            self.rank[x_root] += 1

    def connected(self, x, y):
        """
        check if node x and y are connected with same root
        :param x: node x
        :param y: node y
        :return: true of false depending on if they are connected
        """
        return self.find(x) == self.find(y)

