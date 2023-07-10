import numpy as np

from Hypercube import Hypercube


class UcbNode:
    hypercube_list: np.array

    # The node obj is simply used to visualize the tree

    def __init__(self, parent_node, h, hypercube_list, index=0, children=None):
        """

        Parameters
        ----------
        parent_node
        h
        hypercube_list
        index index of the node where the index of the root is 0 and it's incremented by one
        going from left to right, up to down
        """
        self.index = index
        self.parent_node = parent_node
        self.h = h
        self.hypercube_list = hypercube_list
        self.dimension = self.hypercube_list[0].get_dimension()
        self.children = children

    def reproduce(self, N):
        """
        This fun creates N new nodes and assigns regions (i.e. hypercubes) to them.
        :return: A list of the N new nodes.
        """
        if len(self.hypercube_list) == 1:
            num_new_hypercubes = 2 ** self.dimension
            new_hypercubes = np.empty(num_new_hypercubes, dtype=Hypercube)
            new_hypercube_length = self.hypercube_list[0].length / 2
            old_center = self.hypercube_list[0].center
            for i in range(num_new_hypercubes):
                center_translation = np.fromiter(
                    map(lambda x: new_hypercube_length / 2 if x == '1' else -new_hypercube_length / 2,
                        list(bin(i)[2:].zfill(self.dimension))),
                    dtype=np.float)
                new_hypercubes[i] = Hypercube(new_hypercube_length, old_center + center_translation)
            children_hypercubes = np.split(new_hypercubes, N)

        else:
            children_hypercubes = np.split(self.hypercube_list, N)

        children_nodes = []
        for i, hcube_arr in enumerate(children_hypercubes):
            children_nodes.append(UcbNode(self, self.h + 1, hcube_arr, self.index * N + i + 1))
        #children_nodes = list(map(lambda hcube_arr: UcbNode(self, self.h + 1, hcube_arr), children_hypercubes))
        self.children = children_nodes
        return children_nodes

    def contains_context(self, context):
        for hypercube in self.hypercube_list:
            if hypercube.is_pt_in_hypercube(context):
                return True
        return False

    def __str__(self):
        return str(self.h) + ": " + str(self.hypercube_list)

    def __repr__(self):
        return self.__str__()
