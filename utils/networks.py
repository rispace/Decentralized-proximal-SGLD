import random
import numpy as np


class Network:
    """ Decentralized Network Architectures

    Description:
        This class generates three different networks, such as fully connected
        network (fcn), circular network (cn), and star network (sn) along with
        the baseline network fully disconnected network (fdn)

    Args:
        size_w (int): The size of the network
        seed (int or None): Seed to reproduce the same numbers

    Returns:
        fully_connected_network(numpy ndarray)
        circular_network(numpy ndarray)
        star_network(numpy ndarray)
        fully_disconnected_network(numpy ndarray)
    """

    def __init__(self, size_w=None, seed=42):
        self.size_w = size_w
        self.seed = seed

    def fully_connected_network(self):
        """ Fully Connected Network

        Description:
            A fully connected network is a kind of network in which all nodes
            are connected to all other nodes.

        Returns:
            w (numpy ndarray): fully connected network matrix

        Example:
        >>> net = Network(size_w=4)
        >>> fcn = net.fully_connected_network()
        >>> print(fcn)
            [[0.3605732  0.21314227 0.21314227 0.21314227]
            [0.21314227 0.3605732  0.21314227 0.21314227]
            [0.21314227 0.21314227 0.3605732  0.21314227]
            [0.21314227 0.21314227 0.21314227 0.3605732 ]]
        >>> print('Row sum of fcn', np.sum(fcn, axis=1))
            [1. 1. 1. 1.]
        >>> print('Column sum of fcn', np.sum(fcn, axis=0))
            [1. 1. 1. 1.]
        """
        if self.size_w > 1:
            max_delta = 1/(self.size_w - 1)
            random.seed(self.seed)
            delta = random.uniform(0, max_delta)
        else:
            raise ValueError("Size of W must be greater than 1")

        w = np.full((self.size_w, self.size_w), delta)
        np.fill_diagonal(w, 1 - delta*(self.size_w-1))

        return w

    def star_network(self):
        """ Star-like Network

        Description:
            A star-like network is a kind of network where all nodes
            are disconnected between themselves but connected to a
            central node.

        Returns:
            w (numpy ndarray): star-like network matrix

        Example:
        >>> net = Network(size_w=4)
        >>> sn = net.star_network()
        >>> print(sn)
            [[0.3605732  0.21314227 0.21314227 0.21314227]
            [0.21314227 0.78685773 0.         0.        ]
            [0.21314227 0.         0.78685773 0.        ]
            [0.21314227 0.         0.         0.78685773]]
        >>> print('Row sum of sn', np.sum(sn, axis=1))
            [1. 1. 1. 1.]
        >>> print('Column sum of sn', np.sum(sn, axis=0))
            [1. 1. 1. 1.]
        """
        if self.size_w > 1:
            max_delta = 1/(self.size_w - 1)
            random.seed(self.seed)
            delta = random.uniform(0, max_delta)
        else:
            raise ValueError("Size of W must be greater than 1")

        w = np.zeros((self.size_w, self.size_w))
        for i in range(self.size_w):
            w[0, i] = w[i, 0] = delta
        np.fill_diagonal(w, 1 - delta)
        w[0, 0] = 1 - delta*(self.size_w-1)

        return w

    def circular_network(self):
        """ Circular Network

        Description:
            A circular network is a kind of network each node is connected
            to its left and right node only.

        Returns:
            w (numpy ndarray): circular network matrix

        Example:
        >>> net = Network(size_w=4)
        >>> cn = net.circular_network()
        >>> print(cn)
            [[0.3605732 0.3197134 0.        0.3197134]
            [0.3197134 0.3605732 0.3197134 0.       ]
            [0.        0.3197134 0.3605732 0.3197134]
            [0.3197134 0.        0.3197134 0.3605732]]
        >>> print('Row sum of cn', np.sum(cn, axis=1))
            [1. 1. 1. 1.]
        >>> print('Column sum of cn', np.sum(cn, axis=0))
            [1. 1. 1. 1.]
        """
        if self.size_w > 2:
            max_delta = 1/2
            random.seed(self.seed)
            delta = random.uniform(0, max_delta)
        else:
            raise ValueError("Size of W must be greater than 2")

        w = np.zeros((self.size_w, self.size_w))
        for i in range(self.size_w):
            for j in range(self.size_w):
                if (i+1) == j or (j+1) == i:
                    w[i, j] = delta
        np.fill_diagonal(w, 1-2*delta)
        w[0, (self.size_w-1)] = w[(self.size_w-1), 0] = delta
        return w

    def disconnected_network(self):
        return np.eye(self.size_w)