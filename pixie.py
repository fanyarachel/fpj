from recsys import RecSys
from utils import *


class Pixie(RecSys):
    def __init__(self, train, test):
        super(Pixie, self).__init__(train, test)
        self.network = self.build_network(train, test)

    def build_network(self, train, test):
        """
        build a bipartite graph of nodes PIDs and NIDs
        :return: a networknx graph object
        """
        g = nx.Graph()
        plst = list(train.keys()) + list(test.keys())
        nlst = sorted(list(reduce(set.union,
                                   [set(nlst) for nlst in self.train.values()] +
                                   [set(nlst) for nlst in self.test.values()])))
        pstrlst = ['p_'+str(pid) for pid in plst]
        nstrlst = ['n_'+str(nid) for nid in nlst]
        for tmp_node in pstrlst+nstrlst:
            g.add_node(tmp_node)
        for p, ndct in self.train.items():
            for n, rating in ndct.items():
                if rating > 0:
                    g.add_edge('p_'+str(p), 'n_'+str(n), weight=rating)
        return g

    def recommend(self, p):
        """
        recommend for patient p based on the graph and random walk with restart algorithm
        :param p: patient id
        :return: sorted list of (patient, score) pairs
        """
        rank = random_walk(self.network, 'p_'+str(p))
        rank = {int(k.split('_')[1]): v for k, v in rank.items()}
        return sorted(rank.items(), key=itemgetter(1), reverse=True)


def random_walk(graph, start_node, num_epoch=10000, restart=0.5):
    """
    random walk with restart algorithm, here only the number of visit for narrative nodes
    :param graph: networknx graph object
    :param start_node: the starting node, i.e., the target patient
    :param num_epoch: the number of random walks
    :param restart: the prob of restart
    :return: dict, nid: count of visits
    """
    if not len(list(graph.neighbors(start_node))):
        return {}
    node2cnt = {}
    cur_node = start_node
    for cur_step in range(num_epoch):
        next_node = np.random.choice(list(graph.neighbors(cur_node)))
        if next_node.startswith('n'):
            node2cnt.setdefault(next_node, 0)
            node2cnt[next_node] += 1
        if np.random.rand() < restart:
            cur_node = start_node
        else:
            cur_node = next_node
    return node2cnt
