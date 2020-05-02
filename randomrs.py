from recsys import RecSys
from utils import *


class RandomRS(RecSys):
    def __init__(self, train, test):
        super(RandomRS, self).__init__(train, test)

    def recommend(self, p1):
        """
        recommend randomly
        :param p1: patient id
        :return: list of (nid, score) pairs, sorted by scores
        """
        n_all = sorted(list(reduce(set.union,
                                   [set(nlst) for nlst in self.train.values()] +
                                   [set(nlst) for nlst in self.test.values()])))
        rank = {n: np.random.rand() for n in n_all}
        return sorted(rank.items(), key=itemgetter(1), reverse=True)
