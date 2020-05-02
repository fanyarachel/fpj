from recsys import RecSys
from utils import *


class UserCF(RecSys):
    def __init__(self, train, test):
        super(UserCF, self).__init__(train, test)
        self.sim_mat = self.generate_sim()

    def generate_sim(self):
        """
        generate patient similarity matrix
        :return: dict of dict, sim_mat[pid1][pid2] = similarity(pid1, pid2),
            in which the similarity calculated based on the common-narrative ratings in the training data.
        """
        n2p = {}
        for p, nlst in self.train.items():
            for n in nlst:
                n2p.setdefault(n, set())
                n2p[n].add(p)
        sim_mat = {p1: {p2: 0 for p2 in self.train} for p1 in self.train}
        for n, pset in n2p.items():
            for p1 in pset:
                for p2 in pset:
                    if p1 != p2:
                        sim_mat[p1][p2] += self.train[p1][n] * self.train[p2][n]
        for p1, p2dct in sim_mat.items():
            for p2 in p2dct:
                sim_mat[p1][p2] /= np.sqrt(len(self.train[p1]) * len(self.train[p2]))
        return sim_mat

    def recommend(self, p1):
        """
        recommend based on the patient similarity matrix
        :param p1: patient id
        :return: list of (nid, score) pairs, sorted by scores
        """
        rank = {}
        for p2, sim_val in sorted(self.sim_mat[p1].items(), key=itemgetter(1), reverse=True):
            for n in self.train[p2]:
                if n not in self.train[p1]:
                    rank.setdefault(n, 0)
                    rank[n] += sim_val
        return sorted(rank.items(), key=itemgetter(1), reverse=True)
