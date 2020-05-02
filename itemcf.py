from recsys import RecSys
from utils import *


class ItemCF(RecSys):
    def __init__(self, train, test):
        super(ItemCF, self).__init__(train, test)
        self.sim_mat = self.generate_sim()

    def generate_sim(self):
        """
        generate narrative similarity matrix
        :return: dict of dict, sim_mat[nid1][nid2] = similarity(nid1, nid2),
            in which the similarity calculated based on the ratings of other narratives from the patients who read it.
        """
        n2cnt = {}
        for p, nlst in self.train.items():
            for n in nlst:
                n2cnt.setdefault(n, 0)
                n2cnt[n] += 1

        sim_mat = {n1: {n2: 0 for n2 in n2cnt} for n1 in n2cnt}
        for p, nlst in self.train.items():
            for n1 in nlst:
                for n2 in nlst:
                    if n1 != n2:
                        sim_mat[n1][n2] += self.train[p][n1] * self.train[p][n2]

        for n1, n2dct in sim_mat.items():
            for n2 in n2dct:
                if not n2cnt[n1] * n2cnt[n2]:
                    sim_mat[n1][n2] = 0
                else:
                    sim_mat[n1][n2] /= np.sqrt(n2cnt[n1] * n2cnt[n2])
        return sim_mat

    def recommend(self, p):
        """
        recommend based on the narrative similarity matrix
        :param p: patient id
        :return: list of (nid, score) pairs, sorted by scores
        """
        rank = {}
        for n1, score in self.train[p].items():
            for n2, sim_val in sorted(self.sim_mat[n1].items(), key=itemgetter(1), reverse=True):
                if n2 not in self.train[p]:
                    rank.setdefault(n2, 0)
                    rank[n2] += sim_val * float(score)
        return sorted(rank.items(), key=itemgetter(1), reverse=True)
