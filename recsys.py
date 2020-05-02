from utils import *


class RecSys:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def recommend(self, p):
        """ dummy implementation of RecSys recommend, to be implemented by subclasses """
        return []

    def get_eval_rank(self):
        """
        get ground-truth and test-data ranking for evaluation. Here both the ranks have all of the NIDs.
        :return: (groundtruth_rank, recommend_rank)
            groundtruth_rank: the rank in the test dataset
            recommend_rank: the rank given by calling recommend() method for each pid in the test data
        """
        groundtruth_rank = {}
        recommend_rank = {}
        n_all = sorted(list(reduce(set.union,
                                   [set(nlst) for nlst in self.train.values()] +
                                   [set(nlst) for nlst in self.test.values()])))
        for p in self.train:
            if p not in self.test:
                continue
            tmp_groundtruth = self.test[p]
            tmp_groundtruth = sorted(tmp_groundtruth.items(), key=itemgetter(1), reverse=True)
            tmp_recommend = self.recommend(p)

            # pad groundtruth and recommend
            for n in n_all:
                if n not in list(zip(*tmp_groundtruth))[0]:
                    tmp_groundtruth.append((n, 0))
                if not tmp_recommend or n not in list(zip(*tmp_recommend))[0]:
                    tmp_recommend.append((n, 0))
            """tmp_groundtruth = [(-v, k) for k, v in tmp_groundtruth]
            tmp_recommend = [(-v, k) for k, v in tmp_recommend]
            tmp_groundtruth = [(k, v) for v, k in sorted(tmp_groundtruth)]
            tmp_recommend = [(k, v) for v, k in sorted(tmp_recommend)]"""
            tmp_groundtruth = [(k, -v) for k, v in sorted(tmp_groundtruth, key=itemgetter(1), reverse=True)]
            tmp_recommend = [(k, -v) for k, v in sorted(tmp_recommend, key=itemgetter(1), reverse=True)]
            groundtruth_rank[p] = tmp_groundtruth
            recommend_rank[p] = tmp_recommend
        return groundtruth_rank, recommend_rank

    def eval(self):
        """
        evaluate the RecSys performance for pairwise ranking, specially MRR, MAP and NDCG are used.
        :return: dict of keys = ['mrr', 'map', 'ndcg'], values are the corresponding performance values.
        """
        perf_dct = {}
        groundtruth_rank, recommend_rank = self.get_eval_rank()
        mrr_val = mean_reciprocal_rank(groundtruth_rank, recommend_rank)
        map_val = mean_average_precision(groundtruth_rank, recommend_rank)
        ndcg_val = mean_ndcg_at_k(groundtruth_rank, recommend_rank, 50)

        perf_dct['mrr'] = mrr_val
        perf_dct['map'] = map_val
        perf_dct['ndcg'] = ndcg_val
        return perf_dct
