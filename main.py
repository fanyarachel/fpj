from utils import *
from usercf import UserCF
from itemcf import ItemCF
from randomrs import RandomRS
from pixie import Pixie

num_experiment = 50
mrr_arr = np.zeros((num_experiment, 4))
map_arr = np.zeros((num_experiment, 4))
ndcg_arr = np.zeros((num_experiment, 4))
for tmp_seed in tqdm.trange(num_experiment):
    # set random seed to 0, 1, ..., num_experiment - 1
    np.random.seed(tmp_seed)
    p, n, train, test = load_data()

    # evaluate random recommendation system
    random_model = RandomRS(train, test)
    model_perf = random_model.eval()
    mrr_arr[tmp_seed][0] = model_perf['mrr']
    map_arr[tmp_seed][0] = model_perf['map']
    ndcg_arr[tmp_seed][0] = model_perf['ndcg']

    # evaluate user-based collaborative filtering recommendation system
    user_cf_model = UserCF(train, test)
    model_perf = user_cf_model.eval()
    mrr_arr[tmp_seed][1] = model_perf['mrr']
    map_arr[tmp_seed][1] = model_perf['map']
    ndcg_arr[tmp_seed][1] = model_perf['ndcg']

    # evaluate item-based collaborative filtering recommendation system
    item_cf_model = ItemCF(train, test)
    model_perf = item_cf_model.eval()
    mrr_arr[tmp_seed][2] = model_perf['mrr']
    map_arr[tmp_seed][2] = model_perf['map']
    ndcg_arr[tmp_seed][2] = model_perf['ndcg']

    # evaluate pixie recommendation system
    pixie_model = Pixie(train, test)
    model_perf = pixie_model.eval()
    mrr_arr[tmp_seed][3] = model_perf['mrr']
    map_arr[tmp_seed][3] = model_perf['map']
    ndcg_arr[tmp_seed][3] = model_perf['ndcg']

mrr_avg = np.mean(mrr_arr, 0)
map_avg = np.mean(map_arr, 0)
ndcg_avg = np.mean(ndcg_arr, 0)
for arr, name in [(mrr_avg, 'MRR'), (map_avg, 'MAP'), (ndcg_avg, 'NDCG')]:
    print("\nMethod\t{}".format(name))
    print("Random\t{:.3f}".format(arr[0]))
    print("UserCF\t{:.3f}".format(arr[1]))
    print("ItemCF\t{:.3f}".format(arr[2]))
    print("Pixie\t{:.3f}".format(arr[3]))
