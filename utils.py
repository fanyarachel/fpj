import pandas as pd
import numpy as np
from operator import itemgetter
from functools import reduce
import networkx as nx
from metrics import *
import random
import tqdm

illegal_NID = [96, 41, 62, 110, 199]
illegal_PID = []


def load_patient():
    """
    load feature data of patients, ignoring given illegal PIDs
    :return: numpy array of patient features
    """
    """ load feature data of patients, ignoring given illegal PIDs """
    raw_df = pd.read_excel('dataset/PersonalProfile.xlsx')
    illegal_rows = []
    for tmp_row in range(raw_df.shape[0]):
        illegal_rows.append(tmp_row)
    filtered_df = raw_df.drop(illegal_rows)
    return np.array(filtered_df)


def load_narrative():
    """
    load feature data of narratives, ignoring given illegal NIDs
    :return: numpy array of narrative features
    """
    raw_df = pd.read_excel('dataset/NarrativeCharacteristics.xlsx')
    illegal_rows = []
    for tmp_row in range(raw_df.shape[0]):
        if raw_df.NID[tmp_row] in illegal_NID:
            illegal_rows.append(tmp_row)
    filtered_df = raw_df.drop(illegal_rows)
    return np.array(filtered_df)


def load_feedback(pivot=0.8):
    """
    load feedback data of (patient, narrative) pairs and split dataset into training dataset and test dataset
    :param pivot: the ratio of the training dataset
    :return: (train, test)
        train: dict of dict, train[pid][nid] = score
        test: dict of dict, train[pid][nid] = score
    """
    raw_df = pd.read_excel('dataset/NarrativeFeedback.xlsx')
    # pid_arr = sorted(list(set(raw_df.PID)))
    # nid_arr = sorted(list(set(raw_df.NID)))
    # pid_dct = {tmp_pid: tmp_idx for tmp_idx, tmp_pid in enumerate(pid_arr)}
    # nid_dct = {tmp_nid: tmp_idx for tmp_idx, tmp_nid in enumerate(nid_arr)}
    # score_mat = np.ones((len(pid_arr), len(nid_arr))) * np.nan
    train, test = {}, {}

    # score is set to be the sum of five feedbacks, which can be modified
    score = np.array(raw_df.Hopeful + raw_df.ConnectedNarrator
                     + raw_df.ConnectedNarrative + raw_df.Learning + raw_df.Empathy)

    sample_set=[]
    for tmp_row in range(raw_df.shape[0]):
        tmp_pid, tmp_nid = raw_df.PID[tmp_row], raw_df.NID[tmp_row]
        sample_set.append((tmp_pid, tmp_nid))
    train_set = random.sample(sample_set, int(len(sample_set)*pivot))
    test_test = list(filter(lambda x: x not in train_set, sample_set))
    for tmp_row in range(raw_df.shape[0]):
        tmp_pid, tmp_nid = raw_df.PID[tmp_row], raw_df.NID[tmp_row]
        if raw_df.NID[tmp_row] in illegal_NID:
            continue
        # score_mat[tmp_pid_idx][tmp_nid_idx] = score[tmp_row]
        if (tmp_pid, tmp_nid) in train_set:
            train.setdefault(tmp_pid, {})
            train[tmp_pid][tmp_nid] = score[tmp_row]
        else:
            test.setdefault(tmp_pid, {})
            test[tmp_pid][tmp_nid] = score[tmp_row]
    return train, test


def load_data():
    """
    load data of patient features, narrative features and feedback scores
    :return (p, n, train, test)
        p: patient features
        n: narrative features
        train: training dataset of score data
        test: test dataset of score data
    """
    p = load_patient()
    n = load_narrative()
    train, test = load_feedback()
    return p, n, train, test
