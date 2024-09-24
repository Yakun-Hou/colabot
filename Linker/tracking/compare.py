import os
import argparse
import pickle
import sys

import torch
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.analysis.plot_results import get_auc_curve, get_prec_curve
from lib.test.evaluation.environment import env_settings


def get_overall_performance(eval_data):
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    # AUC
    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
    # Precision
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
    # Normalized Precision
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
    return auc, prec_score, norm_prec_score


def get_topk_ao_diff(eval_data1, eval_data2, topk):
    sequence_names = np.array(eval_data1['sequences'])
    ao1 = torch.tensor(eval_data1['avg_overlap_all'])[:, 0] * 100.0
    ao2 = torch.tensor(eval_data2['avg_overlap_all'])[:, 0] * 100.0
    d_ao = ao1 - ao2

    print('---------------------------------------------------')
    topk_ids = torch.argsort(d_ao)[:topk]
    for sid, sequence_name in zip(topk_ids, sequence_names[topk_ids]):
        print('%20s[%04d]: %.1f, %.1f, %.1f' % (sequence_name, sid, d_ao[sid], ao1[sid], ao2[sid]))

    print('---------------------------------------------------')

    lastk_ids = torch.argsort(d_ao)[-topk:]
    for sid, sequence_name in zip(lastk_ids, sequence_names[lastk_ids]):
        print('%20s[%04d]: %.1f, %.1f, %.1f' % (sequence_name, sid, d_ao[sid], ao1[sid], ao2[sid]))
    print('---------------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of config file.')
    parser.add_argument('--tracker1')
    parser.add_argument('--tracker2')
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    settings = env_settings()

    print('%s  vs  %s  on  %s' % (args.tracker1, args.tracker2, args.dataset))
    print('---------------------------------------------------')

    eval_data_path1 = os.path.join(settings.result_plot_path, args.dataset, '%s.pkl' % args.tracker1)
    with open(eval_data_path1, 'rb') as f:
        eval_data1 = pickle.load(f)
    auc1, p1, np1 = get_overall_performance(eval_data1)
    print('%50s, AUC: %.2f, Precision: %.2f, N-Precision: %.2f' % (args.tracker1, auc1, p1, np1))

    eval_data_path2 = os.path.join(settings.result_plot_path, args.dataset, '%s.pkl' % args.tracker2)
    with open(eval_data_path2, 'rb') as f:
        eval_data2 = pickle.load(f)
    auc2, p2, np2 = get_overall_performance(eval_data2)
    print('%50s, AUC: %.2f, Precision: %.2f, N-Precision: %.2f' % (args.tracker2, auc2, p2, np2))

    get_topk_ao_diff(eval_data1, eval_data2, topk=args.topk)
