################################################################################
#
# This file implements two quantitative measures for speaker identification:
#
# * equal error rate
# * minimum detection cost
#
# It also provides a CLI for calculating these measures on some
# predefined pairs of speaker (mis)matches.
#
# Author(s): Nik Vaessen
################################################################################

from operator import itemgetter
from typing import List, Tuple

import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

################################################################################
# helper methods for both measures


def _verify_correct_scores(
    groundtruth_scores: List[int], predicted_scores: List[float]
):
    if len(groundtruth_scores) != len(predicted_scores):
        raise ValueError(
            f"length of input lists should match, while"
            f" groundtruth_scores={len(groundtruth_scores)} and"
            f" predicted_scores={len(predicted_scores)}"
        )
    # if np.min(predicted_scores) < 0 or np.max(predicted_scores) > 1:
    #     raise ValueError(
    #         f"predictions should be in range [0, 1], while they"
    #         f" are actually in range "
    #         f"[{np.min(predicted_scores)}, "
    #         f"{np.max(predicted_scores)}]"
    #     )
    if not all(np.isin(groundtruth_scores, [0, 1])):
        raise ValueError(
            f"groundtruth values should be either 0 and 1, while "
            f"they are actually one of {np.unique(groundtruth_scores)}"
        )


################################################################################
# EER (equal-error-rate)


def calculate_eer(
    groundtruth_scores: List[int], predicted_scores: List[float], pos_label: int = 1
):
    """
    Calculate the equal error rate between a list of groundtruth pos/neg scores
    and a list of predicted pos/neg scores.

    Adapted from: https://github.com/a-nagrani/VoxSRC2020/blob/master/compute_EER.py

    :param groundtruth_scores: a list of groundtruth integer values (either 0 or 1)
    :param predicted_scores: a list of prediction float values (in range [0, 1])
    :param pos_label: which value (either 0 or 1) represents positive. Defaults to 1
    :return: a tuple containing the equal error rate and the corresponding threshold
    """
    _verify_correct_scores(groundtruth_scores, predicted_scores)

    if not all(np.isin([pos_label], [0, 1])):
        raise ValueError(f"The positive label should be either 0 or 1, not {pos_label}")

    fpr, tpr, thresholds = roc_curve(
        groundtruth_scores, predicted_scores, pos_label=pos_label
    )
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer).item()

    return eer, thresh


################################################################################
# minimum detection cost - taken from
# https://github.com/a-nagrani/VoxSRC2020/blob/master/compute_min_dcf.py
# Copyright 2018  David Snyder
# This script is modified from the Kaldi toolkit -
# https://github.com/kaldi-asr/kaldi/blob/8ce3a95761e0eb97d95d3db2fcb6b2bfb7ffec5b/egs/sre08/v1/sid/compute_min_dcf.py


def _compute_error_rates(
    groundtruth_scores: List[int],
    predicted_scores: List[float],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Creates a list of false-negative rates, a list of false-positive rates
    and a list of decision thresholds that give those error-rates.

    :param groundtruth_scores: a list of groundtruth integer values (either 0 or 1)
    :param predicted_scores: a list of prediction float values (in range [0, 1])
    :return: a triple with a list of false negative rates, false positive rates
     and a list of decision threshold
    for those rates.
    """
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(
        *sorted(
            [(index, threshold) for index, threshold in enumerate(predicted_scores)],
            key=itemgetter(1),
        )
    )

    groundtruth_scores = [groundtruth_scores[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(groundtruth_scores)):
        if i == 0:
            fnrs.append(groundtruth_scores[i])
            fprs.append(1 - groundtruth_scores[i])
        else:
            fnrs.append(fnrs[i - 1] + groundtruth_scores[i])
            fprs.append(fprs[i - 1] + 1 - groundtruth_scores[i])
    fnrs_norm = sum(groundtruth_scores)
    fprs_norm = len(groundtruth_scores) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of correct positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]

    return fnrs, fprs, thresholds


def _compute_min_dfc(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """
    Computes the minimum of the detection cost function. The comments refer to
    equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.

    :param fnrs: the list of false negative rates
    :param fprs: the list of false positive rates
    :param thresholds: the list of decision thresholds
    :param p_target: a priori probability of the specified target speaker
    :param c_miss: cost of a missed detection
    :param c_fa: cost of a spurious detection
    :return: the minimum detection cost and accompanying threshold
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]

    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_threshold


def calculate_mdc(
    groundtruth_scores: List[int],
    predicted_scores: List[float],
    c_miss: float = 1,
    c_fa: float = 1,
    p_target: float = 0.05,
):
    """
    Calculate the minimum detection cost and threshold based on a list of
    groundtruth and prediction pairs.

    :param groundtruth_scores: the list of groundtruth scores
    :param predicted_scores:
    :param groundtruth_scores: a list of groundtruth integer values (either 0 or 1)
    :param predicted_scores: a list of prediction float values (in range [0, 1])
    :param p_target: a priori probability of the specified target speaker
    :param c_miss: cost of a missed detection
    :param c_fa: cost of a spurious detection
    :return: a tuple containing the minimum detection score and the corresponding threshold
    """
    _verify_correct_scores(groundtruth_scores, predicted_scores)
    if c_miss < 1:
        raise ValueError(f"c_miss={c_miss} should be >= 1")
    if c_fa < 1:
        raise ValueError(f"c_fa={c_fa} should be >= 1")
    if p_target < 0 or p_target > 1:
        raise ValueError(f"p_target={p_target} should be between 0 and 1")

    fnrs, fprs, thresholds = _compute_error_rates(groundtruth_scores, predicted_scores)
    mindcf, threshold = _compute_min_dfc(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

    return mindcf, threshold


################################################################################
# Cavg


def _compute_Cavg(scores:List[tuple],
                  trials:List[tuple]):
   #读取trials
   '''
   trails是一个列表，每个元素为元组其形势为(u1,u2,taget) taget只能为0和1，1表示u1和u2为同一类别
   例如('minna', 'ms-my/15273', 1)
   '''
   def get_langid_dict(trials):
    ''' Get lang2lang_id, utt2lang_id dicts and lang nums, lang_id starts from 0. 
      Also return trial list.
    '''
    langs = []
    # lines = open(trials, 'r').readlines()
    for lang, utt, target in trials:
        #lang, utt, target = line.strip().split()
        langs.append(lang)

    langs = list(set(langs))
    langs.sort()
    lang2lang_id = {}
    for i in range(len(langs)):
        lang2lang_id[langs[i]] = i

    utt2lang_id = {}
    trial_list = {}
    # for line in lines:
    for lang, utt, target in trials:
        #lang, utt, target = line.strip().split()
        if target :
            utt2lang_id[utt] = lang2lang_id[lang]
        trial_list[lang + utt] = target
    return lang2lang_id, utt2lang_id, len(langs), trial_list
   #读取得分
   '''
    scores是一个列表，每个元素为元组其形势为(u1,u2,score)
    例如(jajp ct-cn-10010 -1.927292)
   '''
   def process_pair_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list):
        ''' Replace both lang names and utt ids with their lang ids,
        for unknown utt, just with -1. Also return the min and max scores.
         用它们的lang ID替换lang名称和utt ID，
        对于未知的utt，仅使用-1。同时返回最小和最大分数。
        '''
        pairs = []
        stats = []
        # lines = open(scores, 'r').readlines()
        # for line in lines:
        for lang, utt, score in scores:
            # lang, utt, score = line.strip().split()
            # if trial_list.has_key(lang + utt):
            if (lang + utt) in trial_list:
            # if utt2lang_id.has_key(utt):
                if utt in utt2lang_id:
                    pairs.append([lang2lang_id[lang], utt2lang_id[utt], float(score)])
                else:
                    pairs.append([lang2lang_id[lang], -1, float(score)])
                stats.append(float(score))
        return pairs, min(stats), max(stats)
   
   #计算Cavg
   def get_cavg(pairs, lang_num, min_score, max_score, bins = 20, p_target = 0.5):
        ''' Compute Cavg, using several threshhold bins in [min_score, max_score].
        '''
        cavgs = [0.0] * (bins + 1)
        precision = (max_score - min_score) / bins
        for section in range(bins + 1):
            threshold = min_score + section * precision
            # Cavg for each lang: p_target * p_miss + sum(p_nontarget*p_fa)
            target_cavg = [0.0] * lang_num  # 1
            for lang in range(lang_num):
                p_miss = 0.0 # prob of missing target pairs
                LTa = 0.0 # num of all target pairs
                LTm = 0.0 # num of missing pairs
                lang_num_1 = lang_num + 1 # include one unknown lang
                p_fa = [0.0] * lang_num_1 # prob of false alarm, respect to all other langs
                LNa = [0.0] * lang_num_1 # num of all nontarget pairs, respect to all other langs
                LNf = [0.0] * lang_num_1 # num of false alarm pairs, respect to all other langs
                for line in pairs:
                    if line[0] == lang:
                        if line[1] == lang:
                            LTa += 1
                            if line[2] < threshold:
                                LTm += 1
                        else:
                            LNa[line[1]] += 1
                            if line[2] >= threshold:
                                LNf[line[1]] += 1
                if LTa != 0.0:
                    p_miss = LTm / LTa
                for i in range(lang_num_1):
                    if LNa[i] != 0.0:
                        p_fa[i] = LNf[i] / LNa[i]
                p_nontarget = (1 - p_target) / (lang_num_1 - 1)
                target_cavg[lang] = p_target * p_miss + p_nontarget*sum(p_fa)
            cavgs[section] = sum(target_cavg) / lang_num
        return cavgs, min(cavgs)
   

   lang2lang_id, utt2lang_id, lang_num, trial_list = get_langid_dict(trials)
   #form = '-pairs'
   pairs, min_score, max_score = process_pair_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list)
   # form = 'pairs'
   # pairs, min_score, max_score = cavg.process_matrix_scores(score, lang2lang_id, utt2lang_id, lang_num, trial_list)
   threshhold_bins = 20
   p_target = 0.5
   cavgs, min_cavg = get_cavg(pairs, lang_num, min_score, max_score, threshhold_bins, p_target)
   return round(min_cavg, 4)
