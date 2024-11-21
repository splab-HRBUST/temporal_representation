################################################################################
#
# Implement an Evaluator object which encapsulates the process
# computing performance metric of speech recognition task.
#
# Author(s): Nik Vaessen
################################################################################

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union,Dict
from warnings import warn

import numpy as np
import torch as t
import pandas as pd

from src.eval_metrics import calculate_eer, calculate_mdc,_compute_Cavg
from torch.nn.functional import normalize

################################################################################
# define data structures required for evaluating


@dataclass
class EvaluationPair:
    same_speaker: bool
    sample1_id: str
    sample2_id: str


@dataclass
class EmbeddingSample:
    sample_id: str
    ground_truth: int
    embedding: Union[t.Tensor, List[t.Tensor]]


################################################################################
# abstract base class of an evaluator


class SpeakerRecognitionEvaluator:
    def __init__(self, max_num_training_samples: int):
        self.max_num_training_samples = max_num_training_samples
    
    def evaluate1(self, embeding_mean_dict: Dict[int,EmbeddingSample], samples: List[EmbeddingSample],truth_list:Dict[int , str]):
        ground_truth_scores = []
        prediction_pairs = []
        trials=[]
        scores=[]
        for key in embeding_mean_dict:
            if key not in truth_list:
                 raise ValueError(f"erro key {key}")

        for sample in samples:
            for key in embeding_mean_dict:
                prediction_pairs.append((embeding_mean_dict[key],sample))
                gt=1 if key==sample.ground_truth else 0
                ground_truth_scores.append(gt)
                trials.append((truth_list[key],sample.sample_id,gt))
        prediction_scores = self._compute_prediction_scores(prediction_pairs)

        # prediction_scores = np.clip((np.array(prediction_scores) + 1) / 2, 0, 1)
        # prediction_scores = prediction_scores.tolist()

        try:
            eer, eer_threshold = calculate_eer(
                ground_truth_scores, prediction_scores, pos_label=1
            )
        except (ValueError, ZeroDivisionError) as e:
            # if NaN values, we just return a very bad score
            # so that hparam searches don't crash
            print(f"EER calculation had {e}")
            eer = 1
            eer_threshold = 1337

        for i in range(len(prediction_scores)):
            scores.append((trials[i][0],trials[i][1],prediction_scores[i]))

        try:
            # mdc, mdc_threshold = calculate_mdc(ground_truth_scores, prediction_scores)
            Cavg  = _compute_Cavg(trials=trials,scores=scores)
        except (ValueError, ZeroDivisionError) as e:
            print(f"Cavg calculation had {e}")
            Cavg = 1
            #mdc_threshold = 1337
        # for i in range(len(prediction_scores)):
        #     print(scores[i])
        #     print(trials[i])
        return {
            "eer": round(eer,4),
            # "eer_threshold": round(eer_threshold,4),
            "Cavg": round(Cavg,4)
        }
    

    def evaluate(self, pairs: List[EvaluationPair], samples: List[EmbeddingSample]):
        # create a hashmap for quicker access to samples based on key

        sample_map = {}

        for sample in samples:
            if sample.sample_id in sample_map:
                raise ValueError(f"duplicate key {sample.sample_id}")

            sample_map[sample.sample_id] = sample

        # compute a list of ground truth scores and prediction scores
        ground_truth_scores = []
        prediction_pairs = []

        for pair in pairs:
            if pair.sample1_id not in sample_map or pair.sample2_id not in sample_map:
                warn(f"{pair.sample1_id} or {pair.sample2_id} not in sample_map")
                return {
                    "eer": -1,
                    "eer_threshold": -1,
                    "Cavg": -1,
                }

            s1 = sample_map[pair.sample1_id]
            s2 = sample_map[pair.sample2_id]

            gt = 1 if pair.same_speaker else 0

            ground_truth_scores.append(gt)
            prediction_pairs.append((s1, s2))

        prediction_scores = self._compute_prediction_scores(prediction_pairs)

        trials=[]
        scores=[]
        for i in range(len(pairs)):
            s1=pairs[i].sample1_id.split("/")[0]
            s2=pairs[i].sample2_id
            gt = 1 if pairs[i].same_speaker else 0
            trials.append((s1,s2,gt))
            score = prediction_scores[i]
            scores.append((s1,s2,score))


        # print(prediction_scores) 
        # normalize scores to be between 0 and 1
        prediction_scores = np.clip((np.array(prediction_scores) + 1) / 2, 0, 1)
        prediction_scores = prediction_scores.tolist()

        # info statistics on ground-truth and prediction scores
        #print("ground truth scores")
        pd.DataFrame(ground_truth_scores).describe()
        #print("prediction scores")
        pd.DataFrame(prediction_scores).describe()

        # compute EER
        try:
            eer, eer_threshold = calculate_eer(
                ground_truth_scores, prediction_scores, pos_label=1
            )
        except (ValueError, ZeroDivisionError) as e:
            # if NaN values, we just return a very bad score
            # so that hparam searches don't crash
            print(f"EER calculation had {e}")
            eer = 1
            eer_threshold = 1337

        #compute Cavg
        try:
            # mdc, mdc_threshold = calculate_mdc(ground_truth_scores, prediction_scores)
            Cavg  = _compute_Cavg(trials=trials,scores=scores)
        except (ValueError, ZeroDivisionError) as e:
            print(f"Cavg calculation had {e}")
            Cavg = 1
            #mdc_threshold = 1337

        return {
            "eer": round(eer,4),
            "eer_threshold": round(eer_threshold,4),
            "Cavg": round(Cavg,4)
        }
    def caculate_embeding_mean(self,all_embeding: List[EmbeddingSample])->dict:
        dict_embeding={}
        # for i in all_labels:
        #     dict_embeding[i]=[]
        # dict_embeding[0].append(0)
        # print(dict_embeding)
        for emdeding in all_embeding:
            if(emdeding.ground_truth not in dict_embeding.keys()):
                dict_embeding[emdeding.ground_truth]=[]
            dict_embeding[emdeding.ground_truth].append(emdeding.embedding.unsqueeze(dim=0))
        mean_embeding={}
        for key in dict_embeding:
            try:
                means=t.mean(t.cat(dict_embeding[key],dim=0),dim=0)
                means_sample=EmbeddingSample(
                    sample_id=key,
                    ground_truth=key,
                    embedding=means
                )
            except  TypeError as e:
                means_sample=None
                raise SystemExit(e)
            mean_embeding[key]=means_sample
        return mean_embeding    


    @abstractmethod
    def _compute_prediction_scores(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ) -> List[float]:
        pass

    def _transform_pairs_to_tensor(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ):
        # construct the comparison batches
        b1 = []
        b2 = []

        for s1, s2 in pairs:
            b1.append(s1.embedding)
            b2.append(s2.embedding)

        b1 = t.stack(b1)
        b2 = t.stack(b2)

        return b1, b2

    @abstractmethod
    def fit_parameters(
        self, embedding_tensors: List[t.Tensor], label_tensors: List[t.Tensor]
    ):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass


################################################################################
# Utility methods common between evaluators


def compute_mean_std_batch(all_tensors: t.Tensor):
    # compute mean and std over each dimension of EMBEDDING_SIZE
    # with a tensor of shape [NUM_SAMPLES, EMBEDDING_SIZE]
    std, mean = t.std_mean(all_tensors, dim=0)

    return mean, std


def center_batch(embedding_tensor: t.Tensor, mean: t.Tensor, std: t.Tensor):
    # center the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    # using the computed mean and std
    centered = (embedding_tensor - mean) / (std + 1e-12)

    return centered


def length_norm_batch(embedding_tensor: t.Tensor):
    # length normalize the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    return normalize(embedding_tensor, dim=1)
