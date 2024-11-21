################################################################################
#
# Data module for training a speaker recognition system on
# VoxCeleb1 and/or Voxceleb2.
#
# Author(s): Nik Vaessen
################################################################################

import itertools
import json
import pathlib
import shutil
import random
import re
import multiprocessing
import subprocess
import warnings

from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Union,
    List,
    Optional,
    Set,
    Tuple,
    Callable,
    Generator,
    DefaultDict,
    Dict,   
)
import torch
import torchaudio

import webdataset as wds

from torch.utils.data import DataLoader

from src.data.modules.Language.language_data_module import LanguageLightningDataModule
from src.data.preprocess.base import Preprocessor
from src.data.modules.Language.training_batch_language import (
    LanguageClassificationDataSample,
    LanguageClassificationDataBatch,
    PairedLanguageClassificationDataBatch,
    PairedLanguageClassificationDataSample,
    BatchDebugInfo,
)
from src.data.common import (
    WebDataSetShardConfig,
    SpeakerDataLoaderConfig,
)
from src.config_util import CastingConfig
from src.evaluation.speaker.speaker_recognition_evaluator import EvaluationPair
from src.util import (
    remove_directory,
    extract_archive,
)


################################################################################
# the overarching data module of VoxCeleb1 and/or 2


@dataclass
class OLR2020DataModuleConfig(CastingConfig):
    # select which parts of the data to use
    # use_voxceleb1_dev: bool
    # use_voxceleb1_test: bool
    # use_voxceleb2_dev: bool
    # use_voxceleb2_test: bool
    # all_voxceleb1_is_test_set: bool

    # define data kind
    has_train: bool
    has_val: bool
    has_test: bool

    # where to write data to
    test_split_file_path: pathlib.Path
    shards_folder: pathlib.Path
    extraction_folder: pathlib.Path
    data_path:pathlib.Path

    # determine train/val split
    # `equal` mode means each speaker is in both train and val split
    # `different` mode means intersection of speakers in train and val is empty
    split_mode: str  # one of 'equal`, `different`
    eer_validation_pairs: int
    eer_test_pairs: int
    # num_val_speakers: int  # only used when split_mode=different
    train_val_ratio: float  # only used when split_mode=equal

    # settings related to how data is written to shards
    sequential_same_speaker_samples: int  # num back-to-back samples from same speaker
    min_unique_speakers_per_shard: int
    discard_partial_shards: bool  # last shard might not be completely filled

    # how to collate the data when creating a batch
    # one of `default` (assumes same size) or
    # `pad_right` (add 0's so dimensions become equal)
    train_collate_fn: str
    val_collate_fn: str
    test_collate_fn: str

    # whether to add "original" raw data to each batch, useful for debugging
    add_batch_debug_info: bool

    # limit to `x` batches, useful for debugging
    limit_samples: int

    # in what way to create batches
    # one of "categorical", "categorical_triplets", "pairwise_categorical",
    batch_processing_mode: str

    # only required/used when batch_processing_mode == pairwise_categorical
    pos_neg_training_batch_ratio: float = 0.5
    yield_limit: Optional[int] = None


class OLR2020DataModule(LanguageLightningDataModule):
    def __init__(
        self,
        cfg: OLR2020DataModuleConfig,
        shard_cfg: WebDataSetShardConfig,
        dl_cfg: SpeakerDataLoaderConfig,
        train_pipeline: List[Preprocessor],
        val_pipeline: List[Preprocessor],
        test_pipeline: List[Preprocessor],
        test_path: pathlib.Path,
    ):
        super().__init__()

        # configuration
        self.cfg = cfg
        self.shard_cfg = shard_cfg
        self.dl_cfg = dl_cfg

        if cfg.batch_processing_mode not in [
            "categorical",
            "categorical_triplets",
            "pairwise_categorical",
        ]:
            raise ValueError(
                f"unknown batch processing mode {cfg.batch_processing_mode}"
            )

        # pipelines to apply to each subset
        self.train_pipeline = train_pipeline
        self.val_pipeline = val_pipeline
        self.test_pipeline = test_pipeline

        # locations where to save/write data to (once) and read data from
        self.train_shards_folder = self.cfg.shards_folder / "train"
        self.val_shards_folder = self.cfg.shards_folder / "val"
        #self.test_shards_folder = self.cfg.shards_folder / "test/task2"
        # self.test_shards_folder = pathlib.Path("/mnt/g813_u2/shards/test/task1")
        if(test_path==pathlib.Path("")):
            # self.test_shards_folder = pathlib.Path("/mnt/g813_u2/shards/test/task1")
            self.test_shards_folder = self.cfg.shards_folder / "test" / "task1"
        else:
            self.test_shards_folder=test_path
        # self.test_shards_folder_task1 = self.cfg.shards_folder / "test/task1"
        # self.test_shards_folder_task2 = self.cfg.shards_folder / "test/task2"
        # self.test_shards_folder_task3 = self.cfg.shards_folder / "test/task3"

        self.validation_pairs_file = self.val_shards_folder / "validation_pairs.txt"
        self.test_pairs_file = self.test_shards_folder / "test_pairs.txt"
        # self.test_pairs_file_task1 = self.test_shards_folder_task1 / "test_pairs.txt"
        # self.test_pairs_file_task2 = self.test_shards_folder_task2 / "test_pairs.txt"
        # self.test_pairs_file_task3 = self.test_shards_folder_task3 / "test_pairs.txt"

        # values set in self#setup()
        self.train_ds: torch.utils.data.Dataset = None
        self.val_ds: torch.utils.data.Dataset = None
        self.test_ds: torch.utils.data.Dataset = None
        self._num_languages: int = None
        self.num_train_shards: int = None
        self.num_val_shards: int = None
        self.num_test_shards: int = None

    @property
    def num_language(self) -> int:
        return self._num_languages

    @property
    def val_pairs(self) -> List[EvaluationPair]:
        return self.val_evaluation_pairs()

    @property
    def test_pairs(self) -> List[EvaluationPair]:
        return self.test_evaluation_pairs()
    
    def prepare_data(self):
        # skip preparation if it's already done
        if self._is_prepared():
            return
        data_train_path=self.cfg.data_path / "train"
        data_test_path=self.cfg.data_path / "/test/task1"
        # define paths which will be used throughout this method
        #存放临时数据的路径：/g813_u1/mnt/g813_u2/temp_OLR2020
        extract_folder = self.cfg.extraction_folder
        extract_folder.mkdir(exist_ok=True, parents=True)

        train_split_folder = extract_folder / "train"
        val_split_folder = extract_folder / "val"
        test_split_folder =extract_folder / "test"
        # test_split_folder_task1 = extract_folder / "test/task1"
        # test_split_folder_task2 = extract_folder / "test/task2"
        # test_split_folder_task3 = extract_folder / "test/task3"


        # step 2: split extracted data into train and test set
        if not (extract_folder / ".split").exists():
            # all_language_ids = set()

            if self.cfg.split_mode == "equal":
                all_language_ids =_create_train_val_split_equal_num_speakers(
                    data_folder_path=data_train_path,
                    train_folder_path=train_split_folder,
                    validation_folder_path=val_split_folder,
                    val_ratio=1 - self.cfg.train_val_ratio,
                    overwrite_existing_validation_folder=True,
                )
            elif self.cfg.split_mode == "different":
                all_language_ids =_create_train_val_split_diff_num_speakers(
                    data_folder_path=data_train_path,
                    train_folder_path=train_split_folder,
                    validation_folder_path=val_split_folder,
                    val_ratio=1 - self.cfg.train_val_ratio,
                    overwrite_existing_validation_folder=True,
                )
            else:
                raise ValueError(f"unknown value {self.cfg.split_mode=}")
            

            
            # _create_test_split(
            #     data_test_path,
            #     self.cfg.test_split_file_path,
            #     test_split_folder,
            #     all_language_ids,
            # )

            (extract_folder / ".split").touch()

        # step 4: shard each split
        if self.cfg.has_train:
            print("writing training shards")
            write_shards(
                data_folder_path=train_split_folder,
                shards_path=self.train_shards_folder,
                compress_in_place=self.shard_cfg.use_gzip_compression,
                shard_name_pattern="train_shard_{idx:06d}",
                samples_per_shard=self.shard_cfg.samples_per_shard,
                sequential_same_speaker_samples=self.cfg.sequential_same_speaker_samples,
                min_unique_speakers_per_shard=self.cfg.min_unique_speakers_per_shard,
                ensure_all_data_in_shards=self.cfg.split_mode == "equal",
                discard_partial_shards=self.cfg.discard_partial_shards,
            )

        if self.cfg.has_val:
            print("writing validation shards")
            write_shards(
                data_folder_path=val_split_folder,
                shards_path=self.val_shards_folder,
                compress_in_place=self.shard_cfg.use_gzip_compression,
                shard_name_pattern="val_shard_{idx:06d}",
                samples_per_shard=5000,  # all data in one shard
                sequential_same_speaker_samples=1,
                min_unique_speakers_per_shard=1,
                ensure_all_data_in_shards=True,
            )

        if self.cfg.has_test:
            print("writing task1's test shards ")
            write_shards(
                data_folder_path=test_split_folder,
                shards_path=self.test_shards_folder,
                compress_in_place=self.shard_cfg.use_gzip_compression,
                shard_name_pattern="test_shard_{idx:06d}",
                samples_per_shard=5000,  # all data in one shard
                sequential_same_speaker_samples=1,
                min_unique_speakers_per_shard=1,
                ensure_all_data_in_shards=True,
            )
            # print("writing task2's test shards")
            # write_shards(
            #     data_folder_path=test_split_folder_task2,
            #     shards_path=self.test_shards_folder_task2,
            #     compress_in_place=self.shard_cfg.use_gzip_compression,
            #     shard_name_pattern="test_shard_{idx:06d}",
            #     samples_per_shard=5000,  # all data in one shard
            #     sequential_same_speaker_samples=1,
            #     min_unique_speakers_per_shard=1,
            #     ensure_all_data_in_shards=True,
            # )
        #     print("writing task3's test shards")
        #     write_shards(
        #         data_folder_path=test_split_folder_task3,
        #         shards_path=self.test_shards_folder_task3,
        #         compress_in_place=self.shard_cfg.use_gzip_compression,
        #         shard_name_pattern="test_shard_{idx:06d}",
        #         samples_per_shard=5000,  # all data in one shard
        #         sequential_same_speaker_samples=1,
        #         min_unique_speakers_per_shard=1,
        #         ensure_all_data_in_shards=True,
        #     )

        # step 5: validate the train/val/test split
        self._validate_shard_meta()

        # step 6: create validation pairs (for EER calculating)
        if self.cfg.has_train and self.cfg.has_val:
            print(f"generating {self.cfg.eer_validation_pairs} validation pairs...")
            self._generate_pairs(save_path=self.validation_pairs_file,
                                 get_meta=self._get_val_meta,
                                 get_num_pairs=self.cfg.eer_validation_pairs,
                                 )
        if self.cfg.has_test:
            print(f"generating {self.cfg.eer_test_pairs} test pairs...")
            self._generate_pairs(save_path=self.test_pairs_file,
                                 get_meta=self._get_test_meta,
                                 get_num_pairs=self.cfg.eer_test_pairs,
                                 )

        # if self.cfg.has_test:
        #     print(f"generating {self.cfg.eer_test_pairs} task1's test pairs...")
        #     self._generate_pairs(save_path=self.test_pairs_file_task1,
        #                          get_meta=self._get_test_meta_task1,
        #                          get_num_pairs=self.cfg.eer_test_pairs,
        #                          )
        #     print(f"generating {self.cfg.eer_test_pairs} task2's test pairs...")
        #     self._generate_pairs(save_path=self.test_pairs_file_task2,
        #                          get_meta=self._get_test_meta_task2,
        #                          get_num_pairs=self.cfg.eer_test_pairs,
        #                          )
        # #     print(f"generating {self.cfg.eer_test_pairs} task3's test pairs...")
        #     self._generate_pairs(save_path=self.test_pairs_file_task3,
        #                          get_meta=self._get_test_meta_task3,
        #                          get_num_pairs=self.cfg.eer_test_pairs,
        #                          )
        # step 6: clean up
        self._set_is_prepared()
        # remove_directory(extract_folder)

    def _validate_shard_meta(self):
        train_meta = self._get_train_meta()
        val_meta = self._get_val_meta()
        test_meta = self._get_test_meta()
        # test_meta_task1 = self._get_test_meta_task1()
        # test_meta_task2 = self._get_test_meta_task2()
        # test_meta_task3 = self._get_test_meta_task3()

        if self.cfg.has_train and self.cfg.has_val and self.cfg.has_test:
            # same speakers in train/val
            assert train_meta["num_languages"] == val_meta["num_languages"]

            # same speaker labels in train/val
            assert train_meta["language_id_to_idx"] == val_meta["language_id_to_idx"]

            # empty intersection of samples in train/val/test
            train_ids = set(train_meta["sample_ids"])
            val_ids = set(val_meta["sample_ids"])
            test_ids = set(test_meta["sample_ids"])
            # test_ids_task1=set(test_meta_task1["sample_ids"])
            # test_ids_task2=set(test_meta_task2["sample_ids"])
            # test_ids_task3=set(test_meta_task3["sample_ids"]) 


            intersection = train_ids.intersection(val_ids).intersection(test_ids)
            assert len(intersection) == 0
            # intersection = train_ids.intersection(val_ids).intersection(test_ids_task1)
            # assert len(intersection) == 0
            # intersection = train_ids.intersection(val_ids).intersection(test_ids_task2)
            # assert len(intersection) == 0
            # intersection = train_ids.intersection(val_ids).intersection(test_ids_task3)
            # assert len(intersection) == 0

    def _get_shard_info_file(self):
        return self.cfg.shards_folder / ".info"

    def _is_prepared(self):
        return self._get_shard_info_file().exists()

    def _set_is_prepared(self):
        with self._get_shard_info_file().open("w") as f:
            json.dump(
                {
                    "test_file": self.cfg.test_split_file_path.name,
                    "samples_per_shard": self.shard_cfg.samples_per_shard,
                    "use_gzip_compression": self.shard_cfg.use_gzip_compression,
                    "train_ratio": self.cfg.train_val_ratio,
                },
                f,
            )

    def _get_train_meta(self):
        file = self.train_shards_folder / "meta.json"

        if not file.exists():
            return None

        with file.open("r") as f:
            return json.load(f)

    def _get_val_meta(self):
        file = self.val_shards_folder / "meta.json"

        if not file.exists():
            return None

        with file.open("r") as f:
            return json.load(f)

    def _get_test_meta(self):
        file = self.test_shards_folder / "meta.json"

        if not file.exists():
            return None

        with file.open("r") as f:
            return json.load(f)
    
    # def _get_test_meta_task1(self):
    #     file = self.test_shards_folder_task1 / "meta.json"

    #     if not file.exists():
    #         return None

    #     with file.open("r") as f:
    #         return json.load(f)
    # def _get_test_meta_task2(self):
    #     file = self.test_shards_folder_task2 / "meta.json"

    #     if not file.exists():
    #         return None

    #     with file.open("r") as f:
    #         return json.load(f)
    # def _get_test_meta_task3(self):
    #     file = self.test_shards_folder_task3 / "meta.json"

    #     if not file.exists():
    #         return None

    #     with file.open("r") as f:
    #         return json.load(f)

    @staticmethod
    def _find_shard_paths(folder: pathlib.Path, pattern, only_1_shard=False):
        shards = sorted([str(f) for f in folder.glob(pattern) if f.is_file()])

        if len(shards) == 0:
            warnings.warn(
                f"unable to find any shards in {folder} matching pattern `{pattern}`"
            )
            return shards

        if only_1_shard:
            return shards[:1]
        else:
            return shards
        

    def get_train_ds(self):
        return self.train_ds
    def get_val_ds(self):
        return self.val_ds
    def get_test_ds(self):
        return self.test_ds
    
    def setup(self, stage: Optional[str] = None):
        # validate meta information of shards
        self._validate_shard_meta()

        train_meta = self._get_train_meta()
        val_meta = self._get_val_meta()
        test_meta = self._get_test_meta()

        # create train dataset
        if self.cfg.has_train:
            training_shards_paths = self._find_shard_paths(
                self.train_shards_folder,
                "train_shard_*.tar*",
                only_1_shard=True if self.cfg.limit_samples > 0 else False,
            )
            self.num_train_shards = len(training_shards_paths)
            
            self.train_ds: wds.Processor = (
                wds.WebDataset(
                    urls=training_shards_paths,
                    # due to the preprocessing pipes we cannot determine
                    # what the actual length is
                    length=float("inf"),
                )
                .decode("pil")
                .pipe(self._pipe_to_classification_data_sample())
            )
        for p in self.train_pipeline:
            self.train_ds = self.train_ds.pipe(self._pipe_preprocessors(p))
        
        if self.cfg.batch_processing_mode == "categorical_triplets":
            train_batch_processor = TripletSpeakerBatchProcessor(
                max_batch_size=self.dl_cfg.train_batch_size,
                max_queue_size=self.shard_cfg.queue_size,
                collate_fn=self._determine_collate_fn(self.cfg.train_collate_fn),
                ensure_all_samples_seen=False,
            )
        elif self.cfg.batch_processing_mode == "pairwise_categorical":
            train_batch_processor = PairedBatchProcessor(
                batch_size=self.dl_cfg.train_batch_size,
                mode="generate",
                collate_fn=self._determine_collate_fn(self.cfg.train_collate_fn),
                sequential_same_speaker_samples=self.cfg.sequential_same_speaker_samples,
                pos_neg_training_batch_ratio=self.cfg.pos_neg_training_batch_ratio,
                fixed_random_seed=True if self.cfg.limit_samples > 0 else False,
                max_queue_size=self.shard_cfg.queue_size,
                yield_limit=self.cfg.yield_limit,
            )
        else:
            train_batch_processor = BatchProcessor(
                max_batch_size=self.dl_cfg.train_batch_size,
                max_queue_size=self.shard_cfg.queue_size,
                collate_fn=self._determine_collate_fn(self.cfg.train_collate_fn),
            )

        self.train_ds = self.train_ds.pipe(train_batch_processor)

        if self.cfg.limit_samples > 0:
            self.train_ds = self.train_ds.slice(self.cfg.limit_samples)
        
       

        # create val dataset
        if self.cfg.has_val:
            validation_shards_paths = self._find_shard_paths(
                self.val_shards_folder, "val_shard_*.tar*"
            )
            self.num_val_shards = len(validation_shards_paths)
            
            self.val_ds = (
                wds.WebDataset(
                    urls=validation_shards_paths,
                    length=float("inf"),
                )
                .decode("pil")
                .pipe(
                    self._pipe_to_classification_data_sample(),
                )
            )
            for p in self.val_pipeline:
                self.val_ds = self.val_ds.pipe(self._pipe_preprocessors(p))

            if self.cfg.batch_processing_mode == "categorical_triplets":
                val_batch_processor = TripletSpeakerBatchProcessor(
                    max_batch_size=self.dl_cfg.val_batch_size,
                    max_queue_size=self.shard_cfg.queue_size,
                    collate_fn=self._determine_collate_fn(self.cfg.val_collate_fn),
                    ensure_all_samples_seen=True,
                )
            elif self.cfg.batch_processing_mode == "pairwise_categorical":
                val_batch_processor = PairedBatchProcessor(
                    batch_size=self.dl_cfg.val_batch_size,
                    mode="reproduce",
                    collate_fn=self._determine_collate_fn(self.cfg.val_collate_fn),
                    sequential_same_speaker_samples=self.cfg.sequential_same_speaker_samples,
                    pairs=self.val_pairs,
                    max_queue_size=self.shard_cfg.queue_size,
                )
            else:
                val_batch_processor = BatchProcessor(
                    max_batch_size=self.dl_cfg.val_batch_size,
                    max_queue_size=self.shard_cfg.queue_size,
                    collate_fn=self._determine_collate_fn(self.cfg.val_collate_fn),
                )

            self.val_ds = self.val_ds.pipe(val_batch_processor)


        # create test dataset
        if self.cfg.has_test:
            test_shards_paths = self._find_shard_paths(
                self.test_shards_folder, "test_shard_*.tar*"
            )
            self.num_test_shards = len(test_shards_paths)

            self.test_ds = (
                wds.WebDataset(
                    urls=test_shards_paths,
                    length=float("inf"),
                )
                .decode("pil")
                .pipe(self._pipe_to_classification_data_sample())
            )

            for p in self.test_pipeline:
                self.test_ds = self.test_ds.pipe(self._pipe_preprocessors(p))

            if self.cfg.batch_processing_mode == "pairwise_categorical":
                test_batch_processor = PairedBatchProcessor(
                    batch_size=self.dl_cfg.test_batch_size,
                    mode="reproduce",
                    collate_fn=self._determine_collate_fn(self.cfg.test_collate_fn),
                    pairs=self.test_pairs,
                    sequential_same_speaker_samples=self.cfg.sequential_same_speaker_samples,
                    max_queue_size=self.shard_cfg.queue_size,
                )
            else:
                test_batch_processor = BatchProcessor(
                    max_batch_size=self.dl_cfg.test_batch_size,
                    max_queue_size=self.shard_cfg.queue_size,
                    collate_fn=self._determine_collate_fn(self.cfg.test_collate_fn),
                )

            self.test_ds = self.test_ds.pipe(test_batch_processor)

            
        # validate number of speakers
        if self.cfg.has_train and self.cfg.has_val:
            self._num_languages = self._get_train_meta()["num_languages"]

            if self.cfg.split_mode == "equal":
                assert (
                    self._num_languages
                    == self._get_train_meta()["num_languages"]
                    == self._get_val_meta()["num_languages"]
                )

    def _pipe_to_classification_data_sample(self):
        def apply_pipeline(x: dict) -> LanguageClassificationDataSample:
            # x is a dict with keys 'wav.pyd` and `meta.json`
            key: str = x["__key__"]
            audio_sample: torch.Tensor = x["wav.pyd"]
            spk_label: int = x["meta.json"]["language_id_idx"]

            # check for NaN value in audio sample
            if torch.any(torch.isnan(audio_sample)):
                raise ValueError(f"NaN value in audio sample of {key=}")

            return LanguageClassificationDataSample(
                key=key,
                ground_truth=spk_label,
                network_input=audio_sample,
                side_info=BatchDebugInfo(
                    original_tensor=audio_sample,
                    pipeline_progress=[],
                    meta=x["meta.json"],
                )
                if self.cfg.add_batch_debug_info
                else None,
            )

        def pipe_fn(data_iterator):
            for x in data_iterator:
                yield apply_pipeline(x)

        return pipe_fn

    @staticmethod
    def _pipe_preprocessors(preprocessor: Preprocessor):
        def pipe_fn(data_iter):
            for x in data_iter:
                sample = preprocessor.process(x)

                if isinstance(sample, list):
                    for y in sample:
                        yield y
                else:
                    yield sample

        return pipe_fn

    def summary(self):
        train_meta = self._get_train_meta()
        val_meta = self._get_val_meta()
        test_meta = self._get_test_meta()

        print("### OLR2020DataModule statistics ###")
        if self.cfg.has_train:
            print(f"training samples: {train_meta['num_samples']}")
        if self.cfg.has_val:
            print(f"validation samples: {val_meta['num_samples']}")
        if self.cfg.has_test:
            print(f"test samples: {test_meta['num_samples']}")

        if self.cfg.has_train:
            print(f"training languages: {train_meta['num_languages']}")
        if self.cfg.has_val:
            print(f"validation languages: {val_meta['num_languages']}")
        if self.cfg.has_test:
            print(f"test languages: {test_meta['num_languages']}")

    def _generate_pairs(self,save_path: pathlib.Path,get_meta , get_num_pairs: int):
        # randomly generate up to the same amount of validation pairs as
        # test pairs for calculating EER on validation data
        meta = get_meta()

        # determine amount of pairs to generate
        num_validation_pairs = get_num_pairs
        positive_samples = num_validation_pairs // 2
        negative_samples = num_validation_pairs - positive_samples

        # data useful for generating pairs
        all_language_ids = set(meta["language_ids"])

        # randomly sample positive pairs
        positive_pairs = []
        # sorting here should ensure that positive pairs are equal with
        # same random seed
        speaker_id_queue = sorted(list(all_language_ids))

        while len(positive_pairs) < positive_samples:
            if len(speaker_id_queue) == 0:
                raise ValueError(
                    f"not enough possible pairings to generate "
                    f"{positive_samples} positive pairs"
                )

            # cycle between each speaker until we have filled all positive samples
            spk_id = speaker_id_queue.pop()
            speaker_id_queue.insert(0, spk_id)

            # randomly select 2 files from this speaker
            # they shouldn't be equal and we shouldn't have this sample already
            speaker_id_samples = meta["sample_ids_per_language"][spk_id]
            random.shuffle(speaker_id_samples)

            original_length = len(positive_pairs)
            for sample1_key, sample2_key in itertools.combinations(
                speaker_id_samples, r=2
            ):
                # check for failure case of this speaker before appending
                if (
                    sample1_key != sample2_key
                    and (sample1_key, sample2_key) not in positive_pairs
                    and (sample2_key, sample1_key) not in positive_pairs
                ):
                    positive_pairs.append((sample1_key, sample2_key))
                    break

            # if we no break happened no more combinations for this speaker
            # can be added and it should be removed from queue
            if len(positive_pairs) == original_length:
                speaker_id_queue.remove(spk_id)

        # randomly sample negative pairs
        negative_pairs = []
        count_map_all_speakers = {k: 0 for k in set(all_language_ids)}

        fails = 0
        while len(negative_pairs) < negative_samples:
            if fails > 100:
                raise ValueError(
                    f"unable to generate {negative_samples} negative pairs"
                )

            # sorting here should ensure same pairings between different
            # runs with same random seed
            speakers_to_choose_from, num_samples_per_speaker = zip(
                *[(k, v) for k, v in sorted(count_map_all_speakers.items())]
            )
            speakers_to_choose_from = list(speakers_to_choose_from)

            # inverse num_samples_per_speaker such that they can act as weights
            # to choice 2 speakers with least pairs yet
            total_num_samples = 2 * len(negative_pairs)
            num_samples_per_speaker = [
                total_num_samples - n + 1 for n in num_samples_per_speaker
            ]

            # randomly select 2 speakers
            spk_id1 = random.choices(
                population=speakers_to_choose_from, weights=num_samples_per_speaker, k=1
            )[0]

            # print(speakers_to_choose_from, type(speakers_to_choose_from))
            # print(num_samples_per_speaker, type(num_samples_per_speaker))
            # print(spk_id1, type(spk_id1))
            spk_id1_idx = speakers_to_choose_from.index(spk_id1)
            # print(spk_id1_idx, type(spk_id1_idx))
            speakers_to_choose_from.pop(spk_id1_idx)
            num_samples_per_speaker.pop(spk_id1_idx)

            spk_id2 = random.choices(
                population=speakers_to_choose_from, weights=num_samples_per_speaker, k=1
            )[0]

            assert spk_id1 != spk_id2

            # cycle through each combination of 2 different speakers
            spk1_samples = meta["sample_ids_per_language"][spk_id1]
            spk2_samples = meta["sample_ids_per_language"][spk_id2]

            random.shuffle(spk1_samples)
            random.shuffle(spk2_samples)

            # add first non-seen pair
            original_length = len(negative_pairs)
            for sample1_key, sample2_key in itertools.product(
                spk1_samples, spk2_samples
            ):
                # check for collision with previous samples, otherwise store
                pair = (sample1_key, sample2_key)

                if (
                    pair not in negative_pairs
                    and (sample2_key, sample1_key) not in negative_pairs
                ):
                    negative_pairs.append(pair)
                    count_map_all_speakers[spk_id1] += 1
                    count_map_all_speakers[spk_id2] += 1
                    break

            if original_length == len(negative_pairs):
                fails += 1

        # write positive and negative pairs to a file
        with save_path.open("w") as f:
            count = 0

            while not (len(positive_pairs) == len(negative_pairs) == 0):
                count += 1

                # alternate between positive and negative sample
                if count % 2 == 0:
                    if len(positive_pairs) == 0:
                        continue
                    else:
                        pair = positive_pairs.pop()
                        gt = 1
                else:
                    if len(negative_pairs) == 0:
                        continue
                    else:
                        pair = negative_pairs.pop()
                        gt = 0

                # write pair
                path1, path2 = pair
                path1 += ".wav"
                path2 += ".wav"
                f.write(f"{gt} {path1} {path2}\n")

    def val_evaluation_pairs(self) -> List[EvaluationPair]:
        if self.cfg.has_val:
            return load_evaluation_pairs(self.validation_pairs_file)
        else:
            return []
    
    def test_evaluation_pairs(self) -> List[EvaluationPair]:
        #return load_evaluation_pairs(self.cfg.test_split_file_path)
        return load_evaluation_pairs(self.test_pairs_file)

    def train_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=None,
            num_workers=min(self.dl_cfg.num_workers, self.num_train_shards),
            pin_memory=self.dl_cfg.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=None,
            num_workers=min(self.dl_cfg.num_workers, self.num_val_shards),
            pin_memory=self.dl_cfg.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=None,
            num_workers=min(self.dl_cfg.num_workers, self.num_test_shards),
            pin_memory=self.dl_cfg.pin_memory,
        )

    def _determine_collate_fn(self, name: str):
        if name == "default":
            if self.cfg.batch_processing_mode == "pairwise_categorical":
                return PairedLanguageClassificationDataBatch.default_collate_fn
            else:
                return LanguageClassificationDataBatch.default_collate_fn
        elif name == "pad_right":
            if self.cfg.batch_processing_mode == "pairwise_categorical":
                return PairedLanguageClassificationDataBatch.pad_right_collate_fn
            else:
                return LanguageClassificationDataBatch.pad_right_collate_fn
        else:
            raise ValueError(f"cannot determine collate_fn {name}")


################################################################################
# pipe which makes sure that each batch contains at least 2 samples of each speaker


class BatchProcessor:
    def __init__(
        self,
        max_batch_size: int,
        max_queue_size: int,
        collate_fn: Callable[
            [List[LanguageClassificationDataSample]], LanguageClassificationDataBatch
        ],
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size needs to be a positive integer")

        if max_queue_size <= 0 or max_queue_size < max_batch_size:
            raise ValueError(
                f"queue size needs to be >= "
                f"max(1, {max_batch_size=})={max(1, max_batch_size)},"
                f" while given value is {max_queue_size}"
            )

        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.collate_fn = collate_fn

        # state
        self.queue: List[LanguageClassificationDataSample] = []

    def __call__(
        self, sample_iterator: Generator[LanguageClassificationDataSample, None, None]
    ) -> Generator[LanguageClassificationDataBatch, None, None]:
        self.queue.clear()

        for sample in sample_iterator:
            if not isinstance(sample, LanguageClassificationDataSample):
                raise ValueError(
                    f"batch is expected to be of type {LanguageClassificationDataSample}"
                )

            self.queue.append(sample)

            if len(self.queue) >= self.max_queue_size:
                yield self._get_batch()

        while len(self.queue) >= 1:
            yield self._get_batch()

        self.queue.clear()

    def _get_batch(self) -> LanguageClassificationDataBatch:
        if len(self.queue) == 0:
            raise ValueError("cannot get a batch without any samples")

        batch_samples = []

        while len(batch_samples) < self.max_batch_size and len(self.queue) >= 1:
            batch_samples.append(self.queue.pop(random.randint(0, len(self.queue) - 1)))

        return self.collate_fn(batch_samples)


class TripletSpeakerBatchProcessor:
    def __init__(
        self,
        max_batch_size: int,
        max_queue_size: int,
        collate_fn: Callable[
            [List[LanguageClassificationDataSample]], LanguageClassificationDataBatch
        ],
        ensure_all_samples_seen: bool = False,  # needs to be true to calculate val EER
    ):
        if max_batch_size % 2 == 1:
            raise ValueError("batch size needs to be even to allow for triplets")

        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.collate_fn = collate_fn
        self.ensure_all_samples_seen = ensure_all_samples_seen

        # state
        self.map_size: int = 0
        self.sample_keys = set()
        self.speaker_id_sample_map: DefaultDict[
            int, List[LanguageClassificationDataSample]
        ] = defaultdict(list)
        self.valid_keys: Set[int] = set()
        self.invalid_keys: Set[int] = set()

    def __call__(
        self, sample_iterator: Generator[LanguageClassificationDataSample, None, None]
    ) -> Generator[LanguageClassificationDataBatch, None, None]:
        self._reset()

        for sample in sample_iterator:
            if not isinstance(sample, LanguageClassificationDataSample):
                raise ValueError(
                    f"batch is expected to be of type {LanguageClassificationDataSample}"
                )

            self._add_sample(sample)

            if self.is_triplet_batch_possible():
                yield self._get_batch()

            if self.map_size == self.max_queue_size * 2:
                raise ValueError(
                    "queue size is has exceeded limit while unable to to ensure"
                    " triplet"
                )

        while self.is_triplet_batch_possible():
            yield self._get_batch()

        print(f"discarding {self.map_size} samples due to no triplet")

        if self.ensure_all_samples_seen:
            while self.map_size >= 1:
                b = self._get_batch()
                if b is not None:
                    yield b

        self._reset()

    def _reset(self):
        self.map_size: int = 0
        self.speaker_id_sample_map: DefaultDict[
            int, List[LanguageClassificationDataSample]
        ] = defaultdict(list)
        self.valid_keys = set()
        self.invalid_keys = set()
        self.sample_keys.clear()

    def _add_sample(self, sample: LanguageClassificationDataSample):
        self.map_size += 1

        if sample.key in self.sample_keys:
            raise ValueError("this sample has already been loaded")

        self.sample_keys.add(sample.key)

        sample_list = self.speaker_id_sample_map[sample.ground_truth]
        sample_list.append(sample)

        if len(sample_list) >= 2:
            self.invalid_keys.discard(sample.ground_truth)
            self.valid_keys.add(sample.ground_truth)
        else:
            self.invalid_keys.add(sample.ground_truth)

    def _retrieve_triplet_batch_samples(self) -> List[LanguageClassificationDataSample]:
        if not self.is_triplet_batch_possible():
            if not self.ensure_all_samples_seen:
                raise ValueError(
                    "cannot retrieve a batch of samples with current state"
                )
            else:
                return self._retrieve_batch_samples()

        # select up to max_batch_size // 2 keys from which to get the samples
        keys = random.sample(
            self.valid_keys, min(self.max_batch_size // 2, len(self.valid_keys))
        )

        batch_samples = []

        for k in keys:
            sample_list = self.speaker_id_sample_map[k]

            anchor = sample_list.pop(random.randint(0, len(sample_list) - 1))
            positive = sample_list.pop(random.randint(0, len(sample_list) - 1))

            batch_samples.append(anchor)
            batch_samples.append(positive)

            self.map_size -= 2
            if len(sample_list) < 2:
                self.valid_keys.discard(k)
                self.invalid_keys.add(k)

            if len(sample_list) == 0:
                self.invalid_keys.discard(k)
                del self.speaker_id_sample_map[k]

        return batch_samples

    def _retrieve_batch_samples(self):
        batch_samples = []

        while len(batch_samples) < self.max_batch_size and self.map_size >= 1:
            sample_list = None
            key = None

            for k in self.valid_keys:
                key = k
                sample_list = self.speaker_id_sample_map[k]

            if sample_list is None or key is None:
                for k in self.invalid_keys:
                    sample_list = self.speaker_id_sample_map[k]
                    key = None

                    if len(sample_list) >= 1:
                        key = k
                        break

            if sample_list is None or key is None or len(sample_list) == 0:
                raise ValueError("no valid sample")

            sample = sample_list.pop(0)
            batch_samples.append(sample)

            self.map_size -= 1
            if len(sample_list) < 2:
                self.valid_keys.discard(key)
                self.invalid_keys.add(key)
            if len(sample_list) == 0:
                self.invalid_keys.discard(key)
                del self.speaker_id_sample_map[key]

        return batch_samples

    def is_triplet_batch_possible(self) -> bool:
        return len(self.valid_keys) >= 2

    def _get_batch(self) -> LanguageClassificationDataBatch:
        if not self.ensure_all_samples_seen:
            if not self.is_triplet_batch_possible():
                raise ValueError(
                    "cannot get a batch while state does not allow for triplets"
                )
            if self.map_size <= 3:
                raise ValueError("cannot get a batch with less than 4 samples")

        batch_samples = self._retrieve_triplet_batch_samples()

        return self.collate_fn(batch_samples)


class PairedBatchProcessor:
    def __init__(
        self,
        batch_size: int,
        max_queue_size: int,
        mode: str,  # one of 'generate` of `reproduce`
        sequential_same_speaker_samples: int,
        collate_fn: Callable[
            [List[PairedLanguageClassificationDataSample]],
            PairedLanguageClassificationDataBatch,
        ],
        pos_neg_training_batch_ratio: Optional[float] = None,
        pairs: Optional[List[EvaluationPair]] = None,
        fixed_random_seed: bool = False,  # handy for debugging
        yield_limit: Optional[int] = None,  # to prevent uneven data sizes in DDP mode,
    ):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.mode = mode
        self.collate_fn = collate_fn
        self.sequential_same_speaker_samples = sequential_same_speaker_samples
        self.fixed_random_seed = fixed_random_seed
        self.random_state = random.getstate()
        self.yield_limit = yield_limit

        if mode not in ["generate", "reproduce"]:
            raise ValueError(f"{mode=} should be `generate` or `reproduce`")

        if self.batch_size > self.max_queue_size:
            raise ValueError(
                f"cannot generate batches of size {batch_size} "
                f"with a max queue size of {max_queue_size}"
            )

        # parameters for generating training data
        if mode == "generate":
            if pos_neg_training_batch_ratio is not None:
                self.pos_neg_training_batch_ratio = pos_neg_training_batch_ratio

                if self.batch_size % sequential_same_speaker_samples != 0:
                    raise ValueError(
                        f"{batch_size=} must be divisible by "
                        f"{sequential_same_speaker_samples=}"
                    )
            else:
                raise ValueError(
                    f"when {mode=}{pos_neg_training_batch_ratio=} must not be None"
                )

        # parameters for reproducing val/test pairs
        if mode == "reproduce":
            if pairs is not None:
                self.pairs = pairs
            else:
                raise ValueError(f"when {mode=}, {type(pairs)=} must not be None")

    def __call__(
        self,
        sample_generator: Generator[LanguageClassificationDataSample, None, None],
    ) -> Generator[PairedLanguageClassificationDataBatch, None, None]:
        if self.mode == "generate":
            return self._generate_training_pairs(sample_generator)
        elif self.mode == "reproduce":
            return self._reproduce_evaluation_pairs(sample_generator, self.pairs)
        else:
            raise ValueError(f"unknown mode {self.mode=}")

    def _generate_training_pairs(
        self, generator: Generator[LanguageClassificationDataSample, None, None]
    ) -> Generator[PairedLanguageClassificationDataBatch, None, None]:
        if self.fixed_random_seed:
            random.setstate(self.random_state)

        yielded_samples = 0

        # randomly generate pairs until `pairing_generation_limit` pairs have
        # been yielded
        num_pos_batch = round(self.pos_neg_training_batch_ratio * self.batch_size)
        num_neg_batch = self.batch_size - num_pos_batch

        assert num_pos_batch + num_neg_batch == self.batch_size
        assert 0 <= num_neg_batch <= self.batch_size
        assert 0 <= num_pos_batch <= self.batch_size

        # create a queue from which to generate pairs
        queue = []
        max_queue_size = min(
            self.batch_size, (self.max_queue_size // self.batch_size) * self.batch_size
        )

        sequential_samples_left = self.sequential_same_speaker_samples
        for sample in generator:
            queue.append(sample)
            sequential_samples_left -= 1

            if sequential_samples_left > 0:
                continue
            else:
                sequential_samples_left = self.sequential_same_speaker_samples

            if len(queue) >= max_queue_size:
                potential_batch = self._generate_paired_batch(
                    queue=queue,
                    batch_size=self.batch_size,
                    num_pos_samples=num_pos_batch,
                    num_neg_samples=num_neg_batch,
                    num_sequential_speaker_samples=self.sequential_same_speaker_samples,
                )

                if potential_batch is not None:
                    yield self.collate_fn(potential_batch)
                    yielded_samples += self.batch_size
                else:
                    raise ValueError(
                        "cannot yield batch while data is still being loaded"
                    )

                if self.yield_limit is not None and yielded_samples >= self.yield_limit:
                    break

        out_of_data = False
        while len(queue) > 0:
            if out_of_data:
                raise ValueError("queue not empty while out of data")

            potential_batch = self._generate_paired_batch(
                queue=queue,
                batch_size=self.batch_size,
                num_pos_samples=num_pos_batch,
                num_neg_samples=num_neg_batch,
                num_sequential_speaker_samples=self.sequential_same_speaker_samples,
            )

            if potential_batch is not None:
                yield self.collate_fn(potential_batch)
                yielded_samples += self.batch_size
            else:
                out_of_data = True

            if self.yield_limit is not None and yielded_samples >= self.yield_limit:
                break

    @staticmethod
    def _generate_paired_batch(
        queue: List[LanguageClassificationDataSample],
        batch_size: int,
        num_pos_samples: int,
        num_neg_samples: int,
        num_sequential_speaker_samples: int,
    ) -> Union[List[PairedLanguageClassificationDataSample], None]:
        if len(queue) < batch_size:
            queue.clear()
            return

        # create a mapping of each sample for each speaker
        speaker_map = defaultdict(list)

        for sample in queue:
            speaker_map[sample.ground_truth].append(sample)

        # assert each speaker has the expected number of samples
        # note it's possible to have the same speaker more than once
        assert all(
            [
                len(lst) % num_sequential_speaker_samples == 0
                for lst in speaker_map.values()
            ]
        )

        # determine the collection of samples to use for this batch
        speaker_ids = [k for k in speaker_map.keys()]
        num_samples = [len(v) for v in speaker_map.values()]
        num_samples_weights = [2 ** v for v in num_samples]
        num_speakers_in_batch = batch_size // num_sequential_speaker_samples

        if sum(num_samples) < batch_size:
            raise ValueError(
                "not enough speakers to generate paired batch.\n"
                f"{len(queue)=}\n"
                f"{speaker_ids=}\n"
                f"{num_samples=}\n"
                f"{num_speakers_in_batch=}\n"
            )

        batch_speaker_ids = []
        while len(batch_speaker_ids) < num_speakers_in_batch and len(speaker_ids) > 0:
            choice = random.choices(
                population=speaker_ids,
                weights=num_samples_weights,
                k=1,
            )[0]

            idx = speaker_ids.index(choice)
            batch_speaker_ids.append(choice)
            speaker_ids.pop(idx)
            num_samples.pop(idx)
            num_samples_weights.pop(idx)

        batch_map = defaultdict(list)

        for speaker_id in batch_speaker_ids:
            # randomly select `num_sequential_speaker_samples` from each speaker id
            sample_list = speaker_map[speaker_id]
            for _ in range(num_sequential_speaker_samples):
                batch_map[speaker_id].append(
                    sample_list.pop(random.randint(0, len(sample_list) - 1))
                )

        # generate `num_pos_samples` positive pairings from the samples
        # in this batch
        pos_pairs: List[PairedLanguageClassificationDataSample] = []

        fails = 0
        while len(pos_pairs) != num_pos_samples:
            if fails >= 100:
                raise ValueError("too many fails generating positive pairs")

            speaker_id = random.choice(batch_speaker_ids)
            sample_list = batch_map[speaker_id]

            if len(sample_list) < 2:
                fails += 1
                continue

            sample1, sample2 = random.sample(sample_list, 2)

            already_sampled = False
            for p in pos_pairs:
                # note pairing (sample2, sample1) is not equal
                if p.primary_key == sample1.key and p.secondary_key == sample2.key:
                    already_sampled = True

            if already_sampled:
                fails += 1
                continue

            sample_pos = PairedLanguageClassificationDataSample(
                primary_key=sample1.key,
                primary_input=sample1.network_input,
                secondary_key=sample2.key,
                secondary_input=sample2.network_input,
                ground_truth=1,
                side_info=None,
            )
            pos_pairs.append(sample_pos)

        # generate `num_neg_samples` negative pairings from the samples
        # in this batch
        neg_pairs: List[PairedLanguageClassificationDataSample] = []

        fails = 0
        while len(neg_pairs) != num_neg_samples:
            if fails >= 100:
                raise ValueError("too many fails generating negative pairs")

            speaker_id1, speaker_id2 = random.sample(batch_speaker_ids, 2)

            sample_list1 = batch_map[speaker_id1]
            sample_list2 = batch_map[speaker_id2]

            if len(sample_list1) < 1 or len(sample_list2) < 1:
                fails += 1

            sample1 = random.choice(sample_list1)
            sample2 = random.choice(sample_list2)

            already_sampled = False
            for p in pos_pairs:
                # note pairing (sample2, sample1) is not equal
                if p.primary_key == sample1.key and p.secondary_key == sample2.key:
                    already_sampled = True

            if already_sampled:
                fails += 1
                continue

            neg_pair = PairedLanguageClassificationDataSample(
                primary_key=sample1.key,
                primary_input=sample1.network_input,
                secondary_key=sample2.key,
                secondary_input=sample2.network_input,
                ground_truth=0,
                side_info=None,
            )
            neg_pairs.append(neg_pair)

        # remove the samples from the queue:
        for lst in batch_map.values():
            for s in lst:
                queue.remove(s)

        # generate and yield the batch
        pairs = []
        pairs.extend(pos_pairs)
        pairs.extend(neg_pairs)
        random.shuffle(pairs)

        return pairs

    def _reproduce_evaluation_pairs(
        self,
        generator: Generator[LanguageClassificationDataSample, None, None],
        pairs: List[EvaluationPair],
    ) -> Generator[PairedLanguageClassificationDataBatch, None, None]:
        # store all samples
        sample_dict: Dict[str, LanguageClassificationDataSample] = {}

        for sample in generator:
            sample_dict[sample.key] = sample

        if len(sample_dict) == 0:
            # if more workers than shards, generator might be empty
            return

        # now yield each pair
        batch_list = []
        for pair in pairs:
            primary_sample = sample_dict[pair.sample1_id]
            secondary_sample = sample_dict[pair.sample2_id]

            pair = PairedLanguageClassificationDataSample(
                primary_key=primary_sample.key,
                primary_input=primary_sample.network_input,
                secondary_key=secondary_sample.key,
                secondary_input=secondary_sample.network_input,
                ground_truth=1 if pair.same_speaker else 0,
                side_info=None,
            )
            batch_list.append(pair)

            if len(batch_list) == self.batch_size:
                yield self.collate_fn(batch_list)
                batch_list.clear()

        if len(batch_list) > 0:
            yield self.collate_fn(batch_list)


################################################################################
# method for reading the text files containing test pairs


def read_test_pairs_file(pairs_file_path: pathlib.Path) -> Tuple[bool, str, str]:
    with pairs_file_path.open("r") as f:
        for line in f.readlines():
            line = line.strip()

            if line.count(" ") < 2:
                continue

            gt, path1, path2 = line.strip().split(" ")

            yield bool(int(gt)), path1, path2

def read_test_file(pairs_file_path: pathlib.Path) -> Tuple[bool, str, str]:
    with pairs_file_path.open("r") as f:
        for line in f.readlines():
            line = line.strip()

            utterance_id, language_id = line.strip().split(" ")

            yield utterance_id, language_id

def load_evaluation_pairs(file_path: pathlib.Path):
    pairs = []

    for gt, path1, path2 in read_test_pairs_file(file_path):
        utt1id = path1.split(".wav")[0]
        utt2id = path2.split(".wav")[0]

        spk1id = path1.split("/")[0]
        spk2id = path2.split("/")[0]

        if (spk1id == spk2id) != gt:
            raise ValueError(f"read {gt=} for line `{path1} {path2}`")

        pairs.append(EvaluationPair(gt, utt1id, utt2id))

    return pairs


################################################################################
# Create the folder of test samples based on a list of pairs


def _create_test_split(
    root_test_extract_folder: pathlib.Path,#/g813_u1/mnt/OLR2020/test/task2/audio
    test_file_path: pathlib.Path,#
    test_folder: pathlib.Path,
    # all_language_ids: set,
):
    """
    1. <root_test_extract_folder>/task_2/audio/xxxxx.wav

    Given a file with test pairs, where each line is structured as:

    "[0/1] <spk_id>/<youtube_id>/xxxxx.wav <spk_id>/<youtube_id>/xxxxx.wav\n"

    Each folder in train/* which matches a <spk_id> in the test pair file is
    moved to the specified <test_folder>, and moved to <train_folder> otherwise.

    Each folder in test/* which matches a <spk_id> in the test pair file is
    moved to the specified <test_folder>, and ignored otherwise.

    If `all_voxceleb1_is_test_set` is True, all voxceleb1 is assumed to be test
    set and therefore each sample in in `<root_extract_folder>/train/vc1/wav/`
    is assumed to be under test/* as well.

    :param root_extract_folder: root directory which contains unzipped voxceleb1
     and/or voxceleb2 train and test folder.
    :param pairs_file_path: path to text file containing all test pairs

    :param test_folder: the folder to move all test files to
    :return: a set of all speaker ids in the the test set
    """
    all_test_samples=[]
    for utterance_id, language_id in read_test_file(test_file_path):
        # if(language_id in all_language_ids):
        all_test_samples.append((utterance_id,language_id))

    test_folder.mkdir(parents=True, exist_ok=True)
    # for language_id_path in all_language_ids:
        # path=test_folder / language_id_path
        # path.mkdir(parents=True, exist_ok=True)
    # print(all_test_samples)
    move_commands = []
    for sample in all_test_samples:
        utterance_id, language_id =sample
        path=test_folder / language_id
        path.mkdir(parents=True, exist_ok=True)
        utterance_id+=".wav"
        move_commands.append((str(root_test_extract_folder / utterance_id),str(test_folder / language_id)))
    for src, dst in move_commands:
        # print("正在搬运文件"+src+"到"+dst)
        shutil.copy(src, dst)



################################################################################
# create a train/val split based on a given training folder

def _create_train_val_split_diff_num_speakers(
    data_folder_path: pathlib.Path,
    train_folder_path: pathlib.Path,
    validation_folder_path: pathlib.Path,
    val_ratio: float,
    overwrite_existing_validation_folder: bool,
):
    """
    Create a train/val split of a voxceleb-structured folder:

    <train_folder_path>/wav/<spk_id>/<youtube_id>/xxxxx.wav

    This method ensures that `n` speakers are removed from the training set
    and moved to the validation set. Therefore the number of speakers in the
    training set differs from the number of speakers in the val split. There
    also is no overlap between the speakers in the train and val split.

    :param train_folder_path: the path to the root directory of training data
    :param validation_folder_path: the path to a directory where validation data
     is moved to
    :param num_val_speakers: the `n` number of speakers to move from train to val
     split
    :param overwrite_existing_validation_folder: whether to delete
    validation_folder_path if it already exists
    :param test_speaker_ids: set of speaker ids in the test set used to validate
     that no test data will be
    put in train or val set
    """
    # make sure validation folder exist
    if overwrite_existing_validation_folder:
        if(validation_folder_path.exists()): 
            remove_directory(validation_folder_path)
        if(train_folder_path.exists()):
            remove_directory(train_folder_path)
    validation_folder_path.mkdir(parents=True, exist_ok=False)
    train_folder_path.mkdir(parents=True,exist_ok=False)

    # select the validation speaker ids
    all_language_ids = set()
    for language_folder in data_folder_path.iterdir():
        if not language_folder.is_dir():
            continue
        language_id = language_folder.name
        (validation_folder_path / language_id).mkdir(parents=True,exist_ok=False)
        (train_folder_path / language_id).mkdir(parents=True,exist_ok=False)
        #获取所有注册的语种
        all_language_ids.add(language_id.lower())
        all_language_sample = list()
        total_num_sample=0
        for f in language_folder.rglob("*.WAV"):
            all_language_sample.append(f)
            total_num_sample+=1
        val_num_samples= int(total_num_sample * val_ratio)
        val_file_dict=all_language_sample[-val_num_samples:]
        train_file_dict=all_language_sample[:-val_num_samples]
        for val_sample in val_file_dict:
            if(val_sample in train_file_dict):
                raise ValueError(
                f"validtion sample id {val_sample} was found in {train_folder_path}"
            )
            shutil.copy(
                str(val_sample), str(validation_folder_path / language_id)
            )
        for train_sample in train_file_dict:
            if(train_sample in val_file_dict):
                raise ValueError(
                f"validtion sample id {train_sample} was found in {validation_folder_path}"
            )
            shutil.copy(
                str(train_sample), str(train_folder_path / language_id)
            )
    return all_language_ids

def _create_train_val_split_equal_num_speakers(
    data_folder_path: pathlib.Path,
    validation_folder_path: pathlib.Path,
    train_folder_path:pathlib.Path,
    val_ratio: float,
    overwrite_existing_validation_folder: bool,
):
    """
    Create a train/val split of a voxceleb-structured folder:

    <train_folder_path>/wav/<spk_id>/<youtube_id>/xxxxx.wav

    This method ensures that that every speaker is represented in the
    training and validation set; Therefore the number of speakers is equal
    between the train and val split.

    For each <spk_id> in <train_folder_path>, move a certain amount of its
    <youtube_id> subdirectories to <val_folder>, such that around a total of
    <val_ratio> wav files are located in <validation_folder_path>.

    :param train_folder_path: the path to the root directory of training data
    :param validation_folder_path: the path to a directory where validation data
     is moved to
    :param val_ratio: the ratio of validation data over all training data
    :param overwrite_existing_validation_folder: whether to delete
    validation_folder_path if it already exists
    :param test_speaker_ids: set of speaker ids in the test set used to validate
     that no test data will be
    put in train or val set
    """
    # make sure validation folder exist
    if overwrite_existing_validation_folder:
        if(validation_folder_path.exists()): 
            remove_directory(validation_folder_path)
        if(train_folder_path.exists()):
            remove_directory(train_folder_path)
    validation_folder_path.mkdir(parents=True, exist_ok=False)
    train_folder_path.mkdir(parents=True,exist_ok=False)

    # for each speaker we randomly select youtube_ids until we have achieved
    # the desired amount of validation samples
    all_language_ids = set()
    train_files_dict = list()
    total_samples=0
    for language_folder in data_folder_path.iterdir():
        if not language_folder.is_dir():
            continue

        language_id = language_folder.name
        (validation_folder_path / language_id).mkdir(parents=True,exist_ok=False)
        (train_folder_path / language_id).mkdir(parents=True,exist_ok=False)
        #获取所有注册的语种
        all_language_ids.add(language_id.lower())
        for f in language_folder.rglob("*.wav"):
            train_files_dict.append(f)
            total_samples+=1

    # print(total_samples)
    val_samples=int(total_samples * val_ratio)
    ranges= range(0,total_samples-1)
    indexs = random.sample(ranges, val_samples)
    indexs.sort(reverse=True)


    val_file_dict= list()
    for index in indexs:
        val_file_dict.append(train_files_dict.pop(index))
    assert len(val_file_dict)==len(indexs)

    move_commands = []
    for sample in val_file_dict:
        if(sample in train_files_dict):
             raise ValueError(
                f"validtion sample id {sample} was found in {train_folder_path}"
            )
        language_id=str(sample).split("/")[-2]
        dst=validation_folder_path / language_id
        move_commands.append((str(sample),str(dst)))

    for sample in train_files_dict:
        language_id=str(sample).split("/")[-2]
        dst = train_folder_path / language_id
        move_commands.append((str(sample),str(dst)))

    for src, dst in move_commands:
        print("搬移命令"+str(src)+"到"+str(dst))
        pass
        shutil.copy(src, dst)

    return all_language_ids

################################################################################
# shard a folder

ID_SEPARATOR = "/"



def write_shards(
    data_folder_path: pathlib.Path,
    shards_path: pathlib.Path,
    compress_in_place: bool,
    shard_name_pattern: str = "shard-{idx:06d}",
    samples_per_shard: int = 5000,
    sequential_same_speaker_samples: int = 1,
    min_unique_speakers_per_shard: int = 0,
    ensure_all_data_in_shards: bool = True,
    discard_partial_shards: bool = False,
):
    """
    Transform a voxceleb-structured folder of .wav files to WebDataset shards.

    :param voxceleb_folder_path: folder where extracted voxceleb data is located
    :param shards_path: folder to write shards of data to
    :param compress_in_place: boolean value determining whether the shards will
                              be compressed with the `gpig` utility.
    :param samples_per_shard: number of data samples to store in each shards.
    :param shard_name_pattern: pattern of name to give to each shard
    """
    # make sure output folder exist
    shards_path.mkdir(parents=True, exist_ok=True)

    # find all audio files
    audio_files = sorted([f for f in data_folder_path.rglob("*.wav")])

    # create data dictionary {speaker id: List[file_path, sample_key]}}
    data: Dict[str, List[Tuple[str, str, pathlib.Path]]] = defaultdict(list)

    all_language_ids = set()
    all_sample_ids = set()
    sample_keys_per_language = defaultdict(list)
    num_samples = 0
    all_keys = set()

    for f in audio_files:
        language_id = f.parent.name
        utterance_id = f.stem

        # create a unique key for this sample
        key = f"{language_id}{ID_SEPARATOR}{utterance_id}"
        
        if key in all_keys:
          raise ValueError("found sample with duplicate key")
        else:
          all_keys.add(key)

        # store statistics
        num_samples += 1

        all_language_ids.add(language_id)
        all_sample_ids.add(key)

        sample_keys_per_language[language_id].append(key)

    #     # store data in dict
        data[language_id].append((key, language_id, f))

    # randomly shuffle the list of all samples for each speaker
    for language_id in data.keys():
        random.shuffle(data[language_id])

    # determine a specific speaker_id label for each speaker_id
    language_id_to_idx = {
        language_id: idx for idx, language_id in enumerate(sorted(all_language_ids))
    }

    # write a meta.json file which contains statistics on the data
    # which will be written to shards
    all_language_ids = list(all_language_ids)
    all_sample_ids = list(all_sample_ids)

    meta_dict = {
        "language_ids":all_language_ids,
        "sample_ids": all_sample_ids,
        "language_id_to_idx": language_id_to_idx,
        "sample_ids_per_language": sample_keys_per_language,
        "num_samples": num_samples,
        "num_languages": len(all_language_ids),
    }

    with (shards_path / "meta.json").open("w") as f:
        json.dump(meta_dict, f)

    # split the data into shards such that each shard has at most
    # `samples_per_shard` samples and that the sequential order in the
    # shard is:
    # 1 = sample of speaker id `i`
    # ...
    # sequential_same_speaker_samples =sample of speaker id `i`
    # sequential_same_speaker_samples + 1 = sample of speaker id `j`
    # etc
    shards_list = []
    def samples_left():
        num_samples_left = sum(len(v) for v in data.values())
        num_valid_speakers = sum(
            len(v) >= sequential_same_speaker_samples for v in data.values()
        )

        # a shard should contain at least 2 different speakers
        if num_valid_speakers >= 2 or ensure_all_data_in_shards:
            return num_samples_left
        else:
            return 0

    def valid_speakers(n: int, previous_id: Optional[str] = None):
        return [k for k in data.keys() if len(data[k]) >= n and k != previous_id]

    def pop_n_samples(
        n: int, current_speakers_in_shard: Set[str], previous_id: Optional[str] = None
    ):
        valid_speaker_ids = valid_speakers(n, previous_id)


        if len(current_speakers_in_shard) < min_unique_speakers_per_shard:
            valid_speaker_ids = [
                sid for sid in valid_speaker_ids if sid not in current_speakers_in_shard
            ]

        if len(valid_speaker_ids) == 0:
          return [],None,set()
            # raise ValueError(
            #     f"shard cannot be guaranteed to have {min_unique_speakers_per_shard=}"
            # )

        samples_per_speaker = [len(data[k]) for k in valid_speaker_ids]
        random_speaker_id = random.choices(valid_speaker_ids, samples_per_speaker)[0]
        current_speakers_in_shard.add(random_speaker_id)
        popped_samples = []

        for _ in range(n):
            sample_list = data[random_speaker_id]
            popped_samples.append(
                sample_list.pop(random.randint(0, len(sample_list) - 1))
            )

        return popped_samples, random_speaker_id, current_speakers_in_shard

    # write shards
    print(samples_left())
    while samples_left() > 0:
        shard = []
        speakers_in_shard = set()
        previous = None

        print(
            f"determined shards={len(shards_list):>4}\t"
            f"samples left={samples_left():>9,d}\t"
            f"languagw left="
            f"{len(valid_speakers(sequential_same_speaker_samples, previous)):>4,d}"
        )
        while len(shard) < samples_per_shard and samples_left() > 0:
            samples, previous, speakers_in_shard = pop_n_samples(
                n=sequential_same_speaker_samples,
                current_speakers_in_shard=speakers_in_shard,
                previous_id=previous,
            )
            for key, language_id, f in samples:
                shard.append((key, language_id_to_idx[language_id], f))

        shards_list.append(shard)
    

    # assert all data is in a shard
    if ensure_all_data_in_shards:
        assert sum(len(v) for v in data.values()) == 0

    # remove any shard which does share the majority amount of samples
    if discard_partial_shards:
        unique_len_count = defaultdict(int)
        for lst in shards_list:
            unique_len_count[len(lst)] += 1
        
        if len(unique_len_count) > 2:
            raise ValueError("expected at most 2 unique lengths")

        if len(unique_len_count) == 0:
            raise ValueError("expected at least 1 unique length")

        majority_len = -1
        majority_count = -1
        for unique_len, count in unique_len_count.items():
            if count > majority_count:
                majority_len = unique_len
                majority_count = count

        shards_list = [lst for lst in shards_list if len(lst) == majority_len]


    # write shards
    shards_path.mkdir(exist_ok=True, parents=True)

    # seems like disk write speed only allows for 1 process anyway :/
    with multiprocessing.Pool(processes=1) as p:
        for idx, shard_content in enumerate(shards_list):
            args = {
                "shard_name": shard_name_pattern.format(idx=idx),
                "shards_path": shards_path,
                "data_tpl": shard_content,
                "compress": compress_in_place,
            }
            p.apply_async(
                _write_shard,
                kwds=args,
                error_callback=lambda x: print(
                    f"error in apply_async ``_write_shard!\n{x}"
                ),
            )

        p.close()
        p.join()


# function to write shards to disk, used internally
def _write_shard(
    shard_name: str, shards_path: pathlib.Path, data_tpl: List, compress: bool = True
):
    if shard_name.endswith(".tar.gz"):
        # `pigz` will automatically add extension (and would break if it's
        # already there)
        shard_name = shard_name.split(".tar.gz")[0]

    if not shard_name.endswith(".tar"):
        shard_name += ".tar"

    shard_path = str(shards_path / shard_name)
    print(f"writing shard {shard_path}")
    # note that we manually compress with `pigz` which is a lot faster than python
    with wds.TarWriter(shard_path) as sink:
        for key, language_id_idx, f in data_tpl:
            # load the audio tensor
            tensor, sample_rate = torchaudio.load(str(f))

            if(tensor.shape[0]>1):
                tensor=tensor[:1]
            if torch.any(torch.isnan(tensor)):
                raise ValueError(f"NaN value in wav file of {key=} at {f=}")

            # extract speaker_id, youtube_id and utterance_id from key
            language_id,utterance_id = key.split(ID_SEPARATOR)

            # create sample to write
            sample = {
                "__key__": key,
                "wav.pyd": tensor,
                "meta.json": {
                    "language_id":language_id,
                    "utterance_id": utterance_id,
                    "language_id_idx": language_id_idx,
                    "num_frames": len(tensor.squeeze()),
                    "sampling_rate": sample_rate,
                },
            }
            
            # write sample to sink
            sink.write(sample)

            # delete audio file on disk
            # f.unlink()

    if compress:
        subprocess.call(
            ["pigz", shard_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
