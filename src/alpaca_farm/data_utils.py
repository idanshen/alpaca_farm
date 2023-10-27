# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List

import datasets
import pandas as pd
import transformers

from alpaca_farm.utils import jload, jdump
from . import logging, utils
from .data_postprocessor import RewardConditioningPromptPostprocessor
from .data_preprocessor import (
    SoftPreferenceRewardModelingDataset,
    BinaryRewardModelingDataset,
    DataCollatorForSoftPreferenceRewardModelingDataset,
    DataCollatorForBinaryRewardModelingDataset,
    DataCollatorForSFTDataset,
    DataCollatorForStackableDataset,
    QueryDataset,
    SummaryQueryDataset,
    NoInputQueryDataset,
    ReviewQueryDataset,
    QueryResponseDataset,
    SFTDataset,
    split_train_into_train_and_eval,
    format_prompt, OutputValuesDataset, DataCollatorForClassificationRewardModelingDataset,
    ClassificationRewardModelingDataset,
)

logger = logging.get_logger(__name__)
    

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)

    if 'seahorse' in data_args.dataset_path:
        data_files = {"train": "train.json", "validation": "validation.json"}
        alpaca_instructions = datasets.load_dataset(data_args.dataset_path, data_files=data_files)

        seahorse_instruction = "Generate a one-sentence summary of this post."
        alpaca_instructions = alpaca_instructions.filter(lambda example: example['worker_lang'] == 'en-US')
        alpaca_instructions = alpaca_instructions.map(
            lambda example: {
                "instruction": seahorse_instruction,
                "input": example["text"],
                "output": example["summary"],
            },
            remove_columns=["gem_id", "worker_lang", "model", "question1", "question2", "question3", "question4", "question5", "question6"]
        )
    else:
        alpaca_instructions = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)

    train_df = pd.concat([pd.DataFrame(alpaca_instructions[split]) for split in data_args.train_splits])
    train_dataset = SFTDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
    )

    eval_dataset = None
    if data_args.eval_splits is not None:
        found_splits = [
            pd.DataFrame(alpaca_instructions[split]) for split in data_args.eval_splits if split in alpaca_instructions
        ]
        if len(found_splits) > 0:
            eval_df = pd.concat(found_splits)
            eval_dataset = SFTDataset(
                df=eval_df,
                prompt_dict=prompt_dict,
                tokenizer=tokenizer,
            )

    if eval_dataset is None:
        logger.warning("Didn't find evaluation dataset. Disabling evaluation.")
        training_args.do_eval = False

    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def make_binary_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)

    alpaca_human_preference = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)
    train_df = pd.DataFrame(alpaca_human_preference["preference"])

    train_dataset = BinaryRewardModelingDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        end_sequence_with_eos=training_args.end_sequence_with_eos,
    )
    train_dataset, eval_dataset = split_train_into_train_and_eval(
        train_dataset=train_dataset,
        eval_size=data_args.eval_size,
        seed=training_args.seed,
    )
    data_collator = DataCollatorForBinaryRewardModelingDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def make_classification_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)
    data_files = {"train": "train.json", "validation": "validation.json"}
    dataset_json = datasets.load_dataset(data_args.dataset_path, data_files=data_files)
    dataset_json = dataset_json.filter(lambda example: example['worker_lang'] == 'en-US')

    train_dataset = ClassificationRewardModelingDataset(
        df=pd.DataFrame(dataset_json["train"]),
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        end_sequence_with_eos=training_args.end_sequence_with_eos,
        classification_label_key=data_args.classification_label_key,
    )

    eval_dataset = ClassificationRewardModelingDataset(
        df=pd.DataFrame(dataset_json["validation"]),
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        end_sequence_with_eos=training_args.end_sequence_with_eos,
        classification_label_key=data_args.classification_label_key,
    )

    data_collator = DataCollatorForClassificationRewardModelingDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)




def make_soft_preference_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)
    train_df = pd.read_json(data_args.train_data_filpeath)
    eval_df = pd.read_json(data_args.validation_data_filepath)

    train_dataset = SoftPreferenceRewardModelingDataset(
        dataset_path=data_args.dataset_path,
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        llm_label_type='both',
        end_sequence_with_eos=training_args.end_sequence_with_eos,
    )

    # creating separate datasets for eval because we need to compare the scores 
    # for each of the respones and then decide which is preferred by the rm
    # eval dataset where it pairs the same prompt with the first response
    eval_dataset1 = SoftPreferenceRewardModelingDataset(
        dataset_path=data_args.dataset_path,
        df=eval_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        llm_label_type='first',
        end_sequence_with_eos=training_args.end_sequence_with_eos,
    )

    # eval dataset where it pairs the same prompt with the second response
    eval_dataset2 = SoftPreferenceRewardModelingDataset(
        dataset_path=data_args.dataset_path,
        df=eval_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        llm_label_type='second',
        end_sequence_with_eos=training_args.end_sequence_with_eos,
    )

    data_collator = DataCollatorForSoftPreferenceRewardModelingDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=[eval_dataset1, eval_dataset2], data_collator=data_collator)

def make_rl_data_module(
    tokenizer: List[transformers.PreTrainedTokenizer],
    data_args,
    training_args,
):
    tokenizer, _ = tokenizer # only use policy tokenizer for data module (not reward tokenizer)
    prompt_dict = utils.jload(data_args.prompt_dict_path)

    if data_args.dataset_path == 'imdb':
        alpaca_instructions = datasets.load_dataset(data_args.dataset_path) # doesn't require separate path and name
    elif 'seahorse' in data_args.dataset_path:
        data_files = {"train": "train.json", "validation": "validation.json"}
        alpaca_instructions = datasets.load_dataset(data_args.dataset_path, data_files=data_files)
    else:
        alpaca_instructions = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)

    # TODO: may need to impose min and max lengths per task as they do in rewardsoup
    if data_args.dataset_path == 'argilla/news-summary' :
        split_map = {"train": "test", "validation": "train"} # swap train and validation b/c more train dataset is quite small and validation is bigger
        train_split = split_map[data_args.train_splits[0]]
        eval_split = split_map[data_args.eval_splits[0]]
        train_df = alpaca_instructions[train_split]
        eval_df = alpaca_instructions[eval_split]
    elif data_args.dataset_path in {'lvwerra/stack-exchange-paired', 'Anthropic/hh-rlhf'} or 'seahorse' in data_args.dataset_path:
        # TODO: may need to load from offline data on disk for speed for stack exchange
        split_map = {"train": "train", "validation": "test"}
        train_split = split_map[data_args.train_splits[0]]
        eval_split = split_map[data_args.eval_splits[0]]
        train_df = alpaca_instructions[train_split]
        eval_df = alpaca_instructions[eval_split]
    else:
        train_df = pd.concat([pd.DataFrame(alpaca_instructions[split]) for split in data_args.train_splits])
        eval_df = pd.concat([pd.DataFrame(alpaca_instructions[split]) for split in data_args.eval_splits])

    # for Quark training
    if getattr(training_args, "num_reward_tokens", 0) > 0 and not getattr(
        training_args, "train_on_best_quantile", True
    ):
        prompt_postprocessor = RewardConditioningPromptPostprocessor()
    else:
        prompt_postprocessor = None

    # instantiate dataset class depending on the dataset
    if data_args.dataset_path in {'argilla/news-summary', 'openai/summarize_from_feedback'} or 'seahorse' in data_args.dataset_path:
        dataset_cls = SummaryQueryDataset
    elif data_args.dataset_path == {'lvwerra/stack-exchange-paired', 'Anthropic/hh-rlhf'}:
        dataset_cls = NoInputQueryDataset
    elif data_args.dataset_path == 'imdb':
        dataset_cls = ReviewQueryDataset
    else: 
        dataset_cls = QueryDataset

    if not training_args.static_dataset:
        train_dataset = dataset_cls(
            df=train_df,
            prompt_dict=prompt_dict,
            tokenizer=tokenizer,
            query_len=training_args.query_len,
            prompt_postprocessor=prompt_postprocessor,
            dataset_name=data_args.dataset_path,
            split='train',
        )

        eval_dataset = dataset_cls(
            df=eval_df,
            prompt_dict=prompt_dict,
            tokenizer=tokenizer,
            query_len=training_args.query_len,
            prompt_postprocessor=prompt_postprocessor,
            dataset_name=data_args.dataset_path,
            split='val',
        )
    else:
        path_to_data = training_args.static_dataset_path
        if os.path.isfile(path_to_data):
            list_dict_data = jload(path_to_data)
        elif os.path.isdir(path_to_data):
            list_dict_data = []
            for filename in os.listdir(path_to_data):
                if filename.endswith(".json"):
                    path_to_file = os.path.join(path_to_data, filename)
                    list_dict_data.extend(jload(path_to_file))
        else:
            raise ValueError(f"static_dataset_path {path_to_data} is not a file or directory")
        prompts = [format_prompt(example=dict_data, prompt_dict=prompt_dict) for dict_data in list_dict_data]
        if prompt_postprocessor is not None:
            prompts = [prompt_postprocessor(prompt) for prompt in prompts]
        train_dataset = QueryResponseDataset(tokenizer=tokenizer,
                                             queries=prompts,
                                             responses=[dict_data['output'] for dict_data in list_dict_data],
                                             query_len=training_args.query_len,
                                             response_len=training_args.response_len,)

        path_to_val_data = training_args.static_val_dataset_path
        assert os.path.isfile(path_to_val_data)
        list_dict_val_data = jload(path_to_val_data)
        eval_dataset = OutputValuesDataset(tokenizer=tokenizer,
                                           list_dict_data=list_dict_val_data,
                                           query_len=training_args.query_len+training_args.response_len,)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=DataCollatorForStackableDataset())

def make_eval_data_module(tokenizer: transformers.PreTrainedTokenizer, generated_outputs_list_of_dict: List[dict]):
    """
    Creates a data module for evaluation of a trained model's generated samples
    """