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
import os.path
from functools import partial

import accelerate
import torch
import transformers
from torch import Tensor, nn
import t5_encoder # note that this loads in a different model for t5 sentence classification models
from transformers.utils.generic import ModelOutput
from transformers import pipeline

from .. import common


class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(RewardConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path


class RewardModelOutput(ModelOutput):
    rewards: Tensor = None


class RewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig

    def __init__(self, config: RewardConfig, accelerator: accelerate.Accelerator, pretrained_lora_weights: str = None, **kwargs):
        super(RewardModel, self).__init__(config)
        self.accelerator = accelerator

        self.backbone_model = common.get_accelerate_model(
            model_name_or_path=config.backbone_model_name_or_path,
            pretrained_lora_weights=pretrained_lora_weights,
            accelerator=accelerator,
            **kwargs)

        hidden_layer_device = list(self.backbone_model.parameters())[-1].device

        self.model_parallel = True
        self.is_parallelizable = True

        hidden_size = common.get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 2) if 'soft_preference' in kwargs else nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        self.reward_head = reward_head.to(hidden_layer_device)

        if pretrained_lora_weights is not None:
            print("load reward head from checkpoint")
            if os.path.exists(pretrained_lora_weights + "/reward_head.pt"):
                self.reward_head = torch.load(pretrained_lora_weights + "/reward_head.pt").to(hidden_layer_device)
            else:
                print("reward head not found, use random initialization")

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        outputs = self.backbone_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True, **kwargs
        )
        last_hidden_state = outputs.hidden_states[-1]
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone_model.get_output_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.backbone_model.set_input_embeddings(value)

    def gradient_checkpointing_enable(self):
        self.backbone_model.gradient_checkpointing_enable()


class RewardNoLoraModel(transformers.PreTrainedModel):
    config_class = RewardConfig

    def __init__(self, config: RewardConfig, accelerator: accelerate.Accelerator, **kwargs):
        super(RewardNoLoraModel, self).__init__(config)
        self.accelerator = accelerator
        print('Initializing reward model that is not lora based')

        self.model = common.get_accelerate_sc_model(
                    model_name_or_path=config.backbone_model_name_or_path,
                    accelerator=accelerator,
                    **kwargs)
        self.backbone_model = None
        if hasattr(self.model, 'transformer'):
            self.backbone_model = self.model.transformer
        elif hasattr(self.model, 'model'):
            self.backbone_model = self.model.model
        elif hasattr(self.model, 'encoder'):
            self.backbone_model = self.model.encoder
        else:
            raise Exception('Backbone model not found')
        
        self.backbone_model = self.model.transformer if hasattr(self.model, 'transformer') else self.model.model # TODO (seungwook): may need to fix for other models
        
        self.reward_head = None
        if hasattr(self.model, 'score'):
            self.reward_head = self.model.score
        elif hasattr(self.model, 'classifier'):
            self.reward_head = self.model.classifier
        else:
            raise Exception('Reward head not found')
        
        self.model_parallel = True
        self.is_parallelizable = True

        # function to apply to the output of the model
        # self.function_to_apply = None
        # if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
        #     self.function_to_apply = torch.sigmoid
        # elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
        #     self.function_to_apply = partial(torch.softmax, dim=-1)
        # elif hasattr(self.model.config, "function_to_apply") and self.function_to_apply is None:
        #     self.function_to_apply = self.model.config.function_to_apply # TODO: not implemented yet
        # else:

        # use raw outputs of the models
        # self.function_to_apply = lambda x: x
        
    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        rewards = self.model(input_ids, attention_mask=attention_mask, return_dict=True, **kwargs).logits.squeeze(-1)
        # rewards = self.function_to_apply(outputs.logits).squeeze(-1)
        
        # special case of bart summarization reward model, need to take the difference btw faithful (label 1) and hallucination (label 0)
        if rewards.shape[-1] > 1 and rewards.ndim == 2:
            rewards = rewards[:, 1] - rewards[:, 0]

        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)
         
    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.model.get_output_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.model.set_input_embeddings(value)

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

class RewardPipeline():
    def __init__(self, config: RewardConfig, tokenizer: transformers.PreTrainedTokenizer,**kwargs):
        print('Initializing reward pipeline from a pretrained model (no lora weights)')
        self.pipeline = pipeline('text-classification', 
                                 model=config.backbone_model_name_or_path,
                                 device_map='auto',
                                 tokenizer=tokenizer
                                 )

    def forward(self, inputs, return_dict=True, **kwargs):
        outputs = self.pipeline(inputs)
        rewards = torch.tensor([output['score'] for output in outputs])

        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)