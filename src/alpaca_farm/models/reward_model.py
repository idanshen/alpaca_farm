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

import torch
import transformers
from torch import Tensor, nn
from transformers.utils.generic import ModelOutput

from .. import common


class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(RewardConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path
        # self._name_or_path = common.get_pretrained_model_name_with_model_name_or_path(backbone_model_name_or_path)


class RewardModelOutput(ModelOutput):
    rewards: Tensor = None


class RewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig

    def __init__(self, config: RewardConfig, checkpoint_dir: str = None, **kwargs):
        super(RewardModel, self).__init__(config)
        # self.backbone_model = common.make_generative_lm(config.backbone_model_name_or_path, **kwargs)
        self.backbone_model = common.get_accelerate_model(
            model_name_or_path=config.backbone_model_name_or_path,
            flash_attn=kwargs['flash_attn'],
            checkpoint_dir=checkpoint_dir,)
        hidden_size = common.get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        self.reward_head = reward_head.to(next(self.backbone_model.parameters()).device)
        if checkpoint_dir is not None:
            print("load reward head from checkpoint")
            self.reward_head = torch.load(checkpoint_dir + "/reward_head.pt")

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        outputs = self.backbone_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True, **kwargs
        )
        last_hidden_state = outputs.hidden_states[-1]
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        # TODO(lxuechen): Make returning rewards at all positions and last_hidden_state an option.
        if last_hidden_state_at_the_end.dtype != torch.float32:
            last_hidden_state_at_the_end = last_hidden_state_at_the_end.float()
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone_model.get_output_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.backbone_model.set_input_embeddings(value)

