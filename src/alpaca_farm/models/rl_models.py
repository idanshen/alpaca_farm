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

"""Model classes that are shared across different algorithms.

WARNING:
    Do not tamper with the state_dict function for any of these classes.
    If you tamper, make sure the keys are the same, otherwise FSDP will get confused.
"""

import abc
from typing import Dict, Optional, Any

import accelerate
import torch
import transformers
from torch import Tensor, nn

from .. import common, logging, torch_ops

logger = logging.get_logger(__name__)


class Policy(nn.Module, abc.ABC):
    def __init__(
        self, args, base_model: transformers.PreTrainedModel, base_tokenizer: transformers.PreTrainedTokenizer
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.model_parallel = True
        self.is_parallelizable = True

    @abc.abstractmethod
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
        resoinse_len: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        assert not self.training, "Policy must be in eval model for generation."
        return self._post_respond(self._respond(queries, query_attn_masks, temperature, num_return_sequences, resoinse_len))

    @abc.abstractmethod
    def _respond(
        self, queries: Tensor, query_attn_masks: Tensor, temperature: Optional[float] = None, num_return_sequences=1, response_len: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def _post_respond(self, respond_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return respond_outputs


class AutoregressivePolicy(Policy):
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        # TODO(lxuechen): Refactor attention mask. Here query_attn_masks overrides padding-based attention mask.
        if temperature is None:
            temperature = self.args.temperature
        input_ids = torch.cat([queries, responses], dim=1)
        attention_mask = input_ids.ne(self.base_tokenizer.pad_token_id)
        attention_mask[:, : queries.size(1)] = query_attn_masks
        # Fix position id issues and ensure consistency with `respond` for GPT and OPT.
        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        outputs = self.base_model(**inputs, output_hidden_states=True)
        original_logits = outputs.logits[:, -responses.shape[1] - 1 : -1]
        logits = original_logits / temperature
        labels = input_ids[:, -responses.shape[1] :]
        logprobs = torch_ops.compute_logprobs(logits, labels, ignore_index=self.base_tokenizer.pad_token_id)
        entropies = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)
        last_hidden_state = outputs.hidden_states[-1][:, -responses.shape[1] - 1 : -1]
        return dict(
            original_logits=original_logits,
            logits=logits,
            logprobs=logprobs,
            entropies=entropies,
            last_hidden_state=last_hidden_state,
        )

    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
        response_len: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if temperature is None:
            temperature = self.args.temperature
        sequences = self.base_model.generate(
            inputs=queries,
            attention_mask=query_attn_masks,
            do_sample=True,
            max_new_tokens=self.args.response_len if response_len is None else response_len,
            pad_token_id=self.base_tokenizer.pad_token_id,
            top_p=1.0,
            top_k=0,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            # synced_gpus=True,
        )
        responses = torch_ops.right_pad(
            sequences[:, queries.size(1) :],
            target_size=(sequences.size(0), self.args.response_len),
            value=self.base_tokenizer.pad_token_id,
        )
        return dict(responses=responses)  # Size (bsz * num_return_sequences, response_len).


class Value(nn.Module, abc.ABC):
    def __init__(
        self, args, base_model: transformers.PreTrainedModel, base_tokenizer: transformers.PreTrainedTokenizer, accelerator: accelerate.Accelerator,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.accelerator = accelerator

        hidden_size = common.get_transformer_hidden_size(base_model)
        hidden_layer_device = list(self.base_model.parameters())[-1].device
        self.head_device = hidden_layer_device

        value_head = torch.nn.Linear(hidden_size, 1)
        value_head.weight.data.zero_()
        value_head.bias.data.zero_()
        self.value_head = value_head.to(self.head_device)
        self.model_parallel = True
        self.is_parallelizable = True

    @abc.abstractmethod
    def forward(self, queries: Tensor, query_attn_masks: Tensor, responses: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError
    
    def load_v_head(self, path: str, strict: bool=True):
        # TODO (idanshen): fix this once value model saving is fixed to saving weights not the whole model
        value_head_ckpt = torch.load(path, map_location=self.head_device)
        self.value_head.load_state_dict(value_head_ckpt['state_dict'], strict=strict)
        self.value_head.load_state_dict(torch.load(path, map_location=self.head_device)['state_dict'], strict=strict)
        self.value_head.forward = common.cast_with_native_amp(self.value_head.forward, mixed_precision=self.accelerator.mixed_precision)


class AutoregressiveValue(Value):
    def forward(self, queries: Tensor, query_attn_masks: Optional[Tensor] = None, responses: Optional[Tensor] = None, only_last: bool = False, use_cache: bool = False, past_key_values: Tensor = None) -> Dict[str, Tensor]:
        if responses is not None:
            sequences = torch.cat([queries, responses], dim=1)
        else:
            sequences = queries

        sequence_attn_masks = sequences.ne(self.base_tokenizer.pad_token_id)

        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequence_attn_masks,
            use_cache=False,
        )
        outputs = self.base_model.model(**inputs, output_hidden_states=True)
        # get the hidden state of the last layer
        if only_last:
            last_hidden_state = outputs.hidden_states[-1][:, - 1:, :].squeeze(1)
        else:
            # value[t]: \hat{V}(sequences_{:t-1}); must align with `_estimate_advantage`.
            last_hidden_state = outputs.hidden_states[-1][:, queries.size(1) : , :]

        with self.accelerator.autocast():
            values = self.value_head(last_hidden_state).squeeze(-1)

        return dict(values=values)

    def decode(self, input_ids: Tensor, use_cache: bool = False, past_key_values: Tensor = None) -> Dict[str, Tensor]:
        sequences = input_ids
        sequence_attn_masks = sequences.ne(self.base_tokenizer.pad_token_id)

        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequence_attn_masks,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        outputs = self.base_model.model(**inputs, output_hidden_states=True)
        # get the hidden state of the last layer
        if use_cache and past_key_values is not None:
            last_hidden_state = outputs.hidden_states[-1].squeeze(1)
        else:
            last_hidden_state = outputs.hidden_states[-1][:, - 1:, :].squeeze(1)

        with self.accelerator.autocast():
            values = self.value_head(last_hidden_state).squeeze(-1)

        if use_cache:
            return dict(values=values, past_key_values=outputs.past_key_values)
        else:
            return dict(values=values)

class ActorCritic(nn.Module):
    def __init__(self, policy: Policy, value_model: Value):
        super(ActorCritic, self).__init__()
        self.policy = policy
        self.value_model = value_model
        self.model_parallel = True
        self.is_parallelizable = True

    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        # Assume the policy and value model share the same tokenizer.
        # TODO (seungwook): fix this assumption!!!
        o1 = self.policy(queries, query_attn_masks, responses, temperature)
        o2 = self.value_model(queries, query_attn_masks, responses)
        return {**o1, **o2}

    def respond(
        self, queries: Tensor, query_attn_masks: Tensor, temperature: Optional[float] = None
    ) -> Dict[str, Tensor]:
        return self.policy.respond(queries=queries, query_attn_masks=query_attn_masks, temperature=temperature)


class Qfunction(nn.Module, abc.ABC):
    def __init__(
        self, args, base_model: transformers.PreTrainedModel, base_tokenizer: transformers.PreTrainedTokenizer, accelerator: accelerate.Accelerator,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.accelerator = accelerator

        hidden_size = common.get_transformer_hidden_size(base_model)
        hidden_layer_device = list(self.base_model.parameters())[-1].device
        self.head_device = hidden_layer_device

        self.feature_size = hidden_size
        if args.q_head_type == "linear":
            q_head = torch.nn.Linear(self.feature_size, len(base_tokenizer)*self.args.num_q_heads)
            q_head.weight.data.zero_()
            q_head.bias.data.zero_()
            self.q_head = q_head.to(self.head_device)
        elif args.q_head_type == "projection":
            assert args.num_q_heads == 1
            self.token_features = self.base_model.get_input_embeddings().weight.type(torch.float16).to(self.head_device)
            self.feature_size = self.token_features.shape[1]
            w_h = torch.nn.Linear(in_features=hidden_size, out_features=self.feature_size)
            w_h.weight.data.zero_()
            w_h.bias.data.zero_()
            self.q_head = w_h.to(self.head_device)
        elif args.q_head_type == 'dueling':
            # Dueling network from https://arxiv.org/abs/1511.06581
            self.advantage_head = torch.nn.Linear(self.feature_size, len(base_tokenizer)*self.args.num_q_heads)
            self.advantage_head.weight.data.zero_()
            self.advantage_head.bias.data.zero_()
            self.advantage_head = self.advantage_head.to(self.head_device)
            self.value_head = torch.nn.Linear(self.feature_size, 1)
            self.value_head.weight.data.zero_()
            self.value_head.bias.data.zero_()
            self.value_head = self.value_head.to(self.head_device)
            self.q_head = torch.nn.ModuleList([self.advantage_head, self.value_head])
        elif args.q_head_type == 'q_and_v':
            self.value_head = torch.nn.Linear(self.feature_size, 1)
            self.value_head.weight.data.zero_()
            self.value_head.bias.data.zero_()
            self.value_head = self.value_head.to(self.head_device)
            self.out_q_head = torch.nn.Linear(self.feature_size, len(base_tokenizer)*self.args.num_q_heads)
            self.out_q_head.weight.data.zero_()
            self.out_q_head.bias.data.zero_()
            self.out_q_head = self.out_q_head.to(self.head_device)
            self.q_head = torch.nn.ModuleList([self.value_head, self.out_q_head])
        else:
            raise NotImplementedError

        self.model_parallel = True
        self.is_parallelizable = True

    @abc.abstractmethod
    def forward(self, queries: Tensor, query_attn_masks: Tensor, responses: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError
    
    # load q head from checkpoint
    def load_q_head(self, path: str, strict: bool=True):
        q_head_ckpt = torch.load(path, map_location=self.head_device)
        self.q_head.load_state_dict(q_head_ckpt['state_dict'], strict=strict)
        self.q_head.forward = common.cast_with_native_amp(self.q_head.forward, mixed_precision=self.accelerator.mixed_precision)


class AutoregressiveQfunction(Qfunction):
    def forward(self, queries: Tensor, query_attn_masks: Optional[Tensor] = None, responses: Optional[Tensor] = None, only_last: bool = False, use_cache: bool = False, past_key_values: Tensor = None) -> Dict[str, Tensor]:
        if responses is not None:
            sequences = torch.cat([queries, responses], dim=1)
        else:
            sequences = queries

        sequence_attn_masks = sequences.ne(self.base_tokenizer.pad_token_id)

        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequence_attn_masks,
            use_cache=False,
        )
        outputs = self.base_model.model(**inputs, output_hidden_states=True)

        # get the hidden state of the last layer
        if only_last:
            last_hidden_state = outputs.hidden_states[-1][:, -2:, :]
        else:
            last_hidden_state = outputs.hidden_states[-1][:, queries.size(1) - 1 : ,:]

        # project the hidden state to the q values
        with self.accelerator.autocast():
            if self.args.q_head_type == "linear":
                last_hidden_state = last_hidden_state[:, :-1, :]
                qvalues = self.q_head(last_hidden_state).squeeze(-1)
                if self.args.num_q_heads > 1:
                    qvalues = qvalues.view(-1, self.args.num_q_heads, len(self.base_tokenizer))
                return dict(qvalues=qvalues)
            elif self.args.q_head_type == "projection":
                last_hidden_state = last_hidden_state[:, :-1, :]
                h_features = self.q_head(last_hidden_state)  # from B x L x H to B x L x H'
                t_features = self.token_features  # T x H'
                qvalues = h_features @ t_features.T  # B x L x T
                return dict(qvalues=qvalues)
            elif self.args.q_head_type == "dueling":
                # The Q head returns values for tokens queries.size(1) - 1 : -1 while the Value head returns values for tokens queries.size(1) : end
                # The next_advantage is the advantage for the next state, i.e queries.size(1) : end
                advantage = self.advantage_head(last_hidden_state)
                value = self.value_head(last_hidden_state)
                qvalues = value[:, :-1, :].detach() + advantage[:, :-1, :] #- advantage.mean(dim=-1, keepdim=True)
                value = value[:, 1:, :]
                next_advantage = advantage[:, 1:, :]
                if self.args.num_q_heads > 1:
                    qvalues = qvalues.view(-1, self.args.num_q_heads, len(self.base_tokenizer))
                return dict(qvalues=qvalues, values=value, next_advantage=next_advantage)
            elif self.args.q_head_type == "q_and_v":
                # The Q head returns values for tokens queries.size(1) - 1 : -1 while the Value head returns values for tokens queries.size(1) : end
                value = self.value_head(last_hidden_state)
                qvalues = self.out_q_head(last_hidden_state)
                qvalues = qvalues[:, :-1, :]
                value = value[:, 1:, :]
                if self.args.num_q_heads > 1:
                    qvalues = qvalues.view(-1, self.args.num_q_heads, len(self.base_tokenizer))
                return dict(qvalues=qvalues, values=value)

    @torch.inference_mode()
    def decode(self, input_ids: Tensor, use_cache: bool = False, past_key_values: Tensor = None) -> Dict[str, Tensor]:
        sequences = input_ids
        sequence_attn_masks = sequences.ne(self.base_tokenizer.pad_token_id)

        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequence_attn_masks,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        outputs = self.base_model.model(**inputs, output_hidden_states=True)

        # get the hidden state of the last layer
        if use_cache and past_key_values is not None:
            last_hidden_state = outputs.hidden_states[-1].squeeze(1)
        else:
            last_hidden_state = outputs.hidden_states[-1][:, -1:, :].squeeze(1)

        # project the hidden state to the q values
        with self.accelerator.autocast():
            if self.args.q_head_type == "linear":
                raise NotImplementedError
            elif self.args.q_head_type == "projection":
                raise NotImplementedError
            elif self.args.q_head_type == "dueling":
                # The Q head returns values for tokens queries.size(1) - 1 : -1 while the Value head returns values for tokens queries.size(1):
                advantage = self.advantage_head(last_hidden_state)
                value = self.value_head(last_hidden_state)
                qvalues = value.detach() + advantage
                if self.args.num_q_heads > 1:
                    qvalues = qvalues.view(-1, self.args.num_q_heads, len(self.base_tokenizer))
            elif self.args.q_head_type == "q_and_v":
                qvalues = self.out_q_head(last_hidden_state)
                value = None
                if self.args.num_q_heads > 1:
                    qvalues = qvalues.view(-1, self.args.num_q_heads, len(self.base_tokenizer))

        if use_cache:
            return dict(qvalues=qvalues, past_key_values=outputs.past_key_values, values=value)
        else:
            return dict(qvalues=qvalues, values=value)


def make_policy_with_base_model(
    args, base_model: transformers.PreTrainedModel, base_tokenizer: transformers.PreTrainedTokenizer
) -> Policy:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressivePolicy(args, base_model, base_tokenizer)


def make_value_with_base_model(
    args,
    base_model: transformers.PreTrainedModel,
    base_tokenizer: transformers.PreTrainedTokenizer,
    accelerator: accelerate.Accelerator,
) -> Value:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressiveValue(args, base_model, base_tokenizer, accelerator)

def make_qfunction_with_base_model(
    args,
    base_model: transformers.PreTrainedModel,
    base_tokenizer: transformers.PreTrainedTokenizer,
    accelerator: accelerate.Accelerator,
) -> Qfunction:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressiveQfunction(args, base_model, base_tokenizer, accelerator)
