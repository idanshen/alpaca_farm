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
from typing import Callable, Dict, Optional, Tuple, List

import accelerate
import pandas as pd
import torch
import tqdm
import transformers
from torch import nn
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.utils.data import DataLoader
from transformers.modeling_utils import unwrap_model
from accelerate import DistributedType
import peft

from .. import accelerate_patch, common, constants, data_preprocessor, logging, torch_ops, utils
from ..common import save_peft_model
from ..models import reward_model as reward_model_module
from ..models import rl_models
from ..models.make_models import make_generative_policy, make_reward_model
from ..types import AnyPath, AnyPathOrNone, LRScheduler, Tensor
from . import rl_trainer
from .trainer_utils import _make_padded_tokenizer

logger = logging.get_logger(__name__)


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class FQETrainer(rl_trainer.RLTrainer):
    def __init__(
        self,
        args,
        train_dataset: data_preprocessor.QueryDataset,
        eval_dataset: data_preprocessor.QueryDataset,
        data_collator: Callable,
        qfunction_model: rl_models.Qfunction,
        ref_policy: rl_models.Policy,
        reward_model: nn.Module,
        tokenizer: List[transformers.PreTrainedTokenizer],
        accelerator: accelerate_patch.MyAccelerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(FQETrainer, self).__init__(
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            policy=qfunction_model,
            ref_policy=ref_policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        eval_dataloader = DataLoader(
            dataset=self.eval_dataset,
            collate_fn=self.data_collator,
            batch_size=args.per_device_eval_batch_size,
            shuffle=True,
            drop_last=True,)
        self.eval_dataloader = self.accelerator.prepare(eval_dataloader)  # noqa
        if self.args.q_head_type == 'q_and_v':
            assert self.args.lam == 0.0

    def _shape_reward(
        self, rewards: Tensor, responses: Tensor,
    ) -> Dict[str, Tensor]:

        non_score_rewards = torch.zeros_like(responses, dtype=torch.float32)
        shaped_rewards = non_score_rewards.clone()
        # This introduces a small index off by one bug if pad_token_id == eos_token_id.
        terminal_positions = (responses != self.policy_tokenizer.pad_token_id).sum(dim=1) - 1
        shaped_rewards[list(range(rewards.size(0))), terminal_positions] += rewards
        return dict(shaped_rewards=shaped_rewards, non_score_rewards=non_score_rewards)

    def _estimate_mc_return(self, rewards: Tensor) -> Dict[str, Tensor]:

        if self.args.whiten_rewards:
            rewards = torch_ops.whiten(rewards, shift_mean=False)

        # Initialize return tensor
        returns = torch.zeros_like(rewards)
        gen_length = self.args.response_len

        # For each step
        for t in reversed(range(gen_length)):
            # If it's the last timestep, return is just the reward
            if t == gen_length - 1:
                returns[:, t] = rewards[:, t]
            else:
                returns[:, t] = rewards[:, t] + self.args.gamma * returns[:, t + 1]

        return dict(returns=returns)

    def _estimate_advantage(self, rewards: Tensor, values: Tensor) -> Dict[str, Tensor]:
        """Generalized advantage estimation.

        Reference:
            https://arxiv.org/abs/1506.02438
        """
        if self.args.whiten_rewards:
            rewards = torch_ops.whiten(rewards, shift_mean=False)
        lastgaelam = 0
        advantages_reversed = []
        gen_length = self.args.response_len

        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return dict(returns=returns)

    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, Tensor]:
        """Rollout trajectories with policy.

        Args:
            queries_data: Sequence of batches or DataLoader.
                Each batch is a dict with keys 'queries' and 'query_attn_masks'.

        Returns:
            Dictionary with keys
                'queries', 'query_attn_masks', 'responses',
                'logprobs', 'ref_logprobs', 'values',
                'rewards', 'non_score_rewards', 'shaped_rewards'.
        """
        # Give up dropout throughout.
        # self._make_fsdp_happy()

        self.ref_policy.eval()
        self.reward_model.eval()
        self.policy.train()

        rollouts = []
        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
            disable=not self.accelerator.is_main_process,
            desc="rollout",
        ):

            if self.args.static_dataset:
                # Get a batch of queries and responses from the dataset.
                queries, query_attn_masks, responses = common.unpack_dict(
                    common.prepare_inputs(batch, device=self.accelerator.device),
                    keys=("queries", "query_attn_masks", "responses"),
                )

                # Evaluate logprobs of the samples.
                rollouts_batch = {"queries": queries, "query_attn_masks": query_attn_masks, "responses": responses}
            else:
                # Sample rollouts using the reference policy.
                queries, query_attn_masks = common.unpack_dict(
                    common.prepare_inputs(batch, device=self.accelerator.device),
                    keys=("queries", "query_attn_masks"),
                )
                respond_outputs = self.ref_policy.respond(queries, query_attn_masks, temperature=self.args.temperature)
                (responses,) = common.unpack_dict(respond_outputs, ("responses",))

                # Evaluate logprobs of the samples.
                rollouts_batch = {"queries": queries, "query_attn_masks": query_attn_masks, "responses": responses}

            # Evaluate current estimated value of the samples.
            q_value_outputs = self.policy(**rollouts_batch)
            # q_value_outputs["values"] = torch.gather(q_value_outputs.pop("qvalues"), dim=2, index=responses.unsqueeze(-1)).squeeze(-1)
            rollouts_batch.update({"values": q_value_outputs["values"].squeeze(-1)})

            # Evaluate reward of the samples.
            text_queries, text_responses = tuple(
                self.policy_tokenizer.batch_decode(tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for tensor in (queries, responses)
            )
            del queries, responses  # Prevent mistakes.

            # We retokenizer, since policy and reward model might not have the same tokenizer.
            # TODO(lxuechen): Avoid retokenization when policy and reward tokenizer are the same.
            text_sequences = [q + r for q, r in utils.zip_(text_queries, text_responses)]
            # TODO(lxuechen): This response retokenization has issues with OPT, since the tokenizer always prepend
            #  <bos_token>. But the issue is local to post_reward, which isn't an issue if we don't penalize.
            responses = self.reward_tokenizer(text_responses, return_tensors="pt", padding=True, truncation=True)
            responses = common.prepare_inputs(responses, device=self.accelerator.device)

            # tokenizing for reward model with bs==1 and then collating
            reward_outputs_list = []
            for ts in text_sequences:
                s = self.reward_tokenizer(ts, return_tensors="pt", truncation=True)
                s = common.prepare_inputs(s, device=self.accelerator.device)
                r = self.reward_model(**s)
                reward_outputs_list.append(r.rewards)
            
            reward_outputs = torch.cat(reward_outputs_list, dim=0)
            reward_outputs = {'rewards': reward_outputs}

            del text_responses, text_queries # prevent mistakes

            reward_outputs = self.post_reward(reward_outputs, responses.input_ids)
            rollouts_batch.update(reward_outputs)

            # Shape reward with KL penalty.
            shape_reward_outputs = self._shape_reward(
                rewards=rollouts_batch["rewards"],
                responses=rollouts_batch["responses"],
            )
            rollouts_batch.update(shape_reward_outputs)

            rollouts_batch_cpu = {key: value for key, value in rollouts_batch.items()}
            rollouts.append(rollouts_batch_cpu)

        # Items in dict need to be of same shape.
        rollouts = common.merge_dict(rollouts, merge_fn=torch.cat)
        # Estimating advantages outside the loop gives more samples for reward normalization.
        rollouts["values"][rollouts['responses'] == self.policy_tokenizer.pad_token_id] = 0
        returns = self._estimate_advantage(
            rewards=rollouts["shaped_rewards"].to(self.accelerator.device),
            values=rollouts["values"].to(self.accelerator.device).detach(),
        )
        td_one_returns = self._estimate_mc_return(
            rewards=rollouts["shaped_rewards"].to(self.accelerator.device).detach(),
        )
        td_one_returns = {key+"_td_one": value for key, value in td_one_returns.items()}
        returns = {key: value for key, value in returns.items()}
        return {**rollouts, **returns, **td_one_returns}

    def post_reward(self, reward_outputs: Dict[str, Tensor], responses: Tensor) -> Dict[str, Tensor]:
        """Assign bad reward values to sequences which didn't stop properly."""
        if self.args.truncate_token_ids is None:
            return reward_outputs

        def get_validity_mask(sequences: Tensor, end_token_id: int) -> Tensor:
            """Mark a batch element as False if the sequence doesn't end with `end_token_id` after `truncate_after`."""
            assert sequences.dim() == 2
            validity_mask = []
            for sequence in sequences:
                (nonzeros,) = (sequence == end_token_id).nonzero(as_tuple=True)
                if len(nonzeros) == 0:
                    validity_mask.append(False)
                else:
                    validity_mask.append(
                        self.args.truncate_after is None
                        or
                        # Last occurrence of `end_token_id` is after `truncate_after`.
                        nonzeros[-1] > self.args.truncate_after
                    )
            return torch.tensor(validity_mask, device=sequences.device)

        validity_masks = [get_validity_mask(responses, end_token_id) for end_token_id in self.args.truncate_token_ids]
        validity_mask = torch.stack(validity_masks).any(dim=0)  # Sequence is valid if it ends with any end token.
        rewards = reward_outputs["rewards"]
        rewards[~validity_mask] = self.args.penalty_reward_value
        return reward_outputs

    def compute_loss(self, rollouts: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        self.policy.train()

        returns, returns_td_one, rewards, queries, query_attn_masks, responses = common.unpack_dict(
                rollouts,
                keys=("returns", "returns_td_one", "shaped_rewards", "queries", "query_attn_masks", "responses"),
            )
        batch_size = responses.size(0)

        if batch_size == 1:
            # remove all padding tokens. When pad_token_id == eos_token_id it will lead to index off by one bug
            returns = returns[responses != self.policy_tokenizer.pad_token_id].view(1, -1)
            returns_td_one = returns_td_one[responses != self.policy_tokenizer.pad_token_id].view(1, -1)
            rewards = rewards[responses != self.policy_tokenizer.pad_token_id].view(1, -1)
            responses = responses[responses != self.policy_tokenizer.pad_token_id].view(1, -1)
            query_attn_masks = query_attn_masks[queries != self.policy_tokenizer.pad_token_id].view(1, -1)
            queries = queries[queries != self.policy_tokenizer.pad_token_id].view(1, -1)

        # Compute the Q-values for the responses
        outputs = self.policy(queries, query_attn_masks, responses)
        q_values = outputs["qvalues"]
        q_values_logits = q_values / self.args.temperature
        q_preds = torch.gather(q_values, dim=2, index=responses.unsqueeze(-1)).squeeze(-1)
        # After gather, both Q and V have the expected values for every response token.

        with torch.no_grad():
            # Sample rollouts using the reference policy.
            if self.kl_ctl.value > 0:
                rollouts_batch = {"queries": queries, "query_attn_masks": query_attn_masks, "responses": responses}
                ref_policy_outputs = self.ref_policy(**rollouts_batch, temperature=self.args.temperature)
                logits, = common.unpack_dict(ref_policy_outputs, keys=("logits",))

            # Compute the Advantage term from https://arxiv.org/abs/2305.18161
            #probs = logits.softmax(dim=-1)
            #regularizer = self.args.gamma * (probs * outputs["next_advantage"].detach()).sum(dim=-1)
            #regularizer = torch.cat([regularizer[:, 1:], torch.zeros_like(regularizer[:, -1:])], dim=1)

            # Compute the KL-divergence between the reference policy and the Q-value induced policy
            # kl_div = (q_values_logits.softmax(dim=-1) * (
            #             q_values_logits.log_softmax(dim=-1) - logits.log_softmax(dim=-1))).sum(dim=-1).mean()

            # Compute the Q-values for the next states
            if self.args.q_head_type == 'q_and_v':
                target_values = q_preds.detach()
                target_q_values = returns  # lam=0.0 so this is TD(0) with value function as bootstrap
            else:
                target_values = returns_td_one
                target_q_values = returns

        if self.kl_ctl.value > 0:
            cql_loss = torch.sum(logits.softmax(dim=-1) * q_values_logits.log_softmax(dim=-1), dim=2).mean()

        qf_losses = (q_preds - target_q_values) ** 2.0
        if self.policy.args.q_head_type == 'dueling':
            values = outputs["values"].squeeze(-1)
            v_losses = (values - target_values) ** 2.0
            losses = qf_losses + v_losses
        elif self.policy.args.q_head_type == 'q_and_v':
            values = outputs["values"].squeeze(-1)
            tau = 0.8
            v_losses = asymmetric_l2_loss(target_values-values, tau)
            losses = qf_losses + v_losses
        else:
            losses = qf_losses
            v_losses = torch.zeros_like(qf_losses)

        # Taking care of padding in case of batch size >1
        loss_mask = responses != self.policy_tokenizer.pad_token_id
        denom = loss_mask.sum(dim=1)
        per_sample_loss = (losses * loss_mask.detach()).sum(dim=1) / denom

        loss = per_sample_loss.mean()
        if self.kl_ctl.value > 0:
            loss = loss + self.kl_ctl.value * cql_loss

        with torch.no_grad():
            entropy = -(q_values_logits.softmax(dim=-1) * q_values_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
            return_var = returns.var(unbiased=False)
            return_mean = (returns.sum(dim=1) / loss_mask.sum(dim=1)).mean()
            value_mean, value_var = q_preds.mean(), q_preds.var(unbiased=False)

        stats = dict(
            loss=dict(total=losses.mean(), qf=qf_losses.mean(), vf=v_losses.mean()),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                error=((q_preds - returns_td_one) ** 2).mean(),
                mean=value_mean,
                var=value_var,
                entropy=entropy,
                #kl_div=kl_div,
            ),
        )
        return loss, common.flatten_dict(stats, sep="/", postprocess_fn=lambda x: x.detach())

    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        shaped_rewards = rollouts["shaped_rewards"].sum(dim=1).mean(dim=0)
        non_score_rewards = rollouts["non_score_rewards"].sum(dim=1).mean(dim=0)
        rewards = rollouts["rewards"].mean(dim=0)
        stats = {
            f"objective/kl_sum_seq": 0.0,
            f"objective/shaped_rewards": shaped_rewards,
            f"objective/non_score_rewards": non_score_rewards,
            f"objective/rewards": rewards,  # Original model reward.
            f"objective/lr": self.optimizer.param_groups[0]["lr"],
        }
        for k, v in train_stats.items():
            stats[f"fqe/{k}"] = v.mean(dim=0)
        stats = {key: value.item() if torch.is_tensor(value) else value for key, value in stats.items()}
        if self.accelerator.is_main_process:
            self.accelerator.log(stats, step=step_idx)
            if self.args.output_dir is not None and self.args.save_rollouts:
                # Store rollout data to disk to debug.
                rollouts_to_disk = {
                    key: self.policy_tokenizer.batch_decode(
                        tensor, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    for key, tensor in common.unpack_dict(
                        rollouts, keys=("queries", "responses"), return_type=dict
                    ).items()
                }
                rollouts_to_disk = pd.DataFrame(rollouts_to_disk).to_dict(orient="records")
                utils.jdump(rollouts_to_disk, utils.join(self.args.output_dir, "rollouts", f"step_{step_idx}.json"))
        return stats

    @torch.inference_mode()
    def evaluate(self, step_idx: int, unwrapped_policy=None):
        """Evaluate by query the Q-values of the model on the evaluation dataset."""

        logger.warning(f"Start evaluation at step: {step_idx}", main_process_only=True)
        logger.warning(f"Number of evaluation steps: {len(self.eval_dataloader)}", main_process_only=True)

        aggregated_qvalue_loss = []
        aggregated_value_loss = []
        for batch in tqdm.tqdm(self.eval_dataloader,  disable=not self.accelerator.is_main_process, desc="Evaluating"):
            queries, query_attn_masks, target_values = common.prepare_inputs(
                common.unpack_dict(
                    batch,
                    keys=("queries", "query_attn_masks", "values"),
                ),
                device=self.accelerator.device,
            )

            batch_size = queries.shape[0]
            if batch_size == 1:
                # remove all padding tokens
                query_attn_masks = query_attn_masks[queries != self.policy_tokenizer.pad_token_id].view(1, -1)
                queries = queries[queries != self.policy_tokenizer.pad_token_id].view(1, -1)
            else:
                raise NotImplementedError("Batch size > 1 not supported yet.")

            # Start evaluation.
            self.policy.eval()
            self._make_fsdp_happy()  # we can keep this b/c it automatically checks if fsdp or not
            outputs = self.policy(queries, query_attn_masks, only_last=True)

            # Compute the Q-values for the responses
            q_values = outputs["qvalues"]
            last_token = queries[:, -1].unsqueeze(-1)
            q_values = torch.gather(q_values.squeeze(1), dim=1, index=last_token).squeeze(-1)

            # Compute the loss
            assert q_values.shape == target_values.shape, f"{q_values.shape} != {target_values.shape}"
            q_values_loss = (q_values - target_values) ** 2.0
            aggregated_qvalue_loss.append(q_values_loss)

            # If the head predict also values, we can compute the value loss.
            if self.policy.args.q_head_type == 'dueling':
                values = outputs["values"].squeeze(-1).squeeze(-1)
                assert values.shape == target_values.shape, f"{values.shape} != {target_values.shape}"
                values_loss = (values - target_values) ** 2.0
                aggregated_value_loss.append(values_loss)
            else:
                values_loss = torch.zeros_like(q_values_loss)
                aggregated_value_loss.append(values_loss)

        aggregated_value_loss = torch.cat(aggregated_value_loss, dim=0)
        aggregated_qvalue_loss = torch.cat(aggregated_qvalue_loss, dim=0)
        qvalue_loss = aggregated_qvalue_loss.mean()
        value_loss = aggregated_value_loss.mean()
        stats = {'eval/qvalue_mse': qvalue_loss, 'eval/value_mse': value_loss}
        if self.accelerator.is_main_process:
            self.accelerator.log(stats, step=step_idx)

    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None, give_rw_access=True, check_corrupted=True):
        # We don't use accelerator here because, we want to be frugal and only store the policy.
        # Moreover, we want easy loadability -- calling .from_pretrained on the folder. Full dump wouldn't allow this.

        # Logic:
        #   1. Retrieve the complete state dict of the wrapped model.
        #       (retrieving state dict of submodule can lead to loss of keys)
        #   2. Remove keys that are part of the value network.
        #   3. Rename keys that are part of the policy network, so that they match the naming standard.
        output_dir = self.args.output_dir if output_dir is None else output_dir
        utils.makedirs(output_dir)

        model, policy_tokenizer, reward_tokenizer = self.policy, self.policy_tokenizer, self.reward_tokenizer
        if self.accelerator.distributed_type == DistributedType.NO:
            state_dict = model.state_dict()
        else:
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                logger.warning("Gathering full state_dict...")
                state_dict = model.state_dict()
                logger.warning("Finished gathering full state_dict...")

        if self.accelerator.is_main_process:
            # Retain and remap policy keys.
            new_state_dict = dict()
            prefix = "base_model."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_state_dict[key[len(prefix) :]] = value
            state_dict = new_state_dict

            if check_corrupted:  # Let the checks run on GPU.
                is_corrupted = any(value.isnan().any().item() for value in state_dict.values())
                logger.warning(f"Is there nans in the state_dict to be dumped? {is_corrupted}")

            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict

            unwrapped = model.base_model
            peft_model_path = os.path.join(output_dir, "adapter_model")
            save_peft_model(unwrapped, peft_model_path)
            torch.save({'state_dict': model.q_head.state_dict()}, os.path.join(output_dir, "q_head.pt"))

            assert isinstance(
                unwrapped, (transformers.OPTForCausalLM, transformers.LlamaForCausalLM, peft.PeftModelForCausalLM)
            ), f"Expected to save a generative policy, but found model to be of type: {type(unwrapped)}."
            if hasattr(unwrapped, "_keys_to_ignore_on_save"):
                logger.warning(f"keys to ignore on save: {unwrapped._keys_to_ignore_on_save}")
            logger.warning(f"Saving model checkpoint to {output_dir}")
            logger.warning(f"Saving {len(cpu_state_dict)} keys:\n{utils.jdumps(cpu_state_dict.keys())}")
            unwrapped.save_pretrained(output_dir, state_dict=cpu_state_dict)

            policy_tokenizer.save_pretrained(output_dir)
            reward_tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, constants.TRAINING_ARGS_NAME))

            if give_rw_access:
                try:
                    os.system(f"chmod -R a+xwr {output_dir}")
                except Exception as e:
                    logger.fatal(f"Failed to give read-write access to {output_dir}: {e}")


def make_tokenizer(args):
    # policy_tokenizer left pads, since the policy requires batch decoding.
    policy_tokenizer = _make_padded_tokenizer(
        args.policy_model_name_or_path, cache_dir=args.cache_dir, use_fast=args.use_fast_tokenizer, padding_side='left',
    )
    # reward_tokenizer left pads, since we need the embedding of the right most non-pad token.
    reward_tokenizer = _make_padded_tokenizer(
        args.reward_model_name_or_path, cache_dir=args.cache_dir, use_fast=args.use_fast_tokenizer # use default padding side of rm
    )

    if policy_tokenizer.get_vocab() != reward_tokenizer.get_vocab():
        logger.info('Policy and reward tokenizers are different.')
        return [policy_tokenizer, reward_tokenizer]
    else:
        logger.info('Policy and reward tokenizers are the same.')
        return [policy_tokenizer, policy_tokenizer]


def make_models(
    tokenizer: List[transformers.PreTrainedTokenizer],
    args,
    accelerator: accelerate.Accelerator,
) -> dict:
    policy_tokenizer, reward_tokenizer = tokenizer

    # Model construction below seems convoluted, but it's made to trade time for RAM efficiency.
    # For large models, object creation could be extremely RAM intensive.
    # Especially so for multiple processes on single node, each starting off with a copy of the model.
    # General strategy is to 1) create a model, 2) move it to target device / shard it, 3) then start next model,
    # as opposed to creating all needed models on CPU first, and separately moving / sharding each.
    # policy = rl_models.make_policy_with_base_model(args, make_generative_policy(), tokenizer)
    if args.init_value_with_reward:
        # Initialize value from reward model a la OAI.
        logger.warning("Initializing value model with reward model.")
        qfunction_model = rl_models.make_qfunction_with_base_model(args, make_reward_model(args, accelerator, is_trainable=True).backbone_model, policy_tokenizer, accelerator)
    else:
        logger.warning("Initializing value model with policy model.")
        # Initialize value from policy. Works for sanity, but generally performs worse in instruction-following.
        # initializing value model with reward model won't work with encoder-decoder-based models
        qfunction_model = rl_models.make_qfunction_with_base_model(args, make_generative_policy(args, accelerator, is_trainable=True), policy_tokenizer, accelerator)

    qfunction_model = accelerator.prepare(qfunction_model)

    ref_policy = rl_models.make_policy_with_base_model(args, make_generative_policy(args, accelerator, is_trainable=False), policy_tokenizer)
    ref_policy.requires_grad_(False)
    ref_policy = accelerator.prepare(ref_policy)  # wrap again b/c only base model is wrapped with accelerator properly

    reward_model = make_reward_model(args, accelerator, is_trainable=False)
    reward_model.requires_grad_(False)
    # skipping accelerator prepare b/c done within make_reward_model

    # TODO: This is a hack to get FSDP running. Remove in the future when we figure things out.
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        inputs = tokenizer[0]("fsdp are you happy now??? :)" * 50, return_tensors="pt")
        inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
        qfunction_model(inputs["input_ids"], inputs["attention_mask"], inputs["input_ids"])

    return dict(qfunction_model=qfunction_model, ref_policy=ref_policy, reward_model=reward_model)
