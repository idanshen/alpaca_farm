# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, List
import time
import math

import einops
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_utils import speed_metrics

from alpaca_farm import common, torch_ops


class Trainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
        input_ids, attention_mask, index_0, index_1, choice = common.unpack_dict(
            inputs, keys=("input_ids", "attention_mask", "index_0", "index_1", "choice")
        )
        num_candidates, num_pairs = input_ids.size(1), choice.size(1)
        input_ids_flat, attention_mask_flat = tuple(
            einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        )
        outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        rewards_flat = outputs.rewards
        rewards = einops.rearrange(rewards_flat, "(b c) -> b c", c=num_candidates)  # Size: (bsz, num_candidates).

        rewards_0, rewards_1 = tuple(
            torch_ops.batch_select(rewards, index) for index in (index_0, index_1)
        )  # Size: (bsz, num_pairs).
        logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).
        # Type casting of `choice` is due to amp.autocast context manager.
        loss = F.binary_cross_entropy_with_logits(logits, choice.to(logits.dtype), reduction="mean")
        return (loss, dict(logits=logits)) if return_outputs else loss

class CETrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
        input_ids, labels = common.unpack_dict(
            inputs, keys=("input_ids", "labels")
        )
        
        logits = model(input_ids=input_ids).rewards
        loss = F.cross_entropy(F.softmax(logits, dim=-1), labels, reduction="mean")
        
        return (loss, dict(logits=logits)) if return_outputs else loss


def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    logits = torch.tensor(eval_prediction.predictions).squeeze(-1)
    labels = torch.tensor(eval_prediction.label_ids[-1]).squeeze(-1)
    predictions = (logits >= 0.0).long()
    accuracy = predictions.eq(labels).float().mean().item()
    label_positive_rate = (labels == 1).float().mean().item()
    positive_rate = (predictions == 1).float().mean().item()
    true_positive_rate = (predictions * labels).float().sum().item() / labels.sum().item()
    false_positive_rate = (predictions * (1 - labels)).float().sum().item() / (1 - labels).sum().item()
    return dict(
        accuracy=accuracy,
        label_positive_rate=label_positive_rate,
        positive_rate=positive_rate,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
    )

class SoftPreferenceTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
        input_ids, attention_mask, labels = common.unpack_dict(
            inputs, keys=("input_ids", "attention_mask", "labels")
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.rewards
        
        loss = F.cross_entropy(F.softmax(logits, dim=-1), labels, reduction="mean")
        
        return (loss, dict(logits=logits)) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[List[Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # if eval dataset is torch dataset, then just run super method
        if isinstance(self.eval_dataset, list):
            # memory metrics - must set up as early as possible
            self._memory_tracker.start()

            start_time = time.time()
            outputs = []
            for ds in self.eval_dataset:
                eval_dataloader = self.get_eval_dataloader(ds)
                eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
                output = eval_loop(
                    eval_dataloader,
                    description="Evaluation",
                    # No point gathering the predictions if there are no metrics, otherwise we defer to
                    # self.args.prediction_loss_only
                    prediction_loss_only=False,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                )

                total_batch_size = self.args.eval_batch_size * self.args.world_size
                if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
                    start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
                # output.metrics.update(
                #     speed_metrics(
                #         metric_key_prefix,
                #         start_time,
                #         num_samples=output.num_samples,
                #         num_steps=math.ceil(output.num_samples / total_batch_size),
                #     )
                # )

                outputs.append(output)
            
            # compute metrics on both outputs
            metrics = compute_soft_preference_reward_modeling_metrics(outputs[0], outputs[1])

            # update runtime metrics
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=sum(output.num_samples for output in outputs),
                    num_steps=math.ceil(sum(output.num_samples for output in outputs) / total_batch_size),
                )
            )

            self.log(metrics)

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

            self._memory_tracker.stop_and_update_metrics(metrics)

            return metrics
        else:
            return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


def compute_soft_preference_reward_modeling_metrics(eval_prediction1: EvalPrediction, eval_prediction2) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    # human preferences
    assert (torch.tensor(eval_prediction1.label_ids[-1]) == torch.tensor(eval_prediction2.label_ids[-1])).all()

    logits1 = torch.tensor(eval_prediction1.predictions[:, 0]).squeeze(-1)
    logits2 = torch.tensor(eval_prediction2.predictions[:, 0]).squeeze(-1)
    logits = torch.stack([logits1, logits2], dim=-1)
    labels = torch.tensor(eval_prediction1.label_ids[-1]).squeeze(-1) # choice
    
    predictions = logits.argmax(dim=-1).long()
    accuracy = predictions.eq(labels).float().mean().item()
    label_positive_rate = (labels == 1).float().mean().item()
    positive_rate = (predictions == 1).float().mean().item()
    true_positive_rate = (predictions * labels).float().sum().item() / labels.sum().item()
    false_positive_rate = (predictions * (1 - labels)).float().sum().item() / (1 - labels).sum().item()

    llm_labels = torch.tensor(eval_prediction1.label_ids[0]).squeeze(-1)
    llm_labels = llm_labels.argmax(dim=-1).long()
    llm_accuracy = predictions.eq(llm_labels).float().mean().item()
    llm_label_positive_rate = (llm_labels == 1).float().mean().item()
    llm_positive_rate = (predictions == 1).float().mean().item()
    llm_true_positive_rate = (predictions * llm_labels).float().sum().item() / llm_labels.sum().item()
    llm_false_positive_rate = (predictions * (1 - llm_labels)).float().sum().item() / (1 - llm_labels).sum().item()

    return dict(
        accuracy=accuracy,
        label_positive_rate=label_positive_rate,
        positive_rate=positive_rate,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
        llm_accuracy=llm_accuracy,
        llm_label_positive_rate=llm_label_positive_rate,
        llm_positive_rate=llm_positive_rate,
        llm_true_positive_rate=llm_true_positive_rate,
        llm_false_positive_rate=llm_false_positive_rate,
    )