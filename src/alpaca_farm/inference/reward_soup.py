from collections import OrderedDict

import torch

class PeftWeightAverager:
    @staticmethod
    def average_weights(base_model, peft_names, coefficients):
        weights_averaged = OrderedDict()
        i = 0
        for peft_name, coefficient in zip(peft_names, coefficients):
            if coefficient == 0.:
                continue
            if peft_name is None:
                print("Skipping none peft_name")
                continue
            current_model = Loader.load_peft_model(base_model, peft_name)
            assert LOAD_ONLY_LORA
            current_weights = get_peft_model_state_dict(current_model, state_dict=None)
            for key in list(current_weights.keys()):
                if i == 0:
                    weights_averaged[key] = coefficient * current_weights[key]
                else:
                    weights_averaged[key] += coefficient * current_weights[key]
                del current_weights[key]
            del current_model
            torch.cuda.empty_cache()
            i += 1
        return weights_averaged

    @staticmethod
    def build_wa(base_model, peft_names, coefficients):
        weights_averaged = WeightAverager.average_weights(
            base_model=base_model, peft_names=peft_names, coefficients=coefficients
        )

        torch.cuda.empty_cache()
        wa = Loader.load_peft_model(base_model, peft_names[0])
        wa.load_state_dict(weights_averaged, strict=not LOAD_ONLY_LORA)
        return wa