from transformers import TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl
import re
import math

class FreezingCallback(TrainerCallback):
    def __init__(self, freezing_schedule: list, trainer: Trainer, model_config: dict):
        self.model_config = model_config
        self.trainer = trainer
        self.freezing_schedule = freezing_schedule
        self.current_step_idx = 0
        self.model = trainer.model

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Check if current epoch is greater than or equal to the target epoch 
        if state.epoch >= self.freezing_schedule[self.current_step_idx][1]:
            self.current_step_idx += 1
            # Called to unfreeze layers
            self.freeze_model(self.freezing_schedule[self.current_step_idx][0], self.model_config["n_layer"], int(state.epoch))

    def on_save(self, args, state, control, **kwargs):
        for name, param in self.trainer.model.named_parameters():
            param.requires_grad = True

    def freeze_model(self, freeze_to, highest_layer, epoch):
        print(f"\nEpoch {epoch}: Freezing model to layer {freeze_to} of {highest_layer} layers.")

        for name, param in self.trainer.model.named_parameters():
            # Find out the number of every layer
            try:
                layer_number = int(re.search(r'\.h\.\d+\.', name).group().strip(".h"))
            except AttributeError:
                layer_number = math.inf
            # Freeze all layers up to layer `freeze_to`, including embedding layers.
            if '.wte.' in name or '.wpe.' in name or layer_number <= freeze_to:
                param.requires_grad = False
