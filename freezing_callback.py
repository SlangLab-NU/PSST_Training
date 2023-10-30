from transformers import TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl
import re
import math

class FreezingCallback(TrainerCallback):
    def __init__(self, freezing_schedule: list, trainer: Trainer, model_config: dict, num_hidden_layers: int):
        self.model_config = model_config
        self.trainer = trainer
        self.freezing_schedule = freezing_schedule
        self.current_step_idx = 0
        self.model = trainer.model
        self.num_hidden_layers = num_hidden_layers


    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Check if current epoch is greater than or equal to the target epoch 
        if state.epoch >= self.freezing_schedule[self.current_step_idx][1]:
            self.current_step_idx += 1
            # Called to unfreeze layers
            self.freeze_model(self.freezing_schedule[self.current_step_idx][0], int(state.epoch))

    def on_save(self, args, state, control, **kwargs):
        for name, param in self.trainer.model.named_parameters():
            param.requires_grad = True

    def freeze_model(self, unfreeze_layers, epoch):
        print(f"\nEpoch {epoch}: Unfreezing layers from {unfreeze_layers} to {self.num_hidden_layers}.")

        for name, param in self.trainer.model.named_parameters():
            # Find out the number of every layer
            try:
                layer_number = int(re.search(r'\.h\.\d+\.', name).group().strip(".h"))
            except AttributeError:
                layer_number = math.inf
            # Unfreeze layers from `unfreeze_layers` to `self.num_hidden_layers`.
            if '.wte.' in name or '.wpe.' in name or (6 <= layer_number <= self.num_hidden_layers and layer_number >= unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
