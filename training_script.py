import psstdata
import os
import numpy as np
import yaml
import torch
from datasets import load_dataset, Audio, load_metric
import json
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC,
                          TrainingArguments, Trainer)
from data_collator_ctc_with_padding import DataCollatorCTCWithPadding

file_path_column = "filename_old"


def change_file_paths(data_instance, data_dir: str):
    data_instance[file_path_column] = os.path.join(data_dir,
                                                   data_instance[file_path_column])
    return data_instance


def prepare_dataset(data_instance, processor: Wav2Vec2Processor):
    audio = data_instance[file_path_column]

    data_instance["input_values"] = processor(audio["array"],
                                              sampling_rate=audio["sampling_rate"]).input_values[0]

    data_instance["input_length"] = len(data_instance["input_values"])
    with processor.as_target_processor():
        data_instance["labels"] = processor(data_instance["transcript"]).input_ids

    return data_instance


def compute_metrics(pred, processor):
    cer_metric = load_metric("cer", revision="master")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def main(input_dir: str, output_dir: str):
    dataset_dict = load_dataset('csv', data_files={
        "train": '/home/data1/psst-data-csv/train_utterances_excel.csv',
        "valid": '/home/data1/psst-data-csv/valid_utterances_excel.csv',
        "test": '/home/data1/psst-data-csv/test_utterances_excel.csv'
    })

    dict_ltr = {}
    for arpa, idx in psstdata.VOCAB_ARPABET.items():
        dict_ltr[arpa] = idx

    # HF requires pad token to be a part of the dictionary, compared to fairseq where idx 0 is reserved for <pad>
    vocab_file = os.path.join(output_dir, 'psst_dict.json')
    with open(vocab_file, mode="w") as vocab_file_json:
        json.dump(dict_ltr, vocab_file_json)

    tokenizer = Wav2Vec2CTCTokenizer(vocab_file,
                                     unk_token=psstdata.UNK,
                                     pad_token=psstdata.PAD,
                                     word_delimiter_token='|',)

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=False)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                  tokenizer=tokenizer)

    for dataset in ("train", "valid", "test"):
        data_dir = os.path.join(data_input_dir, dataset)
        dataset_dict[dataset] = dataset_dict[dataset].map(change_file_paths,
                                                          fn_kwargs={"data_dir": data_dir})
        dataset_dict[dataset] = dataset_dict[dataset].cast_column(file_path_column,
                                                                  Audio(sampling_rate=16000))
        dataset_dict[dataset] = dataset_dict[dataset].map(prepare_dataset,
                                                          remove_columns=dataset_dict[dataset].column_names,
                                                          num_proc=4,
                                                          fn_kwargs={"processor": processor})

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    with open("config.yml", "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    model_config = cfg["model"]
    training_config = cfg["training_arguments"]

    model = Wav2Vec2ForCTC.from_pretrained(
        model_config["pretrained"],
        ctc_loss_reduction=model_config["ctc_loss_reduction"],
        pad_token_id=processor.tokenizer.pad_token_id,
        attention_dropout=float(model_config["attention_dropout"]),
        hidden_dropout=float(model_config["hidden_dropout"]),
        feat_proj_dropout=float(model_config["feat_proj_dropout"]),
        layerdrop=float(model_config["layerdrop"]),
        mask_time_prob=float(model_config["mask_time_prob"]),
        mask_feature_length=int(model_config["mask_feature_length"]),
        mask_feature_prob=float(model_config["mask_feature_prob"]),
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_extractor()
    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        group_by_length=training_config["group_by_length"],
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),  # added
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        evaluation_strategy=training_config["evaluation_strategy"],
        num_train_epochs=int(training_config["num_train_epochs"]),
        fp16=training_config["fp16"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        save_steps=int(training_config["save_steps"]),
        eval_steps=int(training_config["eval_steps"]),
        logging_steps=training_config["logging_steps"],
        learning_rate=float(training_config["learning_rate"]),  # changed from 3e-4
        weight_decay=float(training_config["weight_decay"]),
        warmup_steps=int(training_config["warmup_steps"]),  # changed from 200
        adam_epsilon=float(training_config["adam_epsilon"]),
        adam_beta1=float(training_config["adam_beta1"]),
        adam_beta2=float(training_config["adam_beta2"]),
        save_total_limit=training_config["save_total_limit"],
        push_to_hub=training_config["push_to_hub"],
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["valid"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()


if __name__ == "__main__":
    data_input_dir = "/home/data1/psst-data/psst-data-2022-03-02-full"
    data_output_dir = "/home/data1/psst-data-out"
    main(data_input_dir, data_output_dir)
