import psstdata
import os
from datasets import load_dataset, Audio
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

file_path_column = "filename_old"


def change_file_paths(data_instance, data_dir: str):
    data_instance[file_path_column] = os.path.join(data_dir,
                                                   data_instance[file_path_column])
    return data_instance

def prepare_dataset(data_instance, processor: Wav2Vec2Processor):
    audio = data_instance[file_path_column]

    data_instance["input_values"] = processor(audio["array"],
                                              sampling_rate=audio["sampling_rate"])

    data_instance["input_length"] = len(data_instance["input_values"])
    with processor.as_target_processor():
        data_instance["labels"] = processor(data_instance["transcript"]).input_ids

    return data_instance


def main(input_dir: str, output_dir: str):
    dataset_dict = load_dataset('csv', data_files={
        "train": '/home/data1/psst-data-csv/train_utterances_excel.csv',
        "valid": '/home/data1/psst-data-csv/valid_utterances_excel.csv',
        "test": '/home/data1/psst-data-csv/test_utterances_excel.csv'
    })

    dict_ltr = {}
    for arpa, idx in psstdata.VOCAB_ARPABET.items():
        if arpa in (psstdata.PAD, psstdata.UNK):
            continue
        dict_ltr[arpa] = idx

    vocab_file = os.path.join(output_dir, 'psst_dict.json')
    with open(vocab_file, mode="w") as vocab_file_json:
        json.dump(dict_ltr, vocab_file_json)

    tokenizer = Wav2Vec2CTCTokenizer(vocab_file,
                                     unk_token=psstdata.UNK,
                                     pad_token=psstdata.PAD,
                                     word_delimiter_token='|')

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

    print("hellos")


if __name__ == "__main__":
    data_input_dir = "/home/data1/psst-data/psst-data-2022-03-02-full"
    data_output_dir = "/home/data1/psst-data-out"
    main(data_input_dir, data_output_dir)
