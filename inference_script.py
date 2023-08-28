import os
import csv
import librosa
import numpy as np
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, load_metric, DatasetDict, Dataset, Audio
import torch

file_path_columns_to_remove = ["aq_index", "test", "duration_frames", "filename_new"]
file_path_column = "filename_old"


def change_file_paths(data_instance, data_dir: str):
    data_instance[file_path_column] = os.path.join(data_dir,
                                                   data_instance[file_path_column])
    return data_instance


def prepare_dataset(data_instance, processor: Wav2Vec2Processor):
    audio = data_instance[file_path_column]

    # Extract the values from the audio files
    data_instance["input_values"] = processor(audio["array"],
                                              sampling_rate=audio["sampling_rate"]).input_values[0]
    data_instance["input_length"] = len(data_instance["input_values"])

    # Encode the transcript to label ids
    with processor.as_target_processor():
        data_instance["labels"] = processor(data_instance["transcript"]).input_ids

    # Remove all columns except for transcript
    data_instance = {key: data_instance[key] for key in data_instance.keys() if key == 'transcript'}

    return data_instance


def prepare_dictionary(processor):
    """
    Extracts the vocab.json file from hugging face model, flips the key value pairs for decoding and assigns the
    dictionary to the tokenizer decoder
    :param processor: The Wav2Vec2Processor from a pretrained model
    """
    vocab = processor.tokenizer.get_vocab()

    vocab = {value: key for key, value in vocab.items()}
    for key, value in vocab.items():
        if "<" not in value and "?" not in value:
            vocab[key] = " " + value + " "
    processor.tokenizer.decoder = vocab


def get_config_file():
    """
    Reads in the config.yml file containing variables for model inference
    :return: the configuration file
    """
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)
    return cfg["inference"]


def predictions_list(dataset, processor, device, model):
    res = []
    for i in range(len(dataset['transcript'])):
        input_values = np.array(dataset['input_values'][i])
        sampling_rate = dataset['input_length'][i]

        # Resample the input speech to match the model's sampling rate
        input_values = librosa.resample(input_values, orig_sr=sampling_rate, target_sr=16000)
        input_values = processor(input_values, sampling_rate=16000, return_tensors="pt").input_values
        input_values = input_values.to(device)  # Move input to the same device as the model

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0], clean_up_tokenization_spaces=False)

        # Uncomment print to see predicted logits
        print(predicted_ids[0])
        prediction = transcription.lstrip().rstrip().replace('  ', ' ').replace('\t', ' ')

        res.append(prediction)

        # Uncomment print statements to view predictions and reference
        reference_transcription = dataset['transcript'][i]
        print("Utterance Id:", dataset['utterance_id'][i])
        print("Reference:", reference_transcription)
        print("Prediction:", prediction)
        print("---")
    return res


def write_tsv(output_dir, filename, dataset, dataset_predictions):
    with open(os.path.join(output_dir, filename), "w") as f:
        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerow(("utterance_id", "asr_transcript"))
        for i in range(len(dataset)):
            utterance_id = dataset['utterance_id'][i]
            writer.writerow((utterance_id, dataset_predictions[i]))


def main(input_dir: str, output_dir: str):
    inference_config = get_config_file()

    model = Wav2Vec2ForCTC.from_pretrained(inference_config["model"])
    processor = Wav2Vec2Processor.from_pretrained(inference_config["model"])

    processor.decode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load the datasets and observe the structure
    dataset_dict = load_dataset('csv', data_files={
        "train": '/home/data1/psst-data-csv/train_utterances_excel.csv',
        "valid": '/home/data1/psst-data-csv/valid_utterances_excel.csv',
        "test": '/home/data1/psst-data-csv/test_utterances_excel.csv'
    })

    for dataset in ("train", "valid", "test"):
        data_dir = os.path.join(data_input_dir, dataset)
        dataset_dict[dataset] = dataset_dict[dataset].map(change_file_paths,
                                                          fn_kwargs={"data_dir": data_dir})
        dataset_dict[dataset] = dataset_dict[dataset].remove_columns(file_path_columns_to_remove)
        dataset_dict[dataset] = dataset_dict[dataset].cast_column(
            file_path_column, Audio(sampling_rate=inference_config["sampling_rate"]))
        dataset_dict[dataset] = dataset_dict[dataset].map(prepare_dataset,
                                                          num_proc=inference_config["num_proc"],
                                                          fn_kwargs={"processor": processor})

    prepare_dictionary(processor)

    print("Inference running...")
    test_predictions = predictions_list(dataset_dict["test"], processor, device, model)
    write_tsv(output_dir, "test.tsv", dataset_dict["test"], test_predictions)


if __name__ == "__main__":
    data_input_dir = "/home/data1/psst-data/psst-data-2022-03-02-full"
    data_output_dir = "/home/data1/psst-data-out"
    main(data_input_dir, data_output_dir)
