import os
import csv
import librosa
import numpy as np
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, Audio
import torch

file_path_columns_to_remove = ["aq_index", "test", "duration_frames", "filename_new"]
file_path_column = "filename_old"


def change_file_paths(data_instance, data_dir: str):
    """
    Updates the file path for the dir of the audio files
    :param data_instance: The test, train or valid dataset to update
    :param data_dir: The directory to the audio files
    :return: THe updated dataset
    """
    data_instance[file_path_column] = os.path.join(data_dir,
                                                   data_instance[file_path_column])
    return data_instance


def prepare_dataset(data_instance, processor: Wav2Vec2Processor):
    """
    Modifies the training, test, or validation dataset dictionary to contain the audio values and
    speech transcripts.
    :param data_instance: The test, validation, or train dataset
    :param processor: The Wav2Vec2Processor from the pretrained model
    :return: The modified dataset
    """

    # Load the audio data into batch
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


def predictions_list(dataset, processor, model):
    """
    Generates a series of predictions for a NLP model. Predictions are based off of torch logits and assigned to
    a phoneme. The prediction pulls the utterance ID and the prediction.
    There are a series of print statements in this script that can be used to visualize the predictions from the
    model. These SHOULD be commented out when running the final version, but for running a sample test this is
    good to have to verify how well the model is working
    :param dataset: The train, test, or validation dataset to run evaluation on
    :param processor: The Wav2Vec2 processor from the trained model
    :param model: The model the predictions are being based on
    :return: A list containing the utterance ID of a given speaker and the predicted phonemes of a word they
             have spoken.
    """
    res = []

    pipe = pipeline(task="automatic-speech-recognition", model=model, tokenizer=processor,
                    feature_extractor=processor.feature_extractor, device=0)

    # Print prediction if you want to see predicted output before running full inference
    for i in range(len(dataset['input_values'])):
        prediction = pipe(np.array(dataset['input_values'][i]))
        prediction = prediction["text"]
        prediction = prediction.lstrip().rstrip().replace('  ', ' ').replace('\t', ' ')
        res.append(prediction)

    return res


def write_tsv(output_dir, filename, dataset, dataset_predictions):
    """
    Writes the output from predictions_list() into a tsv file. This is the format required to calculate PER
    used by the PSST Baseline evaluation
    :param output_dir: The directory to write the tsv to
    :param filename: The name of the tsv file
    :param dataset: The dataset the predictions were based off
    :param dataset_predictions: The predictions list generated from predictions_list()
    :return: A tsv file containing the utterance ID and prediction for that utterance
    """
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

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # Load the datasets and observe the structure
    dataset_dict = load_dataset('csv', data_files={
        "train": '/home/data1/psst-data-csv/train_utterances_excel.csv',
        "valid": '/home/data1/psst-data-csv/valid_utterances_excel.csv',
        "test": '/home/data1/psst-data-csv/test_utterances_excel.csv'
    })

    for dataset in ("valid", "test"):
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
    test_predictions = predictions_list(dataset_dict["test"], processor, model)
    write_tsv(output_dir, "test.tsv", dataset_dict["test"], test_predictions)


if __name__ == "__main__":
    data_input_dir = "/home/data1/psst-data/psst-data-2022-03-02-full"
    data_output_dir = "/home/data1/psst-data-out"
    main(data_input_dir, data_output_dir)
