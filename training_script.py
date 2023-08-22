import psstdata
import os
from datasets import load_dataset, Audio
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

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
    with open(vocab_file) as vocab_file:
        json.dump(dict_ltr, vocab_file)

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

    train_dataset = dataset_dict["train"].cast_column("filename_new",
                                                      Audio(sampling_rate=16000))
    valid_dataset = dataset_dict["valid"].cast_column("filename_new",
                                                      Audio(sampling_rate=16000))
    test




if __name__ == "__main__":
    data_input_dir = "/home/data1/psst-data"
    data_output_dir = "/home/data1/psst-data-out"
    main(data_input_dir, data_output_dir)
