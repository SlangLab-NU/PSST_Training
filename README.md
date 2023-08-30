# PSST_Training

## Training the Model

Ensure git-lfs is installed in the conda envrionment by running ```conda install -c conda-forge git-lfs```

## Inference On Polaris

To run inference on Polaris, we will be running ```inference_script.py```. Before running the script, we need to make sure the configuration variables are up to date. Open ```config.yml``` and ensure the model that inference is being run on is correct. If the sampling rate or processes needs updating this can be updated here, but generally these will remain constant at sr = 16000 and num_proc = 4. 

![image](https://github.com/SlangLab-NU/PSST_Training/assets/105329387/aba74711-137c-419c-ba3d-d2cb9c5fbc4c)

With the appropriate model in the configuration file, to run inference simply run ```python inference_script.py``` with the venv activated. 

### Possible Errors

If there are errors loading the model from Hugging Face, the most likely errors are from the tokenizer_config.json or vocab.json not being loaded to the HF repo. Check the model's repo and verify that these files exist. These should be automatically uploaded when running the training script, but there are instances where this might not occur. If one or both files do not exist these can be manually uploaded to the repo.

## Inference On The Discovery Cluster

To decode a model and run inference, we can run the notebook ```run-inference.ipynb```

When importing a model, make sure you are pulling from the correct Hugging Face repo. Furthermore, ensure the paths to the psst data csv files are up to date.

### Updating the vocab

When running inference on a model we have to make sure we are using the same vocabulary as the model. Furthermore, our inference script requires a 'space' on either side of a phoneme. To ensure you are using the correct vocabulary, copy the vocabulary either from the vocab.json produced by the model script, or if you have run the training script recently, the vocabulary should be avaiable in a dict format.

![image](https://github.com/SlangLab-NU/PSST_Training/assets/105329387/d0a63c76-bd1f-4749-a819-a6f479387577)

Copy into inference script and add spaces like below.

![image](https://github.com/SlangLab-NU/PSST_Training/assets/105329387/95d40442-debb-481d-bb32-2ca64c1983ad)


Once the notebook has been run, tsv file(s) will be written to an ```out``` directory. These will contain the asr predictions for the respective valid, train, or test datasets. The created tsv files will then be used to run evaluation on.

## Running PSST Eval

If running evaluation on the NEU cluster, ensure the following packages are installed in the conda environment.

With the conda environment activated
```conda install pip```

To install the pssteval library run

```/<conda env>/bin/pip install phonologic```

```/<conda env>/bin/pip install psstdata```

```/<conda env>/bin/pip install pssteval```

With the pssteval library installed, we can run the the command
```pssteval-asr --out-dir path/to/directory/to/write/to path/to/decode/*.tsv```

Operating in the work/van-speech-nlp directory on the discovery cluster will require modification to the backend code in the psstdata library. To run the asr evaluation, the code needs the path to the psst-data dataset. This can be seen below in the file loading.py, which is accessed at: ```<conda env>/lib/<python version>/site-packages/psstdata/loading.py```

![image](https://github.com/SlangLab-NU/PSST_Training/assets/105329387/8fa25b4e-32db-4596-8d53-d3b24b6270b1)

The repo for the psst baseline and eval can be found at: https://github.com/PSST-Challenge/psstbaseline
