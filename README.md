# PSST_Training

## Training the Model

Ensure git-lfs is installed in the conda envrionment by running ```conda install -c conda-forge git-lfs```

## Inference

To decode a model and run inference, we can run the notebook ```run-inference.ipynb```

When importing a model, make sure you are pulling from the correct Hugging Face repo. Furthermore, ensure the paths to the psst data csv files are up to date.

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

Operating in the work/van-speech-nlp directory on the discovery cluster will require modification to the backend code in the pssteval library. To run the asr evaluation, the code needs the path to the psst.data dataset. This can be seen below in the file loading.py, which is accessed at: ```path/to/psst-data```

![image](https://github.com/SlangLab-NU/PSST_Training/assets/105329387/8fa25b4e-32db-4596-8d53-d3b24b6270b1)

The repo for the psst baseline and eval can be found at: https://github.com/PSST-Challenge/psstbaseline
