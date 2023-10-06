# PSST_Training

## Installation

First clone the repo and cd into it's directory.

```bash
git clone git@github.com:SlangLab-NU/PSST_Training.git
cd PSST_Training
```

If you are not able to do git clone via ssh (and not https), you might have to set up your ssh tokens correctly. Follow the [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to do so.


Once your SSH key is created will need to [add your new SSH key to your github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

Create a virtual-env
```bash
conda create --name psst-training python=3.8 \
  && conda activate psst-training
```

Next install the dependencies from the requirements-hf.txt file.

'''bash
pip install -r requirements-hf.txt
'''

Lastly, make sure to store your own huggingface credentials in the cache directory.

This will store your access token in your Hugging Face cache folder (~/.cache/ by default):

'''bash
huggingface-cli login
'''

Then use notebook_login to sign-in to the Hub, and follow the link [here](huggingface.co/settings/tokens) to generate a token to login with
'''bash
>>> from huggingface_hub import notebook_login
>>> notebook_login()

## Inference On Polaris

To run inference on Polaris, we will be running ```python inference_script.py```. Before running the script, we need to make sure the configuration variables are up to date. Open ```config.yml``` and ensure the model that inference is being run on is correct. If the sampling rate or processes needs updating this can be updated here, but generally these will remain constant at sr = 16000 and num_proc = 4. 

![image](https://github.com/SlangLab-NU/PSST_Training/assets/105329387/aba74711-137c-419c-ba3d-d2cb9c5fbc4c)

With the appropriate model in the configuration file, to run inference simply run ```python inference_script.py```

### Possible Errors

If there are errors loading the model from Hugging Face, the most likely errors are from the tokenizer_config.json or vocab.json not being loaded to the HF repo. Check the model's repo and verify that these files exist. These should be automatically uploaded when running the training script, but there are instances where this might not occur. If one or both files do not exist these can be manually uploaded to the repo.

If there are token errors logging into hugging face make sure to revisit the installation steps. You may have missed something.


Once the file has been run, tsv file(s) will be written to an ```out``` directory. These will contain the asr predictions for the respective valid, train, or test datasets. The created tsv files will then be used to run evaluation on.

## Training the Model

Ensure git-lfs is installed in the conda envrionment by running ```conda install -c conda-forge git-lfs```

To run inference on Polaris, we will be running ```python training_script.py```.

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
