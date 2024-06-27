# Strategies for Arabic Readability Modelling

This repository containes code and experiments to reproduce the results in our paper Strategies for Arabic Readability Modelling.

### Requirements

The code for our experiments was written for Python >= 3.9, pytorch-lightning >= 2.2.2, and camel_tools >= 1.5.2. You can easily set up the environment using conda:

```
conda create -n gec python=3.9
conda activate gec

pip install -r requirements.txt

### todo script rto do camel tools.

```

The frequency analysis side of our paper requires the the (CAMeL Tools datasets)[https://github.com/CAMeL-Lab/camel_tools]. The script above will download these resources for you.

### Experiments and Reproducibility

This repository is organized as follows:
- data: includes all the data used throughout our paper to train and test different models.
- data_preparation: includes scripts to 
- models: includes scripts to train and evaluate all of our single models. These scripts save the models and output their decisions on the test and development sets of our dataset.
- tuning: includes notebooks used to tune certain parameters of the frequency, lex, and MLE models.
- final_experiments: includes a Python notebook that visualizes all the results of our experiments.

#### Reproducing our results

Our paper is organized in terms of *layered experiments* of different models. Therefore, we first prepare the data and individual models, and then combine them in the final experiments.

We include three scripts to do this:
1. ----- prepares all data needed.
2. ----- saves models and results on our dataset.
3. ----- visualizes the different layered models.

### License
This repo is available under the MIT license.

### Citation



