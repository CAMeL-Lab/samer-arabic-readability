# Strategies for Arabic Readability Modelling

This repository contains code and experiments to reproduce the results in our paper [Strategies for Arabic Readability Modelling](https://arxiv.org/pdf/2407.03032).

## Requirements:

The code for our experiments was written for Python >= 3.9, pytorch-lightning >= 2.2.2, transformers >= 4.29.0 and camel_tools >= 1.5.2. You can easily set up the environment using conda:

```bash
conda create -n readability python=3.9
conda activate readability

pip install -r requirements.txt
```

Our paper also requires the [CAMeL Tools datasets](https://github.com/CAMeL-Lab/camel_tools) for disambiguation. The script `data/setup_cameldata.sh` will set the datasets up for you.

## Experiments and Reproducibility:

This repository is organized as follows:
- data: includes all the data used throughout our paper to train and test different models.
- data_preparation: includes scripts to prepare the raw data.
- models: includes scripts to train and evaluate all of our single models. These scripts save the models and output their decisions on the test and development sets of our dataset.
- tuning_experiments: includes notebooks used in the development of the paper to tune certain parameters of the models used.
- final_experiments: includes notebooks that report on the results of our layered experiments.

#### Reproducing our results

Our paper is organized in terms of *layered experiments* of different models. Therefore, we first prepare the data and individual models, and then combine them in the final experiments.

We include two scripts to do this:
1. `data_preparation/run_data_prep.sh` prepares all data needed.
2. `models/train_and_test_models.sh` saves models and results on our dataset.

Then, run the `final_techniques.ipynb` notebook to see the comparisons between the layered techniques.

## License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.


## Citation:
If you find the code or data in this repo helpful, please cite our [paper](https://arxiv.org/pdf/2407.03032):

```BibTeX
@inproceedings{liberato-etal-2024-strategies,
    title = "Strategies for Arabic Readability Modeling",
    author = "Pineros Liberato, Juan  and
      Alhafni, Bashar and
      Al Khalil, Muhamed  and
      Habash, Nizar",
    booktitle = "Proceedings of ArabicNLP 2024"
    month = "aug",
    year = "2024",
    address = "Bangkok, Thailand",
    abstract = "Automatic readability assessment is relevant to building NLP applications for education, content analysis, and accessibility. However, Arabic readability assessment is a challenging task due to Arabic's morphological richness and limited readability resources. In this paper, we present a set of experimental results on Arabic readability assessment using a diverse range of approaches, from rule-based methods to Arabic pretrained language models. We report our results on a newly created corpus at different textual granularity levels (words and sentence fragments). Our results show that combining different techniques yields the best results, achieving an overall macro F1 score of 86.7 at the word level and 87.9 at the fragment level on a blind test set. We make our code, data, and pretrained models publicly available.",
}
```
