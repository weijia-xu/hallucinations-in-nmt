# Understanding and Detecting Hallucinations in Neural Machine Translation via Model Introspection
Data, model and source code for the TACL paper:

[**Understanding and Detecting Hallucinations in Neural Machine Translation via Model Introspection**](https://arxiv.org/abs/2301.07779)

Weijia Xu, Sweta Agrawal, Eleftheria Briakou, Marianna J. Martindale, Marine Carpuat

## Prerequisite
* Clone [this repository](https://github.com/lena-voita/the-story-of-heads) and follow the installation instructions.
* Put the [code files](./code) in the ``the-story-of-heads`` directory.

## Evaluation Data
The corpora in [data](./data) contain 408 English-Chinese and 423 German-English translations annotated for detached hallucinations (``{hallucinated: 1, non-hallucinated: 0}``). Each line contains the source sentence, model translation, and the label separated by tabs. The guidelines for annotation can be found in the paper.

## Models
We release the models used to produce the translations: [German-English](https://obj.umiacs.umd.edu/hallucinations-in-nmt/model-ende.npz), [English-Chinese](https://obj.umiacs.umd.edu/hallucinations-in-nmt/model-enzh.npz). The models are trained based on [this codebase](https://github.com/lena-voita/the-story-of-heads). We also release the BPE files for data preprocessing [here](https://obj.umiacs.umd.edu/hallucinations-in-nmt).

## Hallucination Detector
* ``code/extract-token-contributions.py`` contains the python scripts for extracting the relative token contributions given an input file that contains the source and translation pairs (tab-separated) and dump the results into a pickle file.
* ``code/classifier.py`` contains the code to train and test the classifier based on the features extracted from relative token contributions.

You may need to change the paths on top of each script before running.