This repo contains code used for the paper "Selective Classifier Based Search Space Shrinking for Radiographs Retrieval" presented on "Machine Learning in Medical Imaging (MLMI 2024)"

Data used in this study is publicly available and can be obtained on the following page: https://publications.rwth-aachen.de/record/667225



Code is organised as a sequence of notebooks which are used to preprocess, train and evaluate model as described in the paper.


"1. IRMA data preparation.ipynb" - This notebook will preprocess all images from training and test sets resulting in a two *.mat files containing resized images and their respective labels.

"2. Model Training.ipynb" - This notebooks performs model training. Since the experiments were performed on a workstation having 128GB RAM, this script is made with assumption that the whole training set can fit in RAM. If the training will be performed on the different setup, then it would be required to write a custom data loader.

"3. Evaluation.ipynb" -  Evaluate model on test set for both selective and regular classifier.

"4. Retrieval.ipynb" - Perform retrieval"

Other important files:

code_eval.py - functionality used to evaluate the results. Original script was downloaded from the official imageclef page (http://www.imageclef.org/system/files/evalscript_1.tgz) and it is adapted to be used with Python 3.
