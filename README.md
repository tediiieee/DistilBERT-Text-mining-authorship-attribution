# DistilBERT-Text-mining-authorship-attribution

Dataset used: https://www.kaggle.com/azimulh/tweets-data-for-authorship-attribution-modelling/version/2
DistilBERT: https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation

dataset - Contains useful functions relating to the datasets.

feature_extraction_selection - Plots all models using best dataset and parameters. Used to compare feature extraction methods.
(Code inspired by: https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)

dataset_selection - Prints out the average accuracy for every dataset.

ds_exploration - Used to print the shapes of each dataset and plot the class distributions.

baseline_process -  Run gridsearch cv on all ml models (excluding BERT variants)

bert - Run gridsearch cv on all ml models (BERT variants)
(Code inspired by: https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)

ml -  Contains useful functions relating to the machine learning.

Project done for the TDDE16 - Text mining course at Linköpings university.
