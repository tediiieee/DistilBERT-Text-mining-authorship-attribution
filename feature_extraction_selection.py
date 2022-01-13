# Plots all models using best dataset and parameters. Used to compare feature extraction methods.
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import time as time
from tqdm import tqdm
#%tensorflow_version 1.x
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from dataset import *
from ml import *
from bert import bert

def main():
    #Load data
    path = "tweet.csv"
    tweets = load_data(path)

    #Remove rem_retweets
    tweets = rem_retweets(tweets)

    #Under/oversample to create datasets
    tweets_o = even_dist(tweets, "oversample")[0]

    features = tweets_o["Tweet description"].to_numpy()

    # Run BERT
    features_bert = bert(tweets_o)
    
    # Get labels
    labels = tweets_o["Name"].to_numpy()

    #test/train
    tr_bert, te_bert, tr, te, tr_lab, te_lab = train_test_split(features_bert, features, labels, test_size=0.2, random_state=42, stratify=labels)

    #Setup multiprocess
    nb_workers = 6 #6 models
    pool = Pool(processes=nb_workers)

    # Define vectorizers
    cv = CountVectorizer(stop_words = "english")
    tfid = TfidfVectorizer(stop_words = "english")

    # Define models
    lr = LogisticRegression(C = 10)
    rf = RandomForestClassifier(n_estimators=200)
    rf_100 = RandomForestClassifier()

    # Define pipelines and model parameters
    p1 = Pipeline([('cv', cv), ('lr', lr)])
    p2 = Pipeline([('cv', cv), ('rf', rf_100)])
    p3 = Pipeline([('tfid', tfid), ('lr', lr)])
    p4 = Pipeline([('tfid', tfid), ('rf', rf)])
    p5 = lr #BERT
    p6 = rf #BERT
    nr_models = 6
    m_names = ["CountVectorizer, Logistic regression","CountVectorizer, Random forest", "TfidVectorizer, Logistic regression",
        "TfidVectorizer, Random forest", "DistillBERT, Logistic regression", "DistillBERT, Random forest"]

    res = [[] for i in range(nr_models)]

    #Define stepwise loop to sample data from the dataset.
    max_ind = len(tr_lab) - 1
    step_size = 1000
    ds_sizes = range(step_size, round(max_ind, -3) + step_size, step_size)
    #Loop through all sampled subsets
    for size in tqdm(ds_sizes, desc = "Subset dataset"):
        # Sample 'size' amount of training samples
        idx = np.random.randint(max_ind, size=size)
        sub_tr = tr[idx]
        sub_bert_tr = tr_bert[idx,:]
        sub_tr_lab = tr_lab[idx]

        # Define pipe parameters
        pipes = [(p1,sub_tr,sub_tr_lab, te, te_lab), (p2,sub_tr,sub_tr_lab, te, te_lab), 
            (p3,sub_tr,sub_tr_lab, te, te_lab), (p4,sub_tr,sub_tr_lab, te, te_lab), (p5,sub_bert_tr,sub_tr_lab, te_bert, te_lab), (p6,sub_bert_tr,sub_tr_lab, te_bert, te_lab)]

        #Fit model and predict on test data
        print(f"-------------Training models with oversampled dataset")
        models = pool.imap(fit_and_pred_fes, pipes)
        for i, acc in tqdm(enumerate(models), total=len(pipes),desc = "Model training"):
            res[i].append(acc)

    # Plot
    plt.style.use('ggplot')

    COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    for i, m_acc in enumerate(res):
        plt.plot(ds_sizes,m_acc,label=m_names[i], color = COLORS[i])

    plt.legend()
    plt.title("Feature extraction comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Training samples")
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))    