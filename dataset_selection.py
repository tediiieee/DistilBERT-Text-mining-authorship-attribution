# Prints out the average accuracy for every dataset.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import numpy as np

from dataset import *
from ml import *

def main():
    print("-------------Program started")
    print("-------------Loading data")
    #Load data
    path = "tweet.csv"
    tweets_rt = load_data(path)

    #Remove rem_retweets
    tweets = rem_retweets(tweets_rt)

    #Under/oversample to create datasets
    tweets_u = even_dist(tweets, "undersample")[0]
    tweets_o = even_dist(tweets, "oversample")[0]
    tweets_u_rt = even_dist(tweets_rt, "undersample")[0]
    tweets_o_rt = even_dist(tweets_rt, "oversample")[0]

    #Split data into training and test data
    u_train, u_test = train_test_split(tweets_u, test_size=0.2, random_state=42, stratify=tweets_u["Name"])
    o_train, o_test = train_test_split(tweets_o, test_size=0.2, random_state=42, stratify=tweets_o["Name"])
    u_train_rt, u_test_rt = train_test_split(tweets_u_rt, test_size=0.2, random_state=42, stratify=tweets_u_rt["Name"])
    o_train_rt, o_test_rt = train_test_split(tweets_o_rt, test_size=0.2, random_state=42, stratify=tweets_o_rt["Name"])
    org_train, org_test = train_test_split(tweets_rt, test_size=0.2, random_state=42, stratify=tweets_rt["Name"])
    no_rt_train, no_rt_test = train_test_split(tweets, test_size=0.2, random_state=42, stratify=tweets["Name"])

    #Setup datasets
    dfs_train = [u_train_rt, o_train_rt, u_train, o_train, org_train, no_rt_train]
    dfs_test = [u_test_rt, o_test_rt, u_test, o_test, org_test, no_rt_test]
    datasetnames = ["undersampling with rt", "oversampling with rt","undersampling", "oversampling", "original", "no retweets"]

    #Setup multiprocess
    nb_workers = 4
    print(f"-------------Starting evaluation process using {nb_workers} workers")
    pool = Pool(processes=nb_workers)
    res = [[] for i in dfs_train]

    #Loop through all datasets
    for i, tr in enumerate(dfs_train):
        #Define preprocess
        cv = CountVectorizer(stop_words = "english")
        tfid = TfidfVectorizer(stop_words = "english")

        #Define models
        lr = LogisticRegression()
        rf = RandomForestClassifier()

        #Define pipelines
        p1 = Pipeline([('cv', cv), ('lr', lr)])
        p2 = Pipeline([('cv', cv), ('rf', rf)])
        p3 = Pipeline([('tfid', tfid), ('lr', lr)])
        p4 = Pipeline([('tfid', tfid), ('rf', rf)])
        pipes = [(p1,tr), (p2,tr), (p3,tr), (p4,tr)]


        #Fit pipe to dataset
        print(f"-------------Training models with dataset {datasetnames[i]}")
        models = pool.imap_unordered(fit_pipe_ds ,pipes)

        #Predict on test data
        te = dfs_test[i]
        model_te = [(model, te) for model in tqdm(models, total=len(pipes))]
        print(f"-------------Testing models with dataset {datasetnames[i]}")
        res1 = pool.imap_unordered(pred_model_ds, model_te)

        #Save res
        for t in tqdm(res1, total=len(pipes)):
            res[i].append(t)
    
    #Print result
    for i, accs in enumerate(res):
        print(f"---Mean accuracy with dataset {datasetnames[i]}: {np.mean(accs)}")

if __name__ == '__main__':
    main()