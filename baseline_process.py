# Run gridsearch cv on all ml models (excluding BERT variants)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from dataset import *
from ml import *

def main():
    print("-------------Program started")
    print("-------------Loading data")
    #Load data
    path = "tweet.csv"
    tweets = load_data(path)

    #Under/oversample to create datasets
    tweets_o = even_dist(tweets, "oversample")[0]

    #Split data into training and test data
    tr, te, tr_lab, te_lab = train_test_split(tweets_o["Tweet description"], tweets_o["Name"] , test_size=0.2, random_state=42, stratify=tweets_o["Name"])

    #Setup multiprocess
    nb_workers = 6 # Lower nb workers if crashing due to too few cores in CPU.
    print(f"-------------Starting evaluation process using {nb_workers} workers")
    pool = Pool(processes=nb_workers)

    #Define preprocess
    cv = CountVectorizer(stop_words = "english")
    tfid = TfidfVectorizer(stop_words = "english")

    #Define models
    lr = LogisticRegression()
    rf = RandomForestClassifier()

    #Define pipelines and model parameters
    p1 = Pipeline([('cv', cv), ('lr', lr)])
    p2 = Pipeline([('cv', cv), ('rf', rf)])
    p3 = Pipeline([('tfid', tfid), ('lr', lr)])
    p4 = Pipeline([('tfid', tfid), ('rf', rf)])
    #Hyperparameter search space
    par_lr = {'lr__C':[1000, 100, 10, 1]}
    par_rf = {'rf__n_estimators':[100,150,200]}
    pipes = [(p1,par_lr,tr,tr_lab), (p2,par_rf,tr,tr_lab), (p3,par_lr,tr,tr_lab), (p4,par_rf,tr,tr_lab)]


    #Gridsaerch cross-validation to tune hyperparameters
    print(f"-------------Training models with oversampled dataset")
    models = pool.imap_unordered(fit_pipe_ms, pipes)

    #Predict on test data
    model_te = [(model, te, te_lab) for model in tqdm(models, total=len(pipes))]
    print(f"-------------Testing models with oversampled dataset")
    res1 = pool.imap_unordered(pred_model_ms, model_te)
    res1 = [1 for i in res1]

if __name__ == '__main__':
    main()