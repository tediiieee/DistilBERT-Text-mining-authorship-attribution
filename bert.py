# Run gridsearch cv on all ml models (BERT variants)
import numpy as np
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import time as time
from tqdm import tqdm
#%tensorflow_version 1.x
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
from sklearn.pipeline import Pipeline

from dataset import *
from ml import *



############## MAIN ################
def bert(tweets, BATCHSIZE = 2000):
    """Extract features from tweets using DistilBERT

    Args:
        tweets (pd.Dataframe): Dataframe containing text and labels.
        BATCHSIZE (int, optional): Approximate batch size that will be used.
            (Not exact since the actual batch size is optimized) Defaults to 2000.

    Returns:
        [np.ndarray]: transformed features
    """

    #Checking if GPU is available.
    #Requires CUDA, Cudnn, pytorch with gpu
    if torch.cuda.is_available():
        #GPU setup
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        

        # Load in DistilBERT:
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        
        #Ordinary BERT
        #model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        
        # Load pretrained model/tokenizer, load model to GPU VRAM
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights).to(device) 

        #Tokenize
        tokenized = tweets["Tweet description"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        
        #Padding
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

        #Masking to tell bert not to care about the padding
        attention_mask = np.where(padded != 0, 1, 0)

        all_features = []
        # Setup batches
        max_ind = padded.shape[0]
        batch_size_opti = optimize_batchsize(max_ind, BATCHSIZE)
        max_batch = int(np.ceil(max_ind /batch_size_opti))

        #Run the model 
        for i in tqdm(range(max_batch), desc = "Batch transformation"):
            # Take batch_size_opti data
            padded_batch = padded[i*batch_size_opti:(i+1)*batch_size_opti, :]
            attention_mask_batch = attention_mask[i*batch_size_opti:(i+1)*batch_size_opti, :]

            # Define tensors to transform, load data to GPU VRAM
            input_ids = torch.tensor(padded_batch).to(device)
            attention_mask_batch = torch.cuda.ByteTensor(attention_mask_batch).to(device)

            # Run bert model
            with torch.no_grad():
                last_hidden_states = model(input_ids, attention_mask=attention_mask_batch)

            #Extract features
            features_batch = last_hidden_states[0][:,0,:].cpu().numpy()
            all_features.append(features_batch)
            torch.cuda.empty_cache()

        # Stack data from all batches
        features = np.vstack(all_features)
        return features
    else:
        print("GPU with pytorch not setup correctly.")

def main():
    #Load data
    path = "tweet.csv"
    tweets = load_data(path)

    #Remove rem_retweets
    tweets = rem_retweets(tweets)

    #Under/oversample to create datasets
    tweets_o = even_dist(tweets, "oversample")[0]

    # Run BERT
    features = bert(tweets_o)

    # Get labels
    labels = tweets_o["Name"]

    #test/train split
    tr, te, tr_lab, te_lab = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    #Setup multiprocess
    nb_workers = 2 
    pool = Pool(processes=nb_workers)

    #Define models, parameters
    lr = LogisticRegression()
    rf = RandomForestClassifier()
    p1 = Pipeline([('lr', lr)])
    p2 = Pipeline([('rf', rf)])
    par_lr = {'lr__C':[1000, 100, 10, 1]}
    par_rf = {'rf__n_estimators':[100,150,200]}
    pipes = [(p1,par_lr,tr,tr_lab), (p2,par_rf,tr,tr_lab)]
    
    #Gridsaerch cross-validation to tune hyperparameters
    print(f"-------------Training models with oversampled dataset")
    models = pool.imap_unordered(fit_pipe_ms, pipes)

    #Predict on test data
    model_te = [(model, te, te_lab) for model in tqdm(models, total=len(pipes))]
    print(f"-------------Testing models with oversampled dataset")
    res1 = pool.imap_unordered(pred_model_ms, model_te)
    res1 = [1 for i in res1]


def optimize_batchsize(n, batch_size):
    #Optimizes batch size so it's much more even. 
    #Thus, avoiding batches with very little data
    return int(np.ceil(n / np.ceil(n / batch_size)))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))    