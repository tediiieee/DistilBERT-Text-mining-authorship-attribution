# Contains useful functions relating to the datasets.
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def load_data(path):
    tweets = pd.read_csv(path)
    return tweets

def count_unique(df):
    return df["Name"].nunique()

def value_counts(df):
    return df["Name"].value_counts()

def author_dist(df):
    return df["Name"].value_counts(normalize = True)

def rem_retweets(df):
    df_no_rt = df[~df["Tweet description"].str.contains("RT ")]
    return df_no_rt

def even_dist(df, method = "undersample"):
    """
    Uses undersampling or oversampling to make the class distributions even.

    Args:
        df (pd.Dataframe): Dataframe with data from the twitter dataset.
        method (str, optional): Choice of sampling method (oversample/undersample). 
        Defaults to "undersample".

    Returns:
        [pd.Dataframe]: Sampled dataframe.
    """
    if method == "undersample":
        sampler = RandomUnderSampler(random_state=42)
    if method == "oversample":
        sampler = RandomOverSampler(random_state=42)
    df = sampler.fit_resample(df, df["Name"])
    return df