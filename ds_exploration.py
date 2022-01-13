# Used to print the shapes of each dataset and plot the class distributions.

import matplotlib.pyplot as plt
from dataset import *

############## MAIN ################

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

    #Print shape for each dataset
    print("nr samples in 'original' dataset: ", tweets_rt.shape)
    print("nr samples in 'no retweets' dataset: ", tweets.shape)
    print("nr samples in 'undersample' dataset: ", tweets_u_rt.shape)
    print("nr samples in 'no retweets, undersample' dataset: ", tweets_u.shape)
    print("nr samples in 'oversample' dataset: ", tweets_o_rt.shape)
    print("nr samples in 'no retweets, oversample' dataset: ", tweets_o.shape)

    #Plot the class distributions
    #With retweets
    tweets_rt["Name"].value_counts().plot(kind='barh')
    plt.title("Class distribution 'original' dataset")
    plt.xlabel("Samples")
    plt.show()
    plt.show()

    #Without retweets
    tweets["Name"].value_counts().plot(kind='barh')
    plt.title("Class distribution 'no retweets' dataset")
    plt.xlabel("Samples")
    plt.show()
    plt.show()
   

if __name__ == '__main__':
    main()