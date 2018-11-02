from __future__ import print_function

import pandas as pd
import json
import numpy as np
import datetime
import scipy.sparse as sp
from itertools import permutations
from sklearn import preprocessing



import time

import random

timeFormat = "%Y-%m-%dT%H:%M:%S.%fZ"

# To identify necessary tweet sequences that have atleast 10 reshares 
# Building target labels
def initial_sorting(file_name):
    
    tweet_final_count = {}
    
    tweet_group = {}
    
    with open(file_name) as json_data:

        for line in json_data:
            
            tweet = json.loads(line)
            
            if tweet['verb'] == "share":
                
                # To compute final reshare count i.e. target label
                key = int(tweet['object']['id'][28:])
                
                if key in tweet_final_count:
                    tweet_final_count[key] += 1
                else:
                    tweet_final_count[key] = 1  
                    
                original_posted_time = tweet['object']['postedTime']
                original_posted_time = datetime.datetime.strptime(original_posted_time, timeFormat)
                
                shared_time = tweet['postedTime']
                shared_time = datetime.datetime.strptime(shared_time, timeFormat)
                
                time_difference = shared_time - original_posted_time
                
                time_difference = round(time_difference.total_seconds()/60)
                
                if time_difference <= 20:
                    
                    if key in tweet_group:
                        
                        #Feature 1 - Update
                        temp_mentions = 0
                        for items in tweet['twitter_entities']['user_mentions']:
                            temp_mentions += 1
                        tweet_group[key]['mentions_count'] += temp_mentions
                        
                        # Feature 3 - Update
                        temp_hash = 0
                        for items in tweet['twitter_entities']['hashtags']:
                            temp_hash += 1
                        tweet_group[key]['hashtag_count'] += temp_hash    
                        
                        # Feature 5,6 - Update
                        if time_difference <= 10:
                            tweet_group[key]['first_10'] += 1
                        else:
                            tweet_group[key]['second_10'] += 1      
                            
                        
                        
                    else:
                        
                        tweet_group[key] = {}

                        
                        # Feature 1 - mentions count
                        mentions = set()
                        for items in tweet['object']['twitter_entities']['user_mentions']:
                            mentions.add(items['id'])
                        for items in tweet['twitter_entities']['user_mentions']:
                            mentions.add(items['id'])
                        tweet_group[key]['mentions_count'] = len(mentions)
                        
                        # Feature 2 - Counting number of words present in the tweet body
                        tweet_group[key]['word_count'] = len(tweet['object']['body'].split())
                    
                        # Feature 3 - Counting number of hashtags available
                        hashtags = set()
                        for items in tweet['object']['twitter_entities']['hashtags']:
                            hashtags.add(items['text'])
                        for items in tweet['twitter_entities']['hashtags']:
                            hashtags.add(items['text'])
                        tweet_group[key]['hashtag_count'] = len(hashtags)
                        
                        # Feature 4 - Media Count
                        if 'media' in tweet['object']['twitter_entities']:
                            tweet_group[key]['media_count'] = len(tweet['object']['twitter_entities']['media'])
                        else:
                            tweet_group[key]['media_count'] = 0
                    

                        # Feature 5 - Number of tweets in the first ten minutes                   
                        # Feature 6 - Number of tweets in the second ten minutes
                        if time_difference <= 10:
                            tweet_group[key]['first_10'] = 1
                        else:
                            tweet_group[key]['first_10'] = 0
                            
                        if time_difference > 10:
                            tweet_group[key]['second_10'] = 1
                        else:
                            tweet_group[key]['second_10'] = 0 
                    

    # Removing tweet sequences that have retweet count lesser than ten                
    for key, value in tweet_final_count.items():
        if value < 10:
            del tweet_final_count[key]
            
    return tweet_final_count,tweet_group                  
                  

# Build generic dataframe common to all actors involved in the dataset
# Compututation at node level
def build_common_features(file_name, needed_tweets):
    
    general = {}
    share_count = {}
    reshare_count = {}
        
    post_count = 0
    total = 0
    share = 0
    
    with open(file_name) as json_data:
        # To get file name
        file_name = file_name.split('-')[0]
        
        for line in json_data:
            
            tweet = json.loads(line)
            
            # To track the total number of tweets in the dataset
            total += 1
            
            # To track the number of original posts present in the dataset
            if tweet['verb'] == 'post':
                post_count += 1
            
            # Checking if the particular tweet is a retweet or new post
            if tweet['verb'] == "share":

                # To track the number of retweet sequences in the dataset
                share += 1
                
                key = int(tweet['object']['id'][28:])
                
                author = int(tweet['object']['actor']['id'][15:])
                shared = int(tweet['actor']['id'][15:])
                
                # Checking if the particular tweet is a needy tweet(tweet sequence that have atleast 10 reshares) or not
                if key in needed_tweets:                    
                    
                    # Computing features at node level
                    if author not in general:
                        general[author] = {}
                        
                        # Feature 1 - Followers Count
                        general[author]['followers_count'] = tweet['object']['actor']['followersCount']
                        
                        # Feature 2 - is Verifed
                        if tweet['object']['actor']['verified'] == False:
                            general[author]['is_verified'] = 0
                        else:
                            general[author]['is_verified'] = 1
                            
                        
                        # Feature 3 - Friends Count
                        general[author]['friends_count'] = tweet['object']['actor']['friendsCount']
                        
                        # Feature 4 - Total Number of tweets
                        general[author]['total_tweets'] = tweet['object']['actor']['statusesCount']
                        
                    if shared not in general:
                        general[shared] = {}
                        
                        # Feature 1 - Followers Count
                        general[shared]['followers_count'] = tweet['actor']['followersCount']
                        
                        # Feature 2 - is Verifed
                        if tweet['actor']['verified'] == False:
                            general[shared]['is_verified'] = 0
                        else:
                            general[shared]['is_verified'] = 1
                        
                        # Feature 3 - Friends Count
                        general[shared]['friends_count'] = tweet['actor']['friendsCount']
                        
                        # Feature 4 - Total Number of tweets
                        general[shared]['total_tweets'] = tweet['actor']['statusesCount']
                
                # To count the total number of tweets that actor has made in the dataset        
                if author in share_count:
                    share_count[author] += 1
                else:
                    share_count[author] = 1
                
                # To count the total number of reshares that actor has made in the dataset     
                if shared in reshare_count:
                    reshare_count[shared] += 1
                else:
                    reshare_count[shared] = 1
    
    # Converting the dictionary into pandas dataframe                        
    dataframe = pd.DataFrame.from_dict(general, orient='index')
    
    # As a requirement for graph convolution network
    dataframe['actor_id'] = dataframe.index
    dataframe['actor_id'] = dataframe['actor_id'].astype(int)
    
    # Adding two new features
    dataframe['tweets_in_ds'] = dataframe['actor_id'].map(share_count) 
    dataframe['tweets_in_ds'].fillna(0, inplace=True)
    
    dataframe['reshares_in_ds'] = dataframe['actor_id'].map(reshare_count) 
    dataframe['reshares_in_ds'].fillna(0, inplace=True)
    
    dataframe = dataframe[['actor_id','followers_count','is_verified','friends_count','total_tweets','tweets_in_ds','reshares_in_ds']]
    
    actors_alone = pd.DataFrame({'actor_id': dataframe['actor_id'].tolist() }, index = dataframe.index)
    
    f = dataframe.loc[:, dataframe.columns != 'actor_id'].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(f)
    new_dataframe = pd.DataFrame(x_scaled, index = dataframe.index)
    
    resuult = pd.concat([actors_alone, new_dataframe], axis=1)
    
    resuult.columns = [['actor_id','followers_count','is_verified','friends_count','total_tweets','tweets_in_ds','reshares_in_ds']]
    
            
    print("Total Number of tweets in the dataframe",total)
    
    print("Total Number of Original tweets is", post_count)
    
    print("Total Number of reshare tweets is",share)
    
    # dataframe contains node level features corresponding to all the actors involved in the tweet sequences
    return dataframe,resuult
    
def new_adjacency_matrix(file_name,actors,actors_mapping):
    
    adj = np.zeros((len(actors),len(actors)))
    
    contributors = {}
    
    with open(file_name) as json_data:
        
        for line in json_data:
            
            tweet = json.loads(line)
                        
            # Checking if the particular tweet is a retweet or new post
            if tweet['verb'] == "share":
                
                key = int(tweet['object']['id'][28:])
                
                author = int(tweet['actor']['id'][15:])
                shared = int(tweet['object']['actor']['id'][15:])
                
                if key not in contributors:
                    contributors[key] = set()
                    if author in actors:
                        contributors[key].add(actors_mapping[author])
                    if shared in actors:
                        contributors[key].add(actors_mapping[shared])
                else:
                    if author in actors:
                        contributors[key].add(actors_mapping[author])
                    if shared in actors:
                        contributors[key].add(actors_mapping[shared])
                        
    for key,value in contributors.iteritems():
        perm = permutations(value, 2)
        perm = list(perm)
        for items in perm:
            adj[items[0]][items[1]] += 1
            
    return adj
                    

# To form adjacency matrix 
# Extract all the mentions available in the tweet sequences    
def build_connections_list(file_name, needed_tweets):
    
    mentions_list = []
        
    with open(file_name) as json_data:
        
        for line in json_data:
            
            tweet = json.loads(line)
            
            # Checking if the particular tweet is a retweet or new post
            if tweet['verb'] == "share":
                
                key = int(tweet['object']['id'][28:])
                
                if key in needed_tweets:
                    
                    # Extracts the pair of mentions present in the original tweet                        
                    for items in tweet['twitter_entities']['user_mentions']:
                        
                        mentions_list.append([items['id'],tweet['actor']['id'][15:]])
                        
                    # Extracts the pair of mentions present in the retweet sequence     
                    for items in tweet['object']['twitter_entities']['user_mentions']:
                        mentions_list.append([items['id'],tweet['object']['actor']['id'][15:]])

    # To remove redundant pair of mentions
    b_set = set(tuple(x) for x in mentions_list)
    b = [ list(x) for x in b_set ]
                
    edges_unordered=np.array([np.array(xi) for xi in b])
    
    edges_unordered = edges_unordered.tolist()
    
    edges_unordered = [map(int,x) for x in edges_unordered]
    
    # edges_unordered contains the pair of non-redundant mentions  
    return edges_unordered

# To find the people contributed to that tweet in the first ten minutes    
# To find the people contributed to that tweet in the first ten minutes
def find_contributors(file_name, needed_tweets):
    
    contributors  = {}
    
    mention_contributors = {}
    active_contributors = {}
    
    i = 1
    
    
    with open(file_name) as json_data:
        
        for line in json_data:
            
            tweet = json.loads(line)
            
            # Checking if the particular tweet is a retweet or new post
            if tweet['verb'] == "share":
                
                key = int(tweet['object']['id'][28:])
                
                if key in needed_tweets:
                    
                    # To find the people who have contributed in the first ten minutes
                    original_posted_time = tweet['object']['postedTime']
                    original_posted_time = datetime.datetime.strptime(original_posted_time, timeFormat)
                    
                    shared_time = tweet['postedTime']
                    shared_time = datetime.datetime.strptime(shared_time, timeFormat)
                    
                    time_difference = shared_time - original_posted_time
                    
                    time_difference = round(time_difference.total_seconds()/60)
                    
                    
                    if time_difference <= 20:
                        # Keeps track of all people who have contributed to that particular tweet
                        if key not in contributors:
                            contributors[key] = set()
                            # Keeps track of people who have tweeted,retweeted that particular tweet
                            active_contributors[key] = set()
                            # Keeps track of people who have contributed to tweet via mentions
                            mention_contributors[key] = set()
                    
                        contributors[key].add(int(tweet['actor']['id'][15:]))
                        contributors[key].add(int(tweet['object']['actor']['id'][15:]))
                        
                        active_contributors[key].add(int(tweet['actor']['id'][15:]))
                        active_contributors[key].add(int(tweet['object']['actor']['id'][15:]))
                        
                        for items in tweet['twitter_entities']['user_mentions']:
                            
                            contributors[key].add(int(items['id']))
                            contributors[key].add(int(tweet['actor']['id'][15:]))
                            
                            mention_contributors[key].add(int(items['id']))
            
            
                        for items in tweet['object']['twitter_entities']['user_mentions']:
                            contributors[key].add(int(items['id']))
                            contributors[key].add(int(tweet['object']['actor']['id'][15:]))
                            
                            mention_contributors[key].add(int(items['id']))

    return contributors,mention_contributors,active_contributors

# Find the valid connections present in the found pair of mentions
# Valid connections refer to the pair that have features for both of them   
def find_present_connections(edges_unordered, actors, actors_mapping):
    output = []
    count = 0

    for items in edges_unordered:

        if items[0] in actors and items[1] in actors:
            count+= 1
            output.append([actors_mapping[items[0]],actors_mapping[items[1]]])

    return count,output
    
def normalize_adj(adj):    
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
    a_norm = adj.dot(d).transpose().dot(d).tocsr()
    return a_norm
    
def preprocess_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)
    return adj

