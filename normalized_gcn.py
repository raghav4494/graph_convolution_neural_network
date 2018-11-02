# -*- coding: utf-8 -*-
from preprocessing import *
import scipy.sparse as sp
import pandas
import numpy as np
import tensorflow as tf


# Input file reading

file_name = "Nigeria-community-relevant-restricted.json"


# Filtering out the required and non-required tweet sequences

needed_tweets,tweet_group = initial_sorting(file_name)


# To find the participants (people who are connected to each other) for each tweet sequence

edges_unordered = build_connections_list(file_name, needed_tweets)


# Finding feature matrix for every tweet sequence

old,dataframe_common = build_common_features(file_name, needed_tweets)


contributors,mention_contributors,active_contributors = find_contributors(file_name, needed_tweets)


# Finding actors (tweet authors) for each sequence

l = dataframe_common['actor_id'].values.tolist()
actors = [item for sublist in l for item in sublist]

actors_mapping = {j: i for i, j in enumerate(actors)}


# To find Adjacency matrix between actors
# The values in the matrix denote the number of times the people have retweeted together
# For example, The value 4 for X and Y denotes, both of them have retweeted the same four tweets

A = new_adjacency_matrix(file_name,actors,actors_mapping)


features = np.empty((len(needed_tweets),dataframe_common.shape[0],dataframe_common.shape[1]+2))


count = 0

index_mapping = {}

for items in needed_tweets:
    index_mapping[count] = items
    count += 1


# To find from the list of all valid actors, listing people whoever is related to the particular tweet sequence.

for i in range(len(needed_tweets)):
  
    new_dataframe = dataframe_common.copy()
    new_dataframe['participation'] = 0
    new_dataframe['mention_participation'] = 0
    new_dataframe['active_participation'] = 0
    
    # Creating additional columns            
    if index_mapping[i] in contributors:
        needs = contributors[index_mapping[i]]
    
        
        for items in needs:
            if items in new_dataframe.index:
                new_dataframe.at[items, 'participation'] = 1
                                
        needs = active_contributors[index_mapping[i]]
    
        for items in needs:
            if items in new_dataframe.index:
                new_dataframe.at[items, 'active_participation'] = 1
        
        needs = mention_contributors[index_mapping[i]]
    
        for items in needs:
            if items in new_dataframe.index:
                new_dataframe.at[items, 'mention_participation'] = 1
                
    features[i] = new_dataframe.iloc[:,1:].as_matrix()


target = np.array(needed_tweets.values())


# Tensorflow computations - weight, bias, graph convolution layer, fully connected layer

x = tf.placeholder(tf.float32, [features.shape[1],3])
y = tf.placeholder(tf.float32, [1])
a = tf.placeholder(tf.float32, [A.shape[0],A.shape[0]])
random_data1 = tf.placeholder(tf.float32, [6,1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def weight_constant(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.01, shape=shape)
    return tf.Variable(initial)

def cons_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)

def fc_layer1(prev, output_size, input_size):
    W = weight_constant([output_size, input_size])
    b = bias_variable([output_size])
    return tf.matmul(W,prev) + b

def fc_layer2(prev,input_size):
    W = cons_variable([1, input_size])
    return tf.matmul(W,prev)

def fc_layer(prev, input_size, output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    c = tf.matmul(prev, W) + b
    return c

def gcn_layer(prev, input_size, output_size, A):
    W = weight_variable([input_size, output_size])
    b = bias_variable([1,output_size])
    prev1 = tf.matmul(prev,W)
    out1 = tf.matmul(A,prev1)+b
    return out1

# Neural Network Architecture

def neural_network(x,random_data):
    
    l1 = tf.nn.relu(gcn_layer(x,3,4,a))
    
    l3 = tf.nn.relu(gcn_layer(l1, 4, 1,a))
    
    l4 = tf.concat([l3,random_data],0)

    output = fc_layer1(l4,1,A.shape[0]+6)
    
    return output

output = neural_network(x,random_data1)

loss = tf.reduce_mean(tf.squared_difference(y, output))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)


# For Fully Connected Layer concatenations

def feature_generator(index):
    
    tweet_id = index_mapping[index]
    
    
    if tweet_id in tweet_group:
        mentions_count = tweet_group[tweet_id]['mentions_count']
        hashtag_count = tweet_group[tweet_id]['hashtag_count']
        first_ten = tweet_group[tweet_id]['first_10']
        second_ten = tweet_group[tweet_id]['second_10']
        word_count = tweet_group[tweet_id]['word_count']
        media_count = tweet_group[tweet_id]['media_count']   
        data = [[mentions_count],[hashtag_count],[first_ten],[second_ten],[word_count],[media_count]]
    else:
        data = [[0],[0],[0],[0],[0],[0]]
        
    return data
    
f = open('Nigeria_GCN_new.txt','w')

import math
top_70 = int(math.floor(0.7 * len(needed_tweets)))

iteration_count = 20


final_values = {}

# Tensorflow computations

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    for ii in range(iteration_count):
        
        
        for i in range(0,top_70):
            
            random_data = feature_generator(i)
            
            sess.run([loss,train_step],feed_dict={x : features[i][:,6:9],y : np.array([target[i]]),a : A, random_data1: random_data})
    
        predictions = []
        
        for j in range(top_70,len(needed_tweets)):  
            
            random_data = feature_generator(j)
                
            pred = sess.run(output, feed_dict={x:features[j][:,6:9],a : A, random_data1:random_data})
            predictions.append(pred)
        
        #predictions = predictions.tolist()[0]
        
        final_values[ii] = predictions
        
        print("Iteration is",ii)
        f.write("\n"+"Iteration is "+str(ii)+"\n")
        
        whole_error = 0
        absolute_error = 0
        
        temp = 0
        
        for v in range(top_70,len(needed_tweets)):
        #for v in range(500,600):
            whole_error += ((target[v] - predictions[temp])**2)
            absolute_error += np.absolute(target[v] - predictions[temp])
            temp += 1
            
        mse = whole_error/(len(needed_tweets)-top_70)
        #mse = whole_error/100
        print("Mean Squared Error")
        print(mse)
        f.write("Mean Squared Error is "+str(mse)+"\n")
        
        
        error = absolute_error/(len(needed_tweets)-top_70)
        #error = absolute_error/100
        print("Mean Absolute Error is")
        print(error)
        f.write("Mean Absolute Error is "+str(error)+"\n")
        
        

        
        f.write("Sample Solution"+"\n")
        for rand in range(0,5):
            f.write("%s " % predictions[rand])

f.close()


# For different metrics evaluation

flattened = np.concatenate(final_values[iteration_count]).ravel().tolist()

comp = pd.DataFrame({'original':target[top_70:len(needed_tweets)],'predicted':flattened})

comp.corr(method='pearson')

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

r2_score(comp['original'].tolist(),comp['predicted'].tolist())

comp.to_csv("GCN_comp.csv")

mean_absolute_error(comp['original'].tolist(),comp['predicted'].tolist())

