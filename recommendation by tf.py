import os
os.listdir('../input')
import numpy as np
import pandas as pd
import tensorflow as tf
rating=pd.read_csv('../input/ratings.csv')
links=pd.read_csv('../input/links.csv')
tags=pd.read_csv('../input/tags.csv')
movies=pd.read_csv('../input/movies.csv')
movies['movie_index']=movies.index
movies.tail()
new_movies=movies[['movie_index','movieId','title']]
new_ratings=pd.merge(rating,new_movies,on='movieId')
new_ratings.head()
ratings =new_ratings[['userId','movie_index','rating']]
userNum=ratings['userId'].max()+1
movieNum=ratings['movie_index'].max()+1
ratingArr=np.zeros((movieNum,userNum))
count=0
for index,row in ratings.iterrows():
    ratingArr[int(row['movie_index']),int(row['userId'])]=row['rating']
    count+=1
validRecord=np.array(ratingArr>0,dtype=int)
validRecord
def normalizeRatings(ratings,validRecord):
    m,n=ratings.shape
    ratings_mean=np.zeros((m,1))
    ratings_norm=np.zeros((m,n))
    for i in range(m):
        no_zero_index= validRecord[i,:]!=0
        ratings_mean[i]=np.mean(ratings[i,no_zero_index])
        ratings_norm[i,no_zero_index]-=ratings_mean[i]
    return ratings_norm,ratings_mean
ratings_norm,ratings_mean=normalizeRatings(ratings,validRecord)
ix=validRecord[2,:]!=0
list(np.where(ix==1)[0])
rating_norm=np.nan_to_num(rating_norm)
rating_mean=np.nan_to_num(rating_mean)
num_features=10
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
x_parameters=tf.Variable(tf.random_normal([movieNum,num_features],stddev=0.35))
theta_parameters=tf.Variable(tf.random_normal([userNum,num_features],stddev=0.35))
loss=1/2*tf.reduce_sum(((tf.matmul(x_parameters,theta_parameters,transpose_b=True)-rating_norm)*validRecord)**2)+1/2*(tf.reduce_sum(x_parameters**2)+tf.reduce_sum(theta_parameters**2))
optimizer=tf.train.GradientDescentOptimizer(0.0001)
train_loss=optimizer.minimize(loss)
for i in range(5000):
    sess.run(train_loss)
cx_parameters,ctheta_parameters=sess.run([x_parameters,theta_parameters])
predicted=np.dot(x_parameters,theta_parameters.T)+rating_mean
error=np.sqrt(np.sum((predicted-ratingArr)**2))

error
