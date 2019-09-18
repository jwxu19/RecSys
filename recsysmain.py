import pickle
from surprise import Reader
from surprise import model_selection
from surprise import accuracy
from surprise import Dataset

from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering

import numpy as np
import random
from collections import *
import time
import re
from sklearn.metrics.pairwise import cosine_similarity


def load_fitered_data():
  file=open("data_after_filter","rb")
  data=pickle.load(file)
  file.close()
  data=data[['reviewerID', 'asin', 'overall']]
  return data

def format_data(data):
  reader= Reader(rating_scale=(1, 5))
  data = Dataset.load_from_df(data, reader)
#  trainset, testset = train_test_split(data, test_size=0.20)
  return data

def precision_recall_at_k(predictions, k, threshold):
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    
    # Precision and recall can then be averaged over all users
    overall_precisions=sum(prec for prec in precisions.values())/len(precisions)
    overall_recalls=sum(rec for rec in recalls.values()) / len(recalls)

    return overall_precisions, overall_recalls

#def iterate_k(precisions_dict, recalls_dict, predictions, k_ls, threshold):
#  
#  for k in k_ls:
#    precisions, recalls=precision_recall_at_k(predictions, k, threshold)
#    precisions_dict[k]=[precisions_dict[k], precisions]
#    recalls_dict[k]=[recalls[k], recalls]

#  return precisions, recalls

def get_top_n(predictions, n):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def personalization(prediction, n):
  #prediction
  #n top n recommendation
  
  top_n = get_top_n(prediction, n)
  
  rec_dict={}
  for uid, user_ratings in top_n.items():
    rec_dict[uid]=[iid for (iid, _) in user_ratings]
  
  
  rec_user_ls=[pred[0] for pred in prediction]
  rec_item_ls=[pred[1] for pred in prediction]

  unique_rec_user_ls=np.unique(rec_user_ls)
  unique_rec_item_ls=np.unique(rec_item_ls)

  #assign each item with index number 
  unique_rec_item_dict={item:ind for ind, item in enumerate(unique_rec_item_ls)}

  n_unique_rec_user=len(unique_rec_user_ls)
  n_unique_rec_item=len(unique_rec_item_ls)
  
  #recommended user item matrix 
  rec_matrix=np.zeros(shape=(n_unique_rec_user, n_unique_rec_item))


  #represent recommended item for each user as binary 0/1
  for user in range(n_unique_rec_user):
    #get userid
    user_id=unique_rec_user_ls[user]
    #get rec item list
    item_ls=rec_dict[user_id]
   
    for item_id in item_ls:
      #get item index
      item=unique_rec_item_dict[item_id]
      rec_matrix[user, item]=1
  
  #calculate cosine similarity matrix across all user recommendations  
  similarity = cosine_similarity(X=rec_matrix, dense_output=False)
  #calculate average of upper triangle of cosine matrix
  upper_right = np.triu_indices(similarity.shape[0], k=1)
  #personalization is 1-average cosine similarity 
  personalization = 1-np.mean(similarity[upper_right])

def iterate_algo(algo_ls, kfold, data, top_n, threshold, k_ls):
  
  kf=model_selection.KFold(n_splits=kfold)
  
  cv_rmse=[]
  cv_precision=[]
  cv_recall=[]
  cv_fit_time=[]
  cv_pred_time=[]
  cv_personalization=[]
  
  algo_name=[]


  for algo in algo_ls:
    rmse_ls=[]
    precisions_dict={}
    recalls_dict={}
    fit_time_ls=[]
    pred_time_ls=[]
    personalization_ls=[]
    
    #perform cross vailidation 
    for train, test in kf.split(data):

    
      fit_start = time.time()
      algo.fit(train)
      fit_time =time.time()-fit_start
    
      pred_start =time.time()
      pred=algo.test(test)
      pred_time=time.time()-pred_start
      
    
      rmse=accuracy.rmse(pred)
      rmse_ls.append(rmse)
      
    
      #iterate k_ls
      for k in k_ls:

        precisions, recalls=precision_recall_at_k(pred, k, threshold)

        if k in precisions_dict:
          precisions_dict[k].append(precisions)
        else:
          precisions_dict[k]=[precisions]
        
        if k in recalls_dict:
          recalls_dict[k].append(recalls)
        else:
          recalls_dict[k]=[recalls]


      
      
      personalization_ls.append(personalization(pred, top_n))
      
    
      fit_time_ls.append(fit_time)
      pred_time_ls.append(pred_time)

      
    #cv_rmse.append(np.mean(rmse_ls))
    
    #cv_precisions={k: np.mean(v) for k, v in precisions_dict.items()}
    #cv_recalls={k: np.mean(v) for k, v in recalls_dict.items()}

    #cv_personalization.append(np.mean(personalization_ls))
    #cv_fit_time.append(np.mean(fit_time_ls))
    #cv_pred_time.append(np.mean(pred_time_ls))
    
    cv_rmse.append(rmse_ls)
    
    cv_precision.append(precisions_dict)
    cv_recall.append(recalls_dict)

    cv_personalization.append(personalization_ls)
    cv_fit_time.append(fit_time_ls)
    cv_pred_time.append(pred_time_ls)
    
    
    regex='(\w+)\s'
    name=re.search(regex,str(algo))
    algo_name.append(name.group())
    
  return cv_rmse, cv_personalization, cv_precision, cv_recall, cv_fit_time, cv_pred_time, algo_name

def main():
  
  seed=0
  random.seed(seed)
  np.random.seed(seed)

  data = load_fitered_data()
  data = format_data(data)
  
  
  kfold=5
  algo_ls=(KNNWithMeans(), SVD())
  #algo_ls=(KNNWithMeans(), KNNBasic(), KNNWithZScore(), SVD(), SVDpp(), NMF(), SlopeOne(), CoClustering())
  top_n=10
  threshold=4
  k_ls=[3,5,7,10]
  cv_rmse, cv_personalization, cv_precision, cv_recall, cv_fit_time, cv_pred_time, algo_name = iterate_algo(algo_ls, kfold, data, top_n, threshold, k_ls)
  ls_name=["cv_rmse", "cv_personalization", "cv_precision", "cv_recall", "cv_fit_time", "cv_pred_time", "algo_name"]
  

  output=ls_name, cv_rmse, cv_personalization, cv_precision, cv_recall, cv_fit_time, cv_pred_time, algo_name
  

  
  with open("metrics.pkl","wb") as f:
    pickle.dump(output, f)
  
  return cv_rmse, cv_personalization, cv_precision, cv_recall, cv_fit_time, cv_pred_time, algo_name

if __name__ == "__main__":
    main()

