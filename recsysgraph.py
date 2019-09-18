import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics():
  file=open("metrics.pkl","rb")
  data=pickle.load(file)
  file.close()
  return data

def make_df(data):
  ls_name, cv_rmse, cv_precision, cv_recall, cv_fit_time, cv_pred_time, algo_name=data
  
  precision=[]
  for i in cv_precision:
    precision.append({k: np.mean(v) for k, v in i.items()})


  recall=[]
  for i in cv_recall:
    recall.append({k: np.mean(v) for k, v in i.items()})


  df_precision=pd.DataFrame(precision, index=algo_name).T
  df_recall=pd.DataFrame(recall, index=algo_name).T
  
  rmse=[np.mean(i) for i in cv_rmse]
  fit_time=[np.mean(i) for i in cv_fit_time]
  pred_time=[np.mean(i) for i in cv_pred_time]

  df=pd.DataFrame([rmse,fit_time,pred_time], index=["rmse", "fit_time", "test_time"], columns=algo_name)
  
  return df_precision, df_recall, df

def plot_precsion_recall_k(df_precision, df_recall):
  

  ax1=df_precision.plot.line(title="Precision Comparsion of Algo at different k")
  ax1.set_xlabel("k")
  ax1.set_ylabel("Precision")

  ax2=df_recall.plot.line(title="Recall Comparsion of Algo at different k")
  ax2.set_xlabel("k")
  ax2.set_ylabel("Recall")
 
  
  return ax1, ax2

def show_table(df_precision, df_recall, df):
  
  print("Precision")
  print(df_precision)
  print("Recall")
  print(df_recall)
  print("Rmse, Fit Time, Test Time")
  print(df)
  



  
#ax1=df.loc["rmse"].plot.barh(title="RMSE Comparsion")
#ax1.set_ylabel("rmse")

#ax2=df.loc["fit_time"].plot.barh(title="Fit Time Comparsion")
#ax2.set_ylabel("fit time")

#ax3=df.loc["test_time"].plot.barh(title="Test Time Comparsion")
#ax3.set_ylabel("test time")

def main():
  data=load_metrics()
  df_precision, df_recall, df=make_df(data)
  plot_precsion_recall_k(df_precision, df_recall)
  show_table(df_precision, df_recall, df)

if __name__ == "__main__":
    main()
