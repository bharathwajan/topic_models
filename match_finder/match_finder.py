import pandas as pd
import pickle

#currently this works for only News20 #without order

def accuracy(plsa_path,baysian_path,no_topics,n_largest,match_threshold):
    bay=pd.read_csv(baysian_path)
    plsa=pd.read_csv(plsa_path)
    print("read sucessfully")

    bay.drop(["Unnamed: 0"],axis=1,inplace=True) 
    plsa.drop(["Unnamed: 0"],axis=1,inplace=True)

    with open(f"../nb_outputs/news_20/missing_values_{no_topics}.pkl","rb") as f:
        missed_idx=pickle.load(f) #got this from nb_gnn 

    news_20=pd.read_csv("../datasets/20news.csv")
    original_idx=list(range(news_20.shape[0]))
    
    for item in missed_idx:
        original_idx.remove(item)
    
    
    
    bay["index"]=original_idx

    bay_idx,plsa_idx=set(bay["index"]),set(plsa["index"])
    
    idx_to_keep=bay_idx.intersection(plsa_idx)

    cleaned_bay=bay[bay["index"].isin(idx_to_keep)]
    cleaned_plsa_5=plsa[plsa["index"].isin(idx_to_keep)]

    cleaned_bay.index=cleaned_bay["index"]
    cleaned_plsa_5.index=cleaned_plsa_5["index"]

    cleaned_plsa_5.drop(["index"],axis=1,inplace=True)
    cleaned_bay.drop(["index"],axis=1,inplace=True)

    print("cleaned PLSA \n",cleaned_plsa_5.head(5),"Cleaned Baysian \n",cleaned_bay.head(5))
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    bay=cleaned_bay
    gnt=cleaned_plsa_5 #changing the gntm to plsa
    print("read sucessfully")
    # bay.drop(["Unnamed: 0"],axis=1,inplace=True) 
    # gnt.drop(["Unnamed: 0"],axis=1,inplace=True) 
    bay,gnt=bay.transpose(),gnt.transpose()
    bay_top_3,gnt_top_3=[],[]

    for col in range(bay.shape[1]):
        try:
            bay_top_3.append(bay.nlargest(n_largest,col).index) 
        except:
            pass

    for col in range(gnt.shape[1]):
        try:
            gnt_top_3.append(gnt.nlargest(n_largest,col).index)
        except:
            pass

    bay_col_names=[]
    gnt_col_names=[]
    for i in range(1,n_largest+1):
        bay_col_names.append(f'bay_top{str(i)}')
        gnt_col_names.append(f'gnt_top{str(i)}')
        
    bay_top3_df=pd.DataFrame(bay_top_3,columns=bay_col_names)
    gnt_top3_df=pd.DataFrame(gnt_top_3,columns=gnt_col_names)
    final_matches=[]

    for gnt_out,bay_out in zip(gnt_top_3,bay_top_3):
        matches=[]
        for out in gnt_out:
            if out in bay_out:
                matches.append(True)
            else:
                matches.append(False)
        percentage_of_acc=(matches.count(True)/len(matches))*100

        if percentage_of_acc >= match_threshold: 
            final_matches.append(True)
        else:
            final_matches.append(False)
            
    result=pd.concat([bay_top3_df,gnt_top3_df,pd.DataFrame(final_matches,columns=["match"])],axis=1)
    # decision=str(input("Do you want to save the o/p(yes/no)?"))
    decision="yes"
    if decision=="Yes" or decision=="yes" or decision=="YES":
        file_name=str(input("enter file name :"))
        result.to_csv(f"{file_name}.csv")
    
    try:
        result["match"].value_counts()[True]
    except:
        print('accuracy : 0 % ')
        
    try:
        result["match"].value_counts()[False]
    except:
        print('accuracy : 100%')

    try:
        accuracy=result["match"].value_counts()[True]/(result["match"].value_counts()[True]+result["match"].value_counts()[False])
        print("The accuracy is :",accuracy)
        with open(f"{no_topics}_plsa.txt","a") as op:
            op.writelines(f"\n Number of topics : {no_topics} \n N-largest : {n_largest} \n Match Threshold :{match_threshold} \n accuracy : {accuracy}")
    except:
        print("result already shown")



topics=str(input("enter no. of topics :"))
n_largest=int(input("enter n_largest : "))
bay_path=f"../nb_outputs/news_20/baysian_topic_{topics}.csv"
plsa_path=f"../plsa_outputs/news_20/{topics}_topics.csv"
    

thres=[60,80,100]

for threshold in thres:
    print("topics:",topics)
    print("n largest :",n_largest)
    print("Threshold is :",threshold)
    accuracy(plsa_path,bay_path,topics,n_largest,threshold)