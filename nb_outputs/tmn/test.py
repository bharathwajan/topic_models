import re
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB 

def fun():
    return [0 for i in range(no_topics)]

exp=r'[a-z]+'
no_topics=5
with open('tagmynews.txt',encoding='utf-8') as doc:
    txt=doc.readlines()
    feature_names=txt[0]
    newses=txt[1:]
newses=[" ".join(re.findall(exp,news.lower())) for news in newses] #conerting to lowercase and removing punctuations
exp='[a-zA-Z]+'
topics_words={}

with open(f'topic{str(no_topics)}.txt',encoding='utf-8') as doc:
    txt=doc.readlines()
    feature_names=[line.split(" ")[0] for line in txt]
# print("feature names",feature_names) #unique no. of words

for line in txt:
    # print(line)
    key=line.split(" ")[0]
    key=int(re.sub(exp,"",key))
    value=line.split(" ")[1:]
    topics_words[key]=value
# print('topic_words',topics_words) #dictionary where topic is the key and values are words

counter=defaultdict(fun)
for news in range(len(newses)):
    # print(news,newses[news])
    for word in newses[news].split(" "):
        if word!="" and word!=" ":
            for key,value in topics_words.items():
                if word in value:
                    counter[news][key]+=1
# print('counter',counter) #contains how many times a news contains words associated with each topics
keys=counter.keys() 
# print('keys',keys,'len of keys',len(keys))

missing_values=[]
for idx in range(len(newses)):
    if idx not in keys:
        # print(idx)
        missing_values.append(idx)

doc_topic_mat= pd.DataFrame.from_dict(counter).transpose()
doc_topic_mat.columns=feature_names
doc_topic_mat
doc_topic_mat.to_csv("count_matrix.csv")

probs=defaultdict(fun)
for row in doc_topic_mat.iterrows():
    # print(row)
    total=row[1].sum()
    for val in range(len(row[1])):
        if total!=0:
            dist=row[1][val]/total
            probs[row[0]][val]=dist
        else:
            dist=row[1][val]
            # print(row[0])
            probs[row[0]][val]=dist
final_dict=dict()
for key in sorted(probs.keys()):
    final_dict[key]=probs[key]
prob_dist=pd.DataFrame.from_dict(final_dict).transpose()
prob_dist
# prob_dist.to_csv(f"gntm_topic_{no_topics}.csv")
# print("GNTM topic prob is generated")

# print('######### Topics words ############ \n',topics_words)
df=[]
for key,value in  topics_words.items():
    df.append(" ".join(value))
df=pd.DataFrame(df,columns=['WORDS'])
# print(df,'\n',df.shape)
cv = CountVectorizer()
x1 = cv.fit(df["WORDS"])
train=x1.transform(df["WORDS"])
y=list(df.index)
train
# print("train shape",train.shape,"\n Train",train,y)
TOPIC_MODEL= MultinomialNB().fit(train,y)
# print("Class prior",priors)
print("Model Trained !")
# print('newses',newses)
test=x1.transform(newses) #changing all the newses in the dataset according to our count vectorizer
test=test[1]
# print("Test shape",test.shape)
# temp=pd.DataFrame(test)
# # print(temp)
Y_prob=TOPIC_MODEL.predict_proba(test)
# bays_prob=pd.DataFrame(Y_prob)
# bays_prob.drop(missing_values,axis=0,inplace=True)
# bays_prob.to_csv(f"baysian_topic_{no_topics}.csv")
# print("Bayesian topic prob is generated")