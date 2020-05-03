import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
from sklearn import svm
from sklearn import metrics


#add labeled vector to dataset
headers=['airline_sentiment','text']
dataList=pd.read_csv("Tweets.csv", usecols=['airline_sentiment','text'])
#print(len(dataList['text']))
labelVector=[]
for i in range (len(dataList)):
    if ( dataList["airline_sentiment"].iloc[i]=="neutral"):
        labelVector.append(2)
    elif(dataList["airline_sentiment"].iloc[i]=="negative"):
        labelVector.append(0)
    elif(dataList["airline_sentiment"].iloc[i]=="positive"):
        labelVector.append(1)
dataList['Output']=labelVector

# for i ,k in zip(dataList['text'][:10],dataList['Output'][:10]):
#     print(i , k )
# print("------------------------------------------------------------------")
#tokenized tweets 
tokenizedLest=[]
temp_list=[]
for i in dataList['text']:
    word=nltk.word_tokenize(i)
    temp_list.append(word)
tokenizedLest=[temp_list,dataList['Output']]

#for i ,k in zip(tokenizedLest[0][:10],tokenizedLest[1][:10]):
#   print(i,k)


#generate feature vector using word embedding
model = Word2Vec(sentences=tokenizedLest[0], min_count=1, size=100,sg=0,window=5)

#generate sentance embedding using avearge of word embedding
avregeList=[]
for i in tokenizedLest[0]:
    sum=[]
    for j in i:
     sum.append(model.wv[j])
    npArray=np.array(sum)
    result=np.sum(npArray,0)
    avregeList.append(result/len(i))
tokenizedLest=[avregeList,dataList['Output']]   
#mask = tokenizedLest[1]['Output']==1

#print(mask)
#build ML Model usig SVM algorithm as a calssifier 
xTrain=[]
yTrain=[]
xTest=[]
yTest=[]


# Split dataset into training set and test set
# =============================================================================
i,counterP,counterN,counterNa=0,0,0,0
while (counterP+counterN+counterNa)<4500:  
     if tokenizedLest[1][i]==1 and counterP<1500:
         xTrain.append(tokenizedLest[0][i])
         yTrain.append(tokenizedLest[1][i])
         tokenizedLest[0].pop(i)
         tokenizedLest[1].pop(i)
         counterP+=1 
     elif tokenizedLest[1][i]==0 and counterN<1500:
         xTrain.append(tokenizedLest[0][i])
         yTrain.append(tokenizedLest[1][i])
         tokenizedLest[0].pop(i)
         tokenizedLest[1].pop(i)
         counterN+=1
     elif tokenizedLest[1][i]==2 and counterNa<1500:
         xTrain.append(tokenizedLest[0][i])
         yTrain.append(tokenizedLest[1][i])
         tokenizedLest[0].pop(i)
         tokenizedLest[1].pop(i)
         counterNa+=1  
     i+=1
     
xTest=tokenizedLest[0]
yTest=tokenizedLest[1]
# =============================================================================

#Create a svm Classifier
# =============================================================================
clf = svm.SVC(kernel='linear') # Linear Kernel
 
#Train the model using the training sets
clf.fit(xTrain, yTrain)
 
#Train the model using the training sets
clf.fit(xTrain, yTrain)
 
#Predict the response for test dataset
yPred = clf.predict(xTest)
 
 
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(yTest, yPred))
 
#getting a sentence input from user and run it on model
sentence=input("Enter you sentence: ")
sentence_tokenized=nltk.word_tokenize(sentence)
words_FV=[model.wv[x] for x in sentence_tokenized]
tempVF=np.array(words_FV)
result=np.sum(tempVF,0)
sentence_VF=result/len(sentence_tokenized)
sentence_out_predict=clf.predict([sentence_VF])
print(sentence_out_predict)
# =============================================================================
#print("Accuracy: %.2f%%" % (scores[1]*100))