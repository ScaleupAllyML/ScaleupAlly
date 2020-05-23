import pandas as pd
import os
import joblib
import pickle

########## Working Directory Set Up ####
os.chdir('F:\LocalDriveD\Analytics\Freelancing\Scaleup\BertQandA\Submission2')

from sentence_transformers import SentenceTransformer
import scipy.spatial

########## Bert Integration #############
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

#### Reading Answer Choices
Answers = pd.read_csv('Answers.txt', header=None)
Answers = list(Answers.iloc[:,0])
corpus_embeddings = embedder.encode(Answers)

#### Reading questions
Questions = pd.read_csv('Question.txt', header=None)
Questions = Questions.iloc[0,0]
Question = []
Question.append(Questions)
query_embeddings = embedder.encode(Question)

#### Algoritgm to identify most relavent answer
results_final = []
for query, query_embedding in zip(Question, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

#### Final Output
for idx, distance in results[0:1]:
    Final_Answer = Answers[idx]
    
    
#### Submission
file1 = open("PredictedAnswer.txt","w") 
file1.writelines(Final_Answer) 
file1.close() #to change file access modes 
