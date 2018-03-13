## Python
#import requests
#
#value = 'Rum'
#data = requests.get('http://dbpedia.org/data/'+value+'.json').json()
#page = data['http://dbpedia.org/resource/'+value]
#types = page["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]
#for ind in range(len(types)):
#    if  "/ontology/" in types[ind]['value']:
#        print(types[ind]['value'])
#        
#
#data = requests.get('http://dbpedia.org/data/Montesquieu.json').json()
#page = data['http://dbpedia.org/resource/Montesquieu']
#desc = page["http://purl.org/dc/terms/description"][0]['value']
#subjects = page["http://purl.org/dc/terms/subject"]
#types = page["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]
#for ind in range(len(types)):
#    if  "/ontology/" in types[ind]['value']:
#        print(types[ind]['value'])
        
        
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
#
#values = ['Rum','Voltaire']
#results = []
#sparql = SPARQLWrapper("http://dbpedia.org/sparql")
#for i in range(len(values)):
#    sparql.setQuery("""
#        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#        SELECT *
#        WHERE {<http://dbpedia.org/resource/"""+values[i]+"""> dct:subject ?subject }
#    """)
#    sparql.setReturnFormat(JSON)
#    results.append(sparql.query().convert())
dataset_result = {}    
results = []
tab = ['French_philosophers','English_philosophers']
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery("""
    PREFIX cat: <http://dbpedia.org/resource/Category:>
    select ?person
    WHERE {
    ?person dct:subject cat:"""+tab[0]+""".
    } 
""")
sparql.setReturnFormat(JSON)
results.append(sparql.query().convert())

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery("""
    PREFIX cat: <http://dbpedia.org/resource/Category:>
    select ?person
    WHERE {
    ?person dct:subject cat:"""+tab[1]+""".
    } 
""")
sparql.setReturnFormat(JSON)
results.append(sparql.query().convert())
for j in range(2):
    for i in range(len(results[j]['results']['bindings'])):
        
        print(results[j]['results']['bindings'][i]['person']['value'])
        sparql.setQuery("""
            PREFIX cat: <http://dbpedia.org/resource/Category:>
            prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            select ?o ?p where {
                 <"""+ results[j]['results']['bindings'][i]['person']['value'] +"""> ?p ?o.
            }
        """)
        sparql.setReturnFormat(JSON)
        dataset_result[results[j]['results']['bindings'][i]['person']['value']] = sparql.query().convert()

results.append(sparql.query().convert())


training = {}

for i in dataset_result:
    k = 0
    tampon = i.split("/")[len(i.split("/")  ) - 1]
    training[tampon] = {} 
    for att in dataset_result[i]["results"]["bindings"]:
       
        tampon2 =att["o"]["value"].split("/")[len(att["o"]["value"].split("/")  ) - 1] 
        tampon3 =att["p"]["value"].split("/")[len(att["p"]["value"].split("/")  ) - 1] 
        if(not (tampon3 in training[tampon]) ):
            training[tampon][tampon3] = {}
            k = 0

        else:
            k = k+ 1
        
        training[tampon][tampon3]["valeur " + str(k)] = tampon2
#        
#category = 'French_philosophers'
#root = 'test'
#while root is not None:
#    results = []
#    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
#    sparql.setQuery("""
#        PREFIX res: <http://dbpedia.org/resource/Category:>
#        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
#        select ?cat
#        WHERE {
#        ?cat skos:broader res:"""+ category + """.
#        } LIMIT 15
#        """)
#    sparql.setReturnFormat(JSON)
#    results.append(sparql.query().convert())

#    print(dataset_result[i])

##########################################Count Attribut####################################################

attCount = {}
for key1 in training:
    for key2 in training[key1]:
        attCount[key2] = {}
        attCount[key2] = 0
        
for key1 in training:
    for key2 in training[key1]:
        attCount[key2] = attCount[key2] + 1
        
nom = []
for key in training:
    nom.append(key)


#########################################Count Subjet########################################################
subjectCount = {}
for key1 in training:
    for key2 in training[key1]["subject"]:
        subjectCount[training[key1]["subject"][key2]] = {}
        subjectCount[training[key1]["subject"][key2]] = 0
        
for key1 in training:
    for key2 in training[key1]["subject"]:
#        print(training[key1]["subject"][key2])
        subjectCount[training[key1]["subject"][key2]] = subjectCount[training[key1]["subject"][key2]] + 1    
        
        
##########################################RDF type Count#######################################################
        
import operator
#french/english philosopher, Living people, writers, 
#subject, gender, 
#Les deux lignes qui ne contiennent pas de rdf:type
#Deleuze_and_Guattari
#List_of_French_philosophers
del training['Deleuze_and_Guattari']
del training['List_of_French_philosophers']
rdfTypeCount = {}
for key1 in training:    
        for key2 in training[key1]['22-rdf-syntax-ns#type'].values():
            rdfTypeCount[key2] = {}
            rdfTypeCount[key2] = 0
for key1 in training:    
    for key2 in training[key1]['22-rdf-syntax-ns#type'].values():
        rdfTypeCount[key2] =rdfTypeCount[key2] +1
y = sorted(rdfTypeCount.items(), key=operator.itemgetter(1), reverse=True)


########################################gender########################################

gender = []
nom_gender = []
for key1 in training:
    for key2 in training[key1]:
        if(key2 == "gender"):
            nom_gender.append(key1)
            gender.append(training[key1][key2]["valeur 0"])


df = pd.DataFrame()

df["nom_philo"] = nom
df["gender"] = None
i = 0
for key in nom_gender:
    row = df.loc[df['nom_philo']==key].index.values[0]
#    print(row)
    print(key)
    print(gender[i])
    df["gender"][row] = gender[i]
    i = i + 1
#    print(i)


#######################################death date######################################

df["deathDate"] = None
death_date = []
nom_death_date = []
 
for key1 in training:
    for key2 in training[key1]:
        if(key2 == "deathDate"):
            nom_death_date.append(key1)
            death_date.append(training[key1][key2]["valeur 0"])

i = 0
for key in nom_death_date:
    row = df.loc[df['nom_philo']==key].index.values[0]
#    print(row)
    print(key)
    print(gender[i])
    df["deathDate"][row] = death_date[i].split("-")[0]
    i = i + 1
#    print(i)            
            
            
#######################################birthDate######################################

df["birthDate"] = None
birth_Date = []
nom_birth_Date = []
 
for key1 in training:
    for key2 in training[key1]:
        if(key2 == "birthDate"):
            nom_birth_Date.append(key1)
            birth_Date.append(training[key1][key2]["valeur 0"])

i = 0
for key in nom_birth_Date:
    row = df.loc[df['nom_philo']==key].index.values[0]
#    print(row)
    print(key)
    print(gender[i])
    df["birthDate"][row] = birth_Date[i].split("-")[0]
    i = i + 1
#    print(i)
###################################death date owner#####################################
df["deathDateowner"] = None
for row in range(df.shape[0]):
    if(df["deathDate"][row] is None):
        df["deathDateowner"][row] = 0   
    else:
        df["deathDateowner"][row] = 1
        

##################################rdf type WikicatEnglishPhilosophers####################
df["WikicatEnglishPhilosophers"] = None
WikicatEnglishPhilosophers = []
nom_WikicatEnglishPhilosophers = []

 
for key1 in training:
    for key2 in training[key1]:
        if(key2 == "22-rdf-syntax-ns#type"):
           for key3 in training[key1][key2]:
               if(training[key1][key2][key3] == "WikicatEnglishPhilosophers"):
                   nom_WikicatEnglishPhilosophers.append(key1)
                   WikicatEnglishPhilosophers.append(training[key1][key2][key3])


i = 0
for key in nom_WikicatEnglishPhilosophers:
    row = df.loc[df['nom_philo']==key].index.values[0]
#    print(row)
    print(key)
    print(gender[i])
    df["WikicatEnglishPhilosophers"][row] = WikicatEnglishPhilosophers[i]
    i = i + 1

##################################rdf type WikicatFrenchPhilosophers####################
df["WikicatFrenchPhilosophers"] = None
WikicatFrenchPhilosophers = []
nom_WikicatFrenchPhilosophers = []

 
for key1 in training:
    for key2 in training[key1]:
        if(key2 == "22-rdf-syntax-ns#type"):
           for key3 in training[key1][key2]:
               if(training[key1][key2][key3] == "WikicatFrenchPhilosophers"):
                   nom_WikicatFrenchPhilosophers.append(key1)
                   WikicatFrenchPhilosophers.append(training[key1][key2][key3])


i = 0
for key in nom_WikicatFrenchPhilosophers:
    row = df.loc[df['nom_philo']==key].index.values[0]
#    print(row)
    print(key)
    print(gender[i])
    df["WikicatFrenchPhilosophers"][row] = WikicatFrenchPhilosophers[i]
    i = i + 1



############################################Subject###########################################

##French philosophers
df["class_French_philosophers"] = None
class_French_philosophers = []
nom_French_philosophers = []
 
for key1 in training:
    for key2 in training[key1]:
        if(key2 == "subject"):
            for key3 in training[key1][key2]:
               print(key3)
               if(training[key1][key2][key3] == "Category:French_philosophers"):
                   nom_French_philosophers.append(key1)
                   print(training[key1][key2][key3])
                   class_French_philosophers.append(training[key1][key2][key3])

i = 0
for key in nom_French_philosophers:
    row = df.loc[df['nom_philo']==key].index.values[0]
#    print(row)
    df["class_French_philosophers"][row] = class_French_philosophers[i].split("-")[0]
    i = i + 1        


##English philosophers
df["class_English_philosophers"] = None
class_English_philosophers = []
nom_English_philosophers = []
 
for key1 in training:
    for key2 in training[key1]:
        if(key2 == "subject"):
            for key3 in training[key1][key2]:
               print(key3)
               if(training[key1][key2][key3] == "Category:English_philosophers"):
                   nom_English_philosophers.append(key1)
                   print(training[key1][key2][key3])
                   class_English_philosophers.append(training[key1][key2][key3])

i = 0
for key in nom_English_philosophers:
    row = df.loc[df['nom_philo']==key].index.values[0]
#    print(row)
    df["class_English_philosophers"][row] = class_English_philosophers[i].split("-")[0]
    i = i + 1



##Living people
df["Class_Living_people"] = None
Living_people = []
nom_English_philosophers = []
 
for key1 in training:
    for key2 in training[key1]:
        if(key2 == "subject"):
            for key3 in training[key1][key2]:
               print(key3)
               if(training[key1][key2][key3] == "Category:Living_people"):
                   nom_English_philosophers.append(key1)
                   print(training[key1][key2][key3])
                   Living_people.append(training[key1][key2][key3])

i = 0
for key in nom_English_philosophers:
    row = df.loc[df['nom_philo']==key].index.values[0]
#    print(row)
    df["Class_Living_people"][row] = Living_people[i]
    i = i + 1


##21h century
df["Class_20th-century_philosophers"] = None
TwentyCenturyPhilosophers = []
nom_20CenturyPhilosophers = []
 
for key1 in training:
    for key2 in training[key1]:
        if(key2 == "subject"):
            for key3 in training[key1][key2]:
               print(key3)
               if(training[key1][key2][key3] == "Category:20th-century_philosophers"):
                   nom_20CenturyPhilosophers.append(key1)
                   print(training[key1][key2][key3])
                   TwentyCenturyPhilosophers.append(training[key1][key2][key3])

i = 0
for key in nom_20CenturyPhilosophers:
    row = df.loc[df['nom_philo']==key].index.values[0]
#    print(row)
    df["Class_20th-century_philosophers"][row] = TwentyCenturyPhilosophers[i]
    i = i + 1






dfbase = df.copy()
dfneg = df.copy()









############################################Data Set 0 et 1 ################################
#0 --> ne possède pas 
#1 --> possède
columns = ["Class_20th-century_philosophers","WikicatEnglishPhilosophers","WikicatFrenchPhilosophers","class_French_philosophers","class_English_philosophers","Class_Living_people"]
for column in columns:
    for col in range(df.shape[0]):
        if(df[column][col] is None):
            df[column][col] = 0
        else:
            df[column][col] = 1
        

dfneg = df.copy()


####################################################Date de naissance et de mort##############
          
###################################test -1500 ##########################################
columns = ["deathDate","birthDate"]
for column in columns:
    for col in range(df.shape[0]):
        print(df[column][col])
        if(df[column][col] is None):
            df[column][col] = 0
            dfneg[column][col] = -500
            
        
           

for column in columns:
    for col in range(df.shape[0]):
        if(df[column][col] == "Unknown"):
           print("damn")
           asupr = col
           print(col)

df = df.drop(asupr)
dfneg = dfneg.drop(asupr)

columns = ["WikicatEnglishPhilosophers","WikicatFrenchPhilosophers","class_French_philosophers","class_English_philosophers","Class_Living_people","deathDate","birthDate","deathDateowner"]

for column in columns:
    df[column] = df[column].astype(int)
    dfneg[column] = dfneg[column].astype(int)


df_predict = df.drop(["WikicatFrenchPhilosophers","WikicatEnglishPhilosophers","gender","nom_philo","deathDateowner","class_French_philosophers","class_English_philosophers","Class_20th-century_philosophers"],axis = 1)
#df_predict = df_predict.drop(["deathDate"],axis = 1)
#df_predict = df_predict.drop(["birthDate"],axis = 1)

############################################Neural Network###########################################
df_predict = df_predict.reset_index(drop=True)

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
	# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y = df_predict["Class_Living_people"]
x = df_predict.drop("Class_Living_people",axis = 1)
xtrain, xtest,ytrain, ytest = train_test_split( x, y, test_size=0.33, random_state=0)
model.fit(xtrain.values,ytrain,batch_size = 5,epochs = 100)
ypred = model.predict(xtest.values)    
ypred = (ypred > 0.5)
ypred_f = np.zeros(len(ypred))
for i in range(len(ypred)):
    if(ypred[i]):
        ypred_f[i]=1
    else:
        ypred_f[i]=0

a = accuracy_score(ytest,ypred_f)

      