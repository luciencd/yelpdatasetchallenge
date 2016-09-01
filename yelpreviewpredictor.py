import json
import random
import nltk
import time
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import itertools
import math
#NLTK_DATA = '/Users/lucienchristie-dervaux/documents/RESTTwitter'

##need to customize this data
nltk.data.path.append('/Users/lucienchristie-dervaux/documents/RESTTwitter/nltk_data')
nltk.data.path.append('/home/vcap/app/nltk_data')
nltk.data.path.append('/Users/luciencd/Dropbox/Ibm/Documents/RESTTwitter/nltk_data')
nltk.data.path = nltk.data.path[1:]

from nltk.corpus import stopwords
print nltk.data.path

from pprint import pprint

random.seed(122)


##stemmer
st = LancasterStemmer()

##lemmatizer
lt = WordNetLemmatizer()

##document frequency of words.
df = {}

#where are you importing data?
review_root = "../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"


#the actual json data object. A dictionary.
review_json = []


#the vector of the machine learning dataset.

review_vector = []

count = 0
def typedet2(n,number):
    if(n/number<0.5):
        return "Training"
    else:
        return "Test"

def typedet():
    if(random.random()>0.5):
        return "Training"
    else:
        return "Test"

def tokenize(body):

    #print body
    ##map single string body to sentence
    sentences = nltk.tokenize.sent_tokenize(body)
    #print sentences

    ##map each sentence to single word tokens.
    tokens = [nltk.tokenize.word_tokenize(s) for s in sentences]
    #print tokens


    ##keep only top 4 words with highest tdidf score in all sentences,
    ##where each sentence is a separate document(or not for now)
    listoftokens = []
    for sentence in tokens:

        for token in sentence:
            #print token
            ##time to normalize every token

            normalizedToken = token.lower()
            listoftokens.append(normalizedToken)
                

    
    return listoftokens


def collectText(reviews):

    for review in reviews:
        stems = ""

        try:
            
            stems = review['stems']
 
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass

        #print body

        ##why is uniques a set?
        ##shouldn't tf
        uniques = set(stems)
        #print uniques
        for word in stems:
            if word in df:
                df[word]+= 1.0
            else:
                df[word] = 1.0


def maketokens(reviews):
    for review in reviews:
        body = ""

        try:
            
            body = review['text']
 
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass

        tokens = tokenize(body)
        review['tokens'] = tokens
        

def makestems(reviews):
    for review in reviews:
        tokens = ""

        try:
            
            tokens = review['tokens']
 
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass

        stems = stemmize(tokens)
        review['stems'] = stems

def makelemmas(reviews):
    for review in reviews:
        tokens = ""

        try:
            
            tokens = review['tokens']
 
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass

        stems = lemmatize(tokens)
        review['lemmas'] = stems
        

def stemmize(tokens):
    stems = []
    for token in tokens:
        stems.append(st.stem(token))

    return stems

def lemmatize(tokens):
    lemmas = []
    for token in tokens:
        lemmas.append(lt.lemmatize(token))

    return lemmas
        
def tfidfize(stems,documentfrequencies):
    tfidf = {}
    for stem in stems:
        #print stems.count(stem),"1.0/",documentfrequencies[stem]," ",1.0/documentfrequencies[stem]
        score = 0
        try:
            score = (float)(stems.count(stem))*(float)(1.0/documentfrequencies[stem])
        except KeyError:
            score = 0.0
            
        tfidf[stem] = score
        #print score
    return tfidf

    
def tfidf(reviews):
    for review in reviews:
        tokens = ""

        try:
            
            tokens = review['stems']
 
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass

        tfidfs = tfidfize(tokens,df)
        
        review['tfidf'] = tfidfs

def distance(vector1,vector2):
    summation = 0.0
    matching = 0.0
    total = 0.0
    #print vector1["tfidf"],"\n\n",vector2["tfidf"]
    for key, value in vector1["tfidf"].iteritems():
        if key in vector2["tfidf"]:
            summation += value*vector2["tfidf"][key]
            matching+=1.0
        total+=1.0

    for key, value in vector2["tfidf"].iteritems():
        if key in vector1["tfidf"]:
            summation += value*vector1["tfidf"][key]
            matching+=1.0
        total+=1.0
    '''        
    print "words matched", matching,"/",total
    print "matching:", matching/total
    print "summation:",summation
    print "cosine distance:", summation/(len(vector1["tfidf"])*len(vector2["tfidf"]))
    print "\n"
    '''
    return summation/(len(vector1["tfidf"])*len(vector2["tfidf"]))

def labelize(review1,rev):
    reviews = sorted(rev,key=lambda review2:distance(review1,review2),reverse=True)

    label = 0
    sumdistance = 0
    #top 5 closest in n-dim space.
    for i in range(min(5,len(reviews))):
        #print "max"
        dist = distance(review1,reviews[i])
        label += reviews[i]['label']*dist
        
        sumdistance += dist
        #print label,sumdistance

    label = label/sumdistance
    return label

number = 4.0
count = 0.0
for line in open(review_root, 'r'):
    
    if(count >= number):
        break
    
    review_json.append(json.loads(line))

    vec = {"text":json.loads(line)["text"],\
           "label":json.loads(line)["stars"],\
           "type":typedet2(count,number)}
    
    review_vector.append(vec)
    count+=1.0


print "done"
#pprint(review_json[0])
#pprint(review_vector[0])

maketokens(review_vector)

makestems(review_vector)
#makelemmas(review_vector)##doesn't work apparently.

training_vector = review_vector[0:int(number/2)]
test_vector = review_vector[int(number/2):]
collectText(training_vector)


tfidf(review_vector)
#print distance(review_vector[0],review_vector[3])



trainingtotal = 0.0
testtotal = 0.0

for review in review_vector:
    
    predicted_label = labelize(review,training_vector)
    actual_label = review["label"]
    print review["type"],"pred:",predicted_label,"   actual:",actual_label
    error = Math.abs(actual_label - predicted_label)

    
    
#for review in review_vector:
#    if(review['type'] == "Test"):
        
#print review_vector[0]
#pprint(tf)
##Parsing the text to create stemmed tdidf values.


