import json
import random
import nltk
import time
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import itertools
import math

#12 hour coding session
#May need Refactoring
#Lots of nltk dependencies.
## sample output:
'''
average restaurant rating: 3.69416666667

worst terms:
[[u'suff', -1.6941666666666668],
 [u'yel', -1.6941666666666668],
 [u'overcook', -1.6941666666666668],
 [u'heal', -1.6941666666666668],
 [u'attempt', -1.8480128205128206],
 [u'rud', -1.919166666666667],
 [u'worst', -1.9914639639639642],
 [u'disgust', -2.0275],
 [u'apolog', -2.131666666666667],
 [u'cli', -2.232628205128205]]

training error avg: 0.012595154931 
Test error avg: 1.28988015205 
Guess error avg: 1.96393718134
'''
#NLTK_DATA = '/Users/lucienchristie-dervaux/documents/RESTTwitter'

##need to customize this data
nltk.data.path.append('/Users/lucienchristie-dervaux/documents/RESTTwitter/nltk_data')
nltk.data.path.append('/home/vcap/app/nltk_data')
nltk.data.path.append('/Users/luciencd/Dropbox/Ibm/Documents/RESTTwitter/nltk_data')
nltk.data.path = nltk.data.path[1:]

from nltk.corpus import stopwords
print nltk.data.path

from pprint import pprint

#random.seed(122)


##stemmer
st = LancasterStemmer()

##lemmatizer
lt = WordNetLemmatizer()

##document frequency of words.
df = {}

##information gain


informationgain = {}


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
    body = "".join(c for c in body if c not in ('!','.',':'))
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
                

    listoftokens = [word for word in listoftokens if word not in stopwords.words('english')]

    return listoftokens


def collectText(reviews):
    global informationgain
    
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
        for word in uniques:
            if word in df:
                df[word]+= 1.0
            else:
                df[word] = 1.0

            #print word
            
            if word in informationgain:                
                informationgain[word].append(float(review['label']))
            else:
                informationgain[word] = [float(review['label'])]
            #print informationgain[word]
                
        stems = ""
        '''
 
        '''
        

               ##for bigrams too
        bigrams = ""
        try:
            
            bigrams = review['bigrams']
 
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass

        #print body

        ##why is uniques a set?
        ##shouldn't tf
        uniques = set(bigrams)
        #print uniques
        for word in uniques:
            if word in df:
                df[word]+= 1.0
            else:
                df[word] = 1.0


    for key, value in informationgain.iteritems():
        #print key,value,
        #print key,value
        avg = reduce(lambda x, y: x + y, value) / len(value)
        ##get variance
        value = map(lambda x: x-avg,value)
        value = map(lambda x: x**2,value)#square each of distances
        variance = reduce(lambda x, y: x + y, value)/len(value)#dividing sum of squares
 
        #print variance,"\n"
        informationgain[key] = variance
    
    ##information gain for single words:

        


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

#Tri-grams. looking to extend later
def bigrammize(tokens):
    bigrams = []
    for i in range(0,len(tokens)-2):
        bigrams.append(tokens[i]+" "+tokens[i+1]+" "+tokens[i+2])
    return bigrams

def makebigram(reviews):
    for review in reviews:
        stems = ""

        try:
            
            stems = review['stems']
 
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass

        bigrams = bigrammize(stems)
        review['bigrams'] = bigrams
    
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
    tfidflist = []
    for stem in stems:
        #print stem,stems.count(stem),"1.0/",documentfrequencies[stem]," ",1.0/documentfrequencies[stem]

        #computing tfidf score
        score = 0
        try:
            score = (float)(stems.count(stem))*(float)(1.0/documentfrequencies[stem])
        except KeyError:
            score = 0.0
            
        
        
                
        tfidflist.append((stem,score))
        
    for i in range(0,len(tfidflist)):
        stem = tfidflist[i][0]
        score = tfidflist[i][1]
        tfidf[stem] = score
        #print score
    #print tfidf
    return tfidf

    
def tfidf(reviews):
    for review in reviews:
        stems = ""

        try:
            
            stems = review['stems']
 
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass

        tfidfs = tfidfize(stems,df)
        
        review['tfidf'] = tfidfs

        ##bigrams
        bigrams = ""
        try:
            
            bigrams = review['bigrams']
 
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass

        tfidfsbigrams = tfidfize(bigrams,df)
        review['bigramstfidf'] = tfidfsbigrams

def distance(vector1,vector2):
    summation = 0.0
    matching = 0.0
    total = 0.0
    '''
    #print vector1["tfidf"],"\n\n",vector2["tfidf"]
    for key, value in vector1["tfidf"].iteritems():
        if key in vector2["tfidf"]:
            #print informationgain[key]
            summation += value*vector2["tfidf"][key]#*informationgain[key]
            matching+=1.0
        total+=1.0
    '''
    
    #print vector1["bigrams"]
    for key, value in vector1["bigramstfidf"].iteritems():
        
        if key in vector2["bigramstfidf"]:
            
            summation += value*vector2["bigramstfidf"][key]
            matching+=1.0
        total+=1.0
    

    '''        
    print "words matched", matching,"/",total
    print "matching:", matching/total
    print "summation:",summation
    print "cosine distance:", summation/(len(vector1["tfidf"])*len(vector2["tfidf"]))
    print "\n"
    '''
    #return matching/total
    #return summation/(len(vector1["tfidf"])*len(vector2["tfidf"]))
    #print len(vector1["bigramstfidf"]),(len(vector2["bigramstfidf"]))
    return (0.00000001+summation)/(1+((len(vector1["bigramstfidf"]))*(len(vector2["bigramstfidf"]))))

def labelize(review1,rev):
    reviews = sorted(rev,key=lambda review2:distance(review1,review2),reverse=True)

    label = 0
    sumdistance = 0
    #top 5 closest in n-dim space.
    for i in range(len(reviews)):
        #print "max"
        dist = distance(review1,reviews[i])
        label += reviews[i]['label']*dist
        
        sumdistance += dist
        #print label,sumdistance

    label = label/sumdistance
    return label

##
number = 1200.0

count = -(random.random()*10000)
for line in open(review_root, 'r'):
    
    if(count >= number):
        break
    if(count >= 0):
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
makebigram(review_vector)

training_vector = review_vector[0:int(number/2)]
test_vector = review_vector[int(number/2):]

collectText(training_vector)


tfidf(review_vector)
'''
print review_vector[0]["bigrams"],"\n:"
print review_vector[0]["tokens"],"\n:","\n:"
print review_vector[1]["bigrams"],"\n:"
print review_vector[1]["tokens"],"\n:","\n:"
print review_vector[2]["bigrams"],"\n:"
print review_vector[2]["tokens"],"\n:","\n:"
'''
#print distance(review_vector[0],review_vector[3])

avgwordrating = {}

##getting the most negative and positive words
averagerating = 0.0
for review in review_vector:
    actual_label = float(review["label"])
    for word in review["stems"]:
        #print word
        if word in df:
            if df[word]>5:
                
                if word in avgwordrating:
                    
                    avgwordrating[word][0]+=float(review["label"])
                    avgwordrating[word][1]+=1.0
                else:
                    avgwordrating[word] = [float(review["label"]),1.0]

    averagerating += float(review["label"])
    
avgvec = []

for key, value in avgwordrating.iteritems():
    temp = [key,value]
    avgvec.append(temp)
    
print "average restaurant rating:",averagerating/len(review_vector)

##To be honest, best terms is pretty awful, something must be wrong with it...
print "best terms:"
newvector = map(lambda word: [word[0],word[1][0]/word[1][1] - averagerating/len(review_vector)],avgvec)

newlist = sorted(newvector,key=lambda word: word[1],reverse=True)

pprint(newlist[0:10])

print "worst terms:"
pprint(newlist[len(newlist)-10:])
trainingtotal = 0.0
testtotal = 0.0
trainingnum = 0.0
testnum = 0.0

guesstotal = 0.0
guessnum = 0.0
i=0.0

for review in review_vector:
    
    predicted_label = labelize(review,training_vector)
    actual_label = review["label"]
    guess_label = random.random()*5
    
    

    error = math.fabs(actual_label - predicted_label)
    #print i,review["type"],"pred:",predicted_label,"   actual:",actual_label,"error:",error
    if(i%(len(review_vector)/10) == 0):
        print str(100*(i/len(review_vector)))+" percent done"
    guess_error = math.fabs(actual_label - guess_label)

    if(review["type"] == "Training"):
        trainingtotal +=float(error)
        trainingnum +=1.0
    else:
        testtotal += float(error)
        testnum +=1.0

    guesstotal +=guess_error
    guessnum+=1.0
    i+=1.0
    
print "training error avg:",trainingtotal/trainingnum,\
      "\nTest error avg:",testtotal/testnum,\
      "\nGuess error avg:",guesstotal/guessnum

    
    
#for review in review_vector:
#    if(review['type'] == "Test"):
        
#print review_vector[0]
#pprint(tf)
##Parsing the text to create stemmed tdidf values.


