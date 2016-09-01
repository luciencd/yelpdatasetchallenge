import json
import random
import nltk
import time
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

##document frequency of words.
df = {}

#where are you importing data?
review_root = "yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"


#the actual json data object. A dictionary.
review_json = []


#the vector of the machine learning dataset.

review_vector = []

count = 0
def typedet():
    if(random.random()>0.5):
        return "Training"
    else:
        return "Test"


def collectText(reviews):

    for review in reviews:
        body = ""

        try:

            body = self.twe[i]['message']['body']
            symbols = self.twe[i]['message']['twitter_entities']['symbols']
            ##Need tool that can extract key words from string sentence.
        except KeyError:
            pass
        except IndexError:
            pass



        tokens = self.tokenize(body)
        ##why is uniques a set?
        ##shouldn't df
        uniques = set(tokens)

        for word in uniques:
            if word in self.df:
                self.df[word]+= 1
            else:
                self.df[word] = 1

                    
for line in open(review_root, 'r'):
    
    if(count > 10000):
        break
    
    review_json.append(json.loads(line))

    vec = {"text":json.loads(line)["text"],\
           "label":json.loads(line)["stars"],\
           "type":typedet()}
    
    review_vector.append(vec)
    count+=1


print "done"
pprint(review_json[0])
pprint(review_vector[0])

collectText()
##Parsing the text to create stemmed tdidf values.


