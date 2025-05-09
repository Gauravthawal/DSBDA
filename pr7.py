# ! pip install nltk -U
# ! pip install bs4 -U

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import nltk
para='Rajgad is a hill fort situated in the pune district'
print(para)

para.split()
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
sent=sent_tokenize(para)
sent[0]
words = word_tokenize(para)
words

from nltk.corpus import stopwords
swords=stopwords.words('english')
swords

x=[word for word in words if word not in swords]
x

from nltk.stem import PorterStemmer
ps=PorterStemmer() 
ps.stem('working')

y=[ps.stem(word) for word in x]

y
from nltk.stem import WordNetLemmatizer

wnl=WordNetLemmatizer()

nltk.download('omw-1.4')
wnl.lemmatize('working',pos='v')
print(ps.stem('went'))
print(wnl.lemmatize('went',pos='v'))

z=[wnl.lemmatize(word,pos='v') for word in x]
z


import string
string.punctuation

t=[word for word in words if word not in string.punctuation]
t

from nltk import pos_tag
import nltk
nltk.download('averaged_perceptron_tagger_eng')

pos_tag(t)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()

v=tfidf.fit_transform(t)
v.shape

import pandas as pd
pd.DataFrame(v)
