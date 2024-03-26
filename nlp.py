import nltk

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from nltk.stem import WordNetLemmatizer


para = "Modi was born and raised in Vadnagar in northeastern Gujarat, where he completed his secondary education. He was introduced to the RSS at the age of eight. His account of helping his father sell tea at the Vadnagar railway station has not been reliably corroborated. At age 18, he was married to Jashodaben Modi, whom he abandoned soon after, only publicly acknowledging her four decades later when legally required to do so. Modi became a full-time worker for the RSS in Gujarat in 1971. The RSS assigned him to the BJP in 1985 and he held several positions within the party hierarchy until 2001, rising to the rank of general secretary."

para


nltk.download('punkt')
nltk.download('stopwords')

ps=PorterStemmer()
sent=nltk.sent_tokenize(para,language="english")

corpus=[]
for i in range(len(sent)):
    r=re.sub('[^a-zA-Z]',' ',sent[i])
    r=r.lower()
    
    words=r.split()
    r=[ps.stem(word)for word in words if not word in set(stopwords.words('english'))]
    r=' '.join(r)
    corpus.append(r)

corpus

#creating Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=100)
dataSet=cv.fit_transform(corpus).toarray()

dataSet

#creating TF AND IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer()
dataset2=tv.fit_transform(corpus).toarray()

dataset2
#creating word2vec
!pip install gensim
import nltk
from gensim.models.doc2vec import Word2Vec
from nltk.corpus import stopwords
import re

para = "Modi was born and raised in Vadnagar in northeastern Gujarat, where he completed his secondary education. He was introduced to the RSS at the age of eight. His account of helping his father sell tea at the Vadnagar railway station has not been reliably corroborated. At age 18, he was married to Jashodaben Modi, whom he abandoned soon after, only publicly acknowledging her four decades later when legally required to do so. Modi became a full-time worker for the RSS in Gujarat in 1971. The RSS assigned him to the BJP in 1985 and he held several positions within the party hierarchy until 2001, rising to the rank of general secretary."

#preprocessing
#for number
text=re.sub('[0-9]',' ', para)

#for conversion of lower cas
text=text.lower()
r=re.sub('[^a-zA-Z]',' ',text)
text=re.sub(r'[^a-zA-Z]',' ',text)
text=re.sub(r'\s',' ',text)

text

sent=nltk.sent_tokenize(para)
sent

sent=nltk.sent_tokenize(text)
sent=[nltk.word_tokenize(s) for s in sent]
for i in range (len(sent)):
    sent[i]=[word for word in sent[i]if word not in set (stopwords.words('english'))]

for s in sent:
    print(s)

#train 
model=Word2Vec(sent,min_count=1)
words=list(model.wv.index_to_key)

s=model.wv.most_similar('modi')
s



