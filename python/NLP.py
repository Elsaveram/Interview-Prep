import nltk
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('https://s3.amazonaws.com/nycdsabt01/reddit_top10.csv')

df.dtypes
df.sample(10)
df.head()

###########Regular Expressions###############################################
#Module re
import re
re.search('a.', 'aa') != None #To test if the pattern is present in the string

#Parentheses make groups
s = re.search('(..)/(..)/(201.)', 'From 06/01/2015')
print(s.group(1))

#? matches the preceding expression either once or zero times.
#+ matches the preceding expression character at least once.
#* matches the preceding expression character arbitrary times.
#{m,n} matches the preceding expression at least m times and at most n times.

print(re.search('ba?b', 'bb') != None)
print(re.search('ba+b', 'baab') != None)  # match
print(re.search('ba*b', 'bab') != None)   # match
print(re.search('ba{1,3}b', 'baaab') != None)  # match

#^ refers to the beginning of a text, while $ refers to the ending of a text.

print(re.search('^a', 'abc') != None)    # match
print(re.search('a$', 'abcba') != None)  # match

#[] characters that you wish to match. For example, [123abc]
# Same as [1-3a-c], [a-z] matches all lower case letters
# [0-9] matches all numbers
# Special characters lose their special meaning inside sets, [(+*)]
# '^', all the characters that are not in the set will be matched.

print(re.search('[1-3a-c]', '2defg') != None)

#() the brackets group the expressions contained inside them
print(re.search('(abc){2,3}', 'abcabcabc')  != None)

#| is a logical operator. For examples, a|b matches a or b
print(re.search('abc|123', 'a') != None)

#If you want to match exactly ?, it is necessary to add a backslash \?
print(re.search('\?', 'Hi, how are you today?') != None)

###########Clean the text###############################################

# Fill na with empty string
df['selftext'] = df['selftext'].fillna('')
# Replace `removed` and `deleted` with empty string
tbr = ['[removed]', '[deleted]']
df['selftext'] = df['selftext'].apply(lambda x: '' if x in tbr else x)

#Most of the column selftext is empty values therfore we concatenate with its title
sum(df['selftext']==''))/df.shape[0]
df['selftext'] = df['title'] + ' ' + df['selftext']

df['selftext'].sample(3)

###########Preprocessing###############################################

#1.Filtering

#Removing punctuation
df['selftext']=df['selftext'].apply(lambda x: re.sub('[^\w\s]','',x))

#Removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop.append('computer')
print(stop)

df['selftext']=df['selftext'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop))

#Tokenization
sample_text = "This is a toy example. Illustrate this example below."
sample_tokens = sample_text.split()
print(sample_tokens)

from nltk.tokenize import word_tokenize #Treats '.' as a word
word_tokenize(sample_text)

from textblob import TextBlob
TextBlob(sample_text).words #Treats '.' as a period

#Stemming and Lemmatization

nonprocess_text = "I am writing a Python string"

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_text = ' '.join([stemmer.stem(word) for word in nonprocess_text.split()])
print(stemmed_text)

from nltk import WordNetLemmatizer

lemztr = WordNetLemmatizer()
lemztr.lemmatize('feet')

#N-Grams
TextBlob(df['selftext'][10]).ngrams(2)

#Wordcloud
from wordcloud import WordCloud

wc = WordCloud(background_color="white", max_words=2000)
# generate word cloud
wc.generate(''.join(df['selftext']))

import matplotlib.pyplot as plt
%matplotlib inline

#%%
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure(figsize=(4, 3))
plt.axis("off")
plt.show()
#%%

#Sentimant Analysis
sample_size = 10000

def sentiment_func(x):
    sentiment = TextBlob(x['selftext'])
    x['polarity'] = sentiment.polarity
    x['subjectivity'] = sentiment.subjectivity
    return x

sample = df.sample(sample_size).apply(sentiment_func, axis=1)

sample.plot.scatter('ups', 'polarity')

#Word Embeddings
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)

#CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
X_train_count = count_vec.fit_transform(newsgroups_train.data)
X_train_count.shape

np.sum(X_train_count.todense()[0]) #tansform to a regular matrix

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(ngram_range=(1,2), min_df=10)
X_train_tf = tf_idf.fit_transform(newsgroups_train.data)
X_train_tf.shape

#Predicting methods
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
multiNB = MultinomialNB()

cntvecMNB = multiNB.fit(X_train_count, newsgroups_train.target)
tf_idfMNB = multiNB.fit(X_train_tf, newsgroups_train.target)

new_docs = ["""In the ancient and medieval world
            the etymological Latin root religio was understood as an individual virtue of worship
            never as doctrine, practice, or actual source of knowledge.
            Furthermore, religio referred to broad social obligations to family, neighbors, rulers, and even towards God.
            When religio came into English around the 1200s as religion, it took the meaning of "life bound by monastic vows".
            The compartmentalized concept of religion, where religious things were separated from worldly things,
            was not used before the 1500s. The concept of religion was first used in the 1500s to distinguish
            the domain of the church and the domain of civil authorities.""",

           """A graphics processing unit (GPU) is a specialized electronic circuit designed to rapidly manipulate and
           alter memory to accelerate the creation of images in a frame buffer intended for output to a display device.
           GPUs are used in embedded systems, mobile phones, personal computers, workstations, and game consoles.
           Modern GPUs are very efficient at manipulating computer graphics and image processing,
           and their highly parallel structure makes them more efficient than general-purpose CPUs
           for algorithms where the processing of large blocks of data is done in parallel.
           In a personal computer, a GPU can be present on a video card, or it can be embedded
           on the motherboard or—in certain CPUs—on the CPU die"""]

new_doc_count = count_vec.transform(new_docs)
new_doc_tfidf = tf_idf.transform(new_docs)
cnt_predicted = cntvecMNB.predict(new_doc_count)
tfidf_predicted = tf_idfMNB.predict(new_doc_count)

#LDA
from gensim.utils import simple_preprocess ##For LDA choose gensim ##It also allowd to save the object
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer

lemtzer = WordNetLemmatizer()

def lemmatize_stemming(text):
    return lemtzer.lemmatize(text, pos='v')

# Write a function to perform the pre processing steps on the entire dataset
def preprocess(text):
    result=[]
    for token in simple_preprocess(text) :
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result

document_num = 50
doc_sample = 'This disk has failed many times. I would like to get it replaced.'

print("Original document: ")
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print("\n\nTokenized and lemmatized document: ")
print(preprocess(doc_sample))

processed_docs  = []

for doc in newsgroups_train.data:
    processed_docs.append(preprocess(doc))
