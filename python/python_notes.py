import pandas as pd
import numpy as np
from scipy import stats
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

####PANDAS##############################################################################################################

#Series
obj=pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
obj.index
obj[1]
obj[4]='d'
obj.values
dict_={1:'a', 2:'b', 3:'c'}
obj2=pd.Series(dict_)
obj2


#DataFrame
#From dict
data={'commodity':['Gold', 'Gold', 'Silver','Silver'], 'year':[2013, 2014, 1015, 2016]}
#From nested list
data2=pd.DataFrame([['Gold', 2013], ['Gold', 2014], ['Silver', 2015], ['Silver', 2016]], columns=['commodity', 'year'])
type(data2)

df=pd.DataFrame(data)
type(df)
df.columns
df.index
df.set_index('year')
df.values
df.reset_index
df.year
df['year']
df.columns
df.columns=['commodity', 'year']
df.columns

dir(df.columns.str)
df.columns=df.columns.str.replace('year', 'y')
df

#Slice a df
df['commodity'][0]
df['commodity'][1:3]

boro=['a', 'b', 'c']
number=['1', '2', '3']

#Creating df
d={'boro':boro, 'number':number}
NYC=pd.DataFrame(d)

#to/from csv
NYC.to_csv('NYC.csv')
pd.read_csv('NYC.csv')

#joining dfs
df1=pd.DataFrame(np.arange(9).reshape(3,3), columns=['a','b','c'], index=['one', 'two', 'three'])
df2=pd.DataFrame(np.arange(6).reshape(3,2), columns=['b','e'], index=['One','two','three'])
pd.concat([df1, df2], axis=1, join='inner')

#sorting
df1
df1.sort_values('b', ascending=False)
df1

#merging
pd.merge(df1, df2, how='inner', on='b')
pd.merge(df1, df2, left_on='a', right_on='e')

#To extract a column as a Series or df
type(df1['a'])
type(df1[['a']])

#Selecting a row
#loc accepts labels
df1.loc['one','b']
df1['b'][0]
df1.loc[df1.a==0,:]
df1.loc[:, df1.loc['one']==1]

#iloc accepts numbers
df1.iloc[0, :2]
df1.iloc[[0,2], :2]

#Applying a function to the column of the df
df1
df1.apply(lambda x: min(x), axis=1)

#Applying a function to a specific column of the df
df1.a.map(lambda x: x+1)

#Removing data
df1.loc[df1.index != 'two']
df1.drop('a', 1)
df1.loc[:, df1.columns != 'b']

#Missing data
df.isnull()
np.sum(df.isnull(), axis=1)
mask=df.isnull().any(axis=1) #returns T/F
df.loc[mask, :]

#Selecting columns
df.head()
df.columns
df=df[['commodity']]

#Change name to lowercase
df.columns=df.columns.str.lower()

#drop na
df
df.dropna(axis=0, how='any')

#Filter
df=df[df['commodity']=='Gold']

#add boolean column
df['TF']=df['commodity']=='Gold'

#Calculating mean
df1['c'].mean()
df1['c'].sum()

#Groupby
group=df.groupby('commodity')
group.size()
group.mean()
group.agg(['size', 'mean'])


###DATA STRUCTURES & CONTROL FLOWS #####################################################################################

#File input/output
f=open('simple.txt', 'r')
lines=f.readline()
f.write(s) #write a string to a file
f.close()
with open('simple.txt', 'r') as f: #No need to f.close()
    lines=f.readlines()

#Lists
L=['a', 'b', 'c']
list(map(lambda s: s.upper(), L))
L
L.append(['d', 'e', 'f'])
L
L
L.extend(['e', 'f'])
L
L.insert(2, 'e')
L
lis=[1, 7, 3, 4, 6, 2]
lis.sort() #mutating
lis
sorted(lis) #non mutating

#Set (Unordered collection of no duplicate elements).Mutable.
vowels={'a', 'e', 'i', 'o', 'u'}
fruit=set(['apple', 'orange'])

primes={2, 3, 4, 5, 6}
set(map(lambda x: x*x, primes))

#Dictionaries: unordered, hashable data structures that stores key-value pairs
employee={'sex':'male', 'height':6.1, 'age':38}
employee['age']
employee['city']='New York'
employee
'weight' in employee
employee.keys()
employee.values()
dict([('hola', 32), ('hola2', 44)])
del employee['age']
employee.clear()
del employee
dir(employee)

#Conditionals
def firstelt(L):
    if len(L)==0:
        return None
    else:
        return L[0]

#For loops
words=['a', 'b', 'c']

for word in words:
    print(word)

for i in range(len(words)):
    print(i, words[i])

for i, e in enumerate(words): #returns index and element
    print (i, e)

#List comprehension
[ x*x for x in [1,2,3,4,5,6] if x%2==0]
[ x*x if x%2==0 else x+2 for x in [1, 2, 3, 4, 5, 6]]

#While loops
i=0
while i<10:
    print(i)
    i=i+1

#Break or continue

L=[10, 20, -20, -10, 40, 50]

sum_=0
for x in L:
    if x<0:
        continue #terminates the current iteration
    sum_=sum_ + x
    if sum_>100:
        break #terminates the loop

#Exception handling
raise Exception ("Do not do that!!") #generic error
raise TypeError ("We have a problem") #specific error

def openfile(filename, mode):
    try:
        f=open(filename, mode)
    except:
        print('Error:', filename, 'does not exist')

###NUMPY################################################################################################################

#Ndarray
my_ary=np.array([1,2,3]) #Each element has the same type (homogeneity condition)
my_ary[0]
nested_lst=[[1,2,3], [4,5,6]]
multi_ary=np.array(nested_lst) #Each list is a row
multi_ary

#Arange
np.arange(2,10, 2) #start, end, step
np.arange(10)
list(range(10)) #Equivalent to native range

#Linspace
np.linspace(0,10, 51, dtype='Int64') #start, end, number of elements, type

#Ones/zeros
np.ones(10)
np.zeros(10)
np.ones([2,3]) #A tuple is passed to create a matrix
np.zeros([3,4])

#Subscribing, slicing and updating elelments
x=np.arange(0, 120, 2)
x[20]=66 #reassigning
x[-10] #slect elements from opposite side

#Cahnging Type
x.astype(float)

#2D arrays can be indexed as matrices
multi_ary[0,1]

#See dimensions
multi_ary.shape

#Set and change dimensions
x
x.shape=(2,30)
x.reshape(1,60)

#Arithmetic operation can be done pointwise
#Broadcasting works
#Comparison
#Logical operators (&, |, .logical_and(), .logical_or(), .logical_not())
#Aggregating boolean arrays .all(), .any() equivalent to logical_and and logical_or
#Filtering an array
x[x==1]

#Diagonal
A=np.ones((3,3))
np.diagonal(A)

#Matrix type
x=np.matrix([[3,2],[2,3]])
y=np.matrix([[3,3],[2,2]])
x
x[1,1]
x.shape[0]

#Multiplication DOES NOT work pointwise (as in arrays). DOT PRODUCT (row times a column)
z=x*y
z.shape
z=x.dot(y)

#Transpose (.T). Change to a column vector
z.T

#Identity Matrix (ones in diagonal, rest zeros). Equivalent to number 1.
np.eye(3)

#Inverse. Multiplying it by a matrix returns 1.
x.I

#Functions for random sampling
np.random #random number generator
np.random.seed(1)
np.random.choice(range(366), size=20, replace=True)
np.random.randn(10) #Normal
np.random.rand(10) #Uniform from [0,1]
np.random.randint(2,10,2) #int

#####SCIPY(Hypothesis test)##############################################################################################
########################################################################################################################

#One sample t-test
X=np.random.rand(100)
stats.ttest_1samp(X, 10)

#Two samples t-ttest
Y=np.random.rand(100)
stats.ttest_ind(X, Y)

#ANOVA
Z=np.random.rand(100)
stats.f_oneway(X, Y, Z)

####Data Visualization ################################################################################################
from matplotlib import pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv('https://s3.amazonaws.com/nycdsabt01/movie_metadata.csv')

df.columns.tolist()
pd.set_option('display.max_columns', 50)
df.head()
df.shape
df.describe()

df['language'].value_counts()

langs = ["English" ,"French", "Mandarin"]
cond=df['language'].isin(langs)
ans=df[cond & (df.imdb_score >7)]
ans.shape


#Histogram
plt.rcParams['figure.figsize']=6,6
plt.hist(df['imdb_score'], density=True)
x=plt.hist(df['imdb_score'], bins=20, color="#5ee3ff")
x

import seaborn as sns
#%%
log_budget = np.log10(df['budget'])
log_budget.plot.hist()
plt.xlabel('log of budget')
plt.ylabel('count')
plt.title('Histogram of budget', fontsize=20)
#%%

log_budget.plot(kind='hist')


#Scatterplot

#%%
plt.scatter(df['budget'], df['gross'])
plt.xlabel('Budget')
plt.ylabel('Gross Income')
#%%

#%%
#df.plot.scatter(x='budget', y='gross')
df.plot(kind='scatter',x='budget', y='gross')
plt.xlabel('Budget')
plt.ylabel('Gross Income')
#%%

outliers = df[['gross', 'budget']].dropna()
outliers = outliers.loc[~outliers.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

score_df = df[['gross', 'imdb_score']]
score_df.plot.scatter('gross', 'imdb_score')

#Barplot
plt.figure(figsize=(12,6))
df.groupby('country')['imdb_score'].median().sort_values(ascending=False).plot.bar(color='b')

df[['country','imdb_score']].groupby('country').median().sort_values(ascending=False, by='imdb_score').plot(kind='bar', color='b')

plt.figure(figsize=(12,6))
df.groupby('country')['imdb_score'].median().sort_values(ascending=False).head(10).plot.bar(color='b')


dg = df.groupby('country').agg({'duration':'count','imdb_score':'mean'})
dg[dg.duration>10].sort_values(ascending=False, by='imdb_score').plot(kind='bar',y='imdb_score', color='b')

#Boxplot
#%%
df_score = df[['color', 'imdb_score']]
df_score.boxplot(by='color', column='imdb_score')
plt.ylabel('Imdb Score')
#%%

#Convert data types
str(43)
int('4')
str(3.14)
float('3.14')

#String formatting
# % formatting
'hello %s' % name
'hello %.2f' % number

#String formatting
'hello {0}.You are{1}'.format(name, age) #You can reference the index of the variable or the name
'hello {name}.You are{age}'.format(name=name, age=age)

#f-string
f'Hello {name}.You are {age}'
