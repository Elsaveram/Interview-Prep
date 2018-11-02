import pandas as pd
import numpy as np

#Function to switch the ith and the jth items in a List
def switch_item(L, i, j):
    L[i], L[j]= L[j], L[i]
    print(L)

my_list=['first', 'second', 'third', 'fourth']
switch_item(my_list, 1, 2)

#Given a dictionary modify the inventory
inventory={'pumpkin':20, 'fruit':['apple', 'pear'], 'vegetables':['potato', 'onion', 'lettuce']}
inventory['meat']=['beef', 'chicken', 'pork']
sorted(inventory['vegetables'])
inventory['vegetables'].sort()
inventory['pumpkin']=inventory['pumpkin'] + 5
inventory

#Define the following functions using conditionals:
#choose(response, choice1, choice2) returns choice1 if response is the string 'y' or 'yes', and choice2 otherwise.
#leap_year(y) returns true if y is divisible by 4, except if it is divisible by 100; but it is still true if y is divisible by 400. Thus, 1940 is a leap year, 1900 isn’t, and 2000 is.
#Use filter to define a function leap_years that selects from a list of numbers those that represent leap years.

def choose(response, choice1, choice2):
    if response in ['yes', 'y']:
        return choice1
    else:
        return choice2

def leap_year(y):
    if y % 4 ==0:
        if y % 400 ==0:
            return True
        elif y % 100 ==0:
            return False
        return True
    else:
        return False

leap_year(1940)
leap_year(1900)
leap_year(2000)

def leap_year_ls(yearList):
    leap_year_List=[]
    for year in yearList:
        if y % 4 ==0:
            if y % 400 ==0:
                leap_year_List.append(y)
            elif y % 100 ==0:
                continue
            leap_year_List.append(y)
        else:
            continue
    return leap_year_List

def leap_year(yearList):
    return filter(leap_year, yearList)

#Write list comprehensions to create the following lists:
#The square roots of the numbers in [1, 4, 9, 16]. (Recall that math.sqrt is the square root function.)
#The even numbers in a numeric list L. Define several lists L to test your list comprehension. Hint (n is even if and only if n % 2 == 0.)

from math import sqrt
print([sqrt(x) for x in [1, 4, 9, 16]])

list(range(10))
L=[x**2 for x in range(10)]
[x for x in L if x %2 ==0]

#Calculate the sum of integers that can be divided by 7 and less than 100. In the following example code, we use while True, which means the while loop will keep running until you break it.

sum_=0
i=0
while True:
    i=i+1
    if i % 7 !=0:
        continue
    if i>=100:
        break
    sum_=sum_+i

print(sum_)

#Create a 4 by 5 five matrix with all the entries equal to 8.
#Create an array corresponding to the matrix below: np.array([[9,8,7],[6,5,4]]

8*(np.ones(20).reshape(4,5))
np.arange(9, 3, -1).reshape(2, 3)

#If we randomly choose 20 people, what is the probability that some of them share the same birth day?
num_people=20
num_simu=int(1e4)
Bool = np.zeros(num_simu)
for i in range(num_simu):
    test=np.random.choice(range(366), size=20, replace=True)
    Bool[i]=(len(set(test)) != num_people)
np.mean(Bool)

#Coding challenges:

def oddNumbers(l, r):
    return [number for number in range(l, r+1) if number%2 != 0]

oddNumbers(3, 9)

#HackerRank

#Repeated string
import math

def repeatedString(s, n):
    occourences=s.count('a')
    total_occurences=math.floor(n/len(s)) * occourences + s[:(n %len(s))].count('a')
    return total_occurences

repeatedString('a', 1000000000000)


#Jumping on the Clouds
def jumpingOnClouds(c):
    i = 0
    jumps = 0
    while( i < len(c)-1 ):
        if i == len(c)-2:
            i = i+1
        elif c[i+2] == 0:
            i = i+2
        else:
            i = i+1
        jumps += 1
    return jumps

jumpingOnClouds([0, 0, 0, 1, 0, 0])

#Counting VAlleys

def countingValleys(n, s):
    altitude=0
    valleys=0
    for i in range(n):
        if s[i]=='D':
            altitude += -1
        elif s[i]=='U':
            altitude += 1
            if altitude == 0:
                valleys += 1
    return valleys

countingValleys(8, ['U', 'D', 'D', 'D', 'U', 'D', 'U', 'U'])

#Sock merchant
import math
def sockMerchant(n, ar):
    pairs=0
    match = 0
    for element in set(ar):
        for u in range(n):
            if element==ar[u]:
                match += 1
        pairs += math.floor(match/2)
        match = 0
    return pairs

def sockMerchant(n, ar):
    pairs = 0
    for number_to_count in set(ar):
        pairs += math.floor(ar.count(number_to_count)/2)
    return pairs

def sockMerchant(n, ar):
    matches = []
    pairs = 0
    for number in ar:
        if number not in matches:
            matches.append(number)
        else:
            matches.remove(number)
            pairs += 1
    return pairs

sockMerchant(9, [10, 20, 20, 10, 10, 30, 50, 10, 20])
nested_lst=[[1,2,3], [4,5,6]]
multi_ary=np.array(nested_lst)
(

#2D Array-DS
def hourglassSum(arr):
    hourglass=[]
    for i in range(4):
        hourglass.append(arr[i][0]+arr[i][1]+arr[i][2]+ arr[i+1][1]+ arr[i+2][0]+ arr[i+2][1] +arr[i+2][2])
        hourglass.append(arr[i][1]+arr[i][2]+arr[i][3]+ arr[i+1][2]+ arr[i+2][1]+ arr[i+2][2] +arr[i+2][3])
        hourglass.append(arr[i][2]+arr[i][3]+arr[i][4]+ arr[i+1][3]+ arr[i+2][2]+ arr[i+2][3] +arr[i+2][4])
        hourglass.append(arr[i][3]+arr[i][4]+arr[i][5]+ arr[i+1][4]+ arr[i+2][3]+ arr[i+2][4] +arr[i+2][5])
    return max(hourglass)


def hourglassSum(arr):
    hourglass=[]
    for i in range(4):
        for j in range(4):
            hourglass.append( sum(arr[i][0+j:2+j+1]) + arr[i+1][1+j] + sum(arr[i+2][0+j:2+j+1]) )
    return max(hourglass)

hourglassSum(([[1, 1, 1, 0, 0, 0],[0, 1, 0, 0, 0, 0],[1, 1, 1, 0, 0, 0],[0, 0, 2, 4, 4, 0],[0, 0, 0, 2, 0, 0],[0, 0, 1, 2, 4, 0]]))

#Arrays:Left Rotation
def rotLeft(a, d):
    L=[]
    L.extend(a[d:])
    L.extend(a[:d])
    return L
rotLeft([1,2,3,4,5], 4)
rotLeft([41, 73, 89, 7, 10, 1, 59, 58, 84, 77, 77, 97, 58, 1, 86, 58, 26, 10, 86, 51], 10)

#New Year's Chaos

def minimumBribes(skip_array):
    skippers = []
    array_len = len(skip_array)
    for i in range(1,array_len+1):
        skips = skip_array.index(i)
        skippers += skip_array[:skips]
        skip_array.remove(i)

    for i in set(skippers):
        if skippers.count(i) > 2:
          print("Too chaotic")
          return

    print(len(skippers))

def minimumBribes(q):
    bribes = 0
    for i in range(len(q)-1,-1,-1):
        if q[i] - (i + 1) > 2:
            print('Too chaotic')
            return
        for j in range(max(0, q[i] - 2), i):
            if q[j] > q[i]:
                bribes+=1
    print(bribes)

#String Manipulation
#Making Anagrams
def makeAnagram(a, b):
    deletions=0
    a=list(a)
    b=list(b)
    for letter in list(set(list(a+b))):
        a_count=a.count(letter)
        b_count=b.count(letter)
        if abs(a_count-b_count) != 0:
            deletions += abs(a_count-b_count)
    return deletions

#Alternating Characters
def alternatingCharacters(s):
    deletions=0
    for i in range(len(s)-1):
        if s[i]==s[i+1]:
            deletions += 1
    return deletions

#Bubble Sort: It compares adjacent elements and exchanges those that are out of order.
def bubbbleSort(myList):
    for i in range (0, len(myList)-1):
        for j in range(0, len(myList)-1-i):
            if myList[j] > myList[j+1]:
                myList[j], myList[j+1] = myList[j+1], myList[j]
    return myList

bubbbleSort([3, 2, 1, 4])

def countSwaps(a):
    numSwaps=0
    for i in range (0, len(a)-1):
        for j in range(0, len(a)-1-i):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                numSwaps += 1
    print ('Array is sorted in {} swaps.'.format(numSwaps))
    print ('First Element: {}'.format(a[0]))
    print ('Last Element: {}'.format(a[-1]))

countSwaps([3,2,1])

#Mark and Toys
def maximumToys(prices, k):
    cost=0
    toys=[]
    prices.sort()
    for toy in prices:
        cost += toy
        toys.append(toy)
        if cost > k:
            break
    return len(toys)-1


def maximumToys(prices, budget):
    cost=0
    i=0
    prices.sort()
    while i < len(prices) and cost + prices[i] < budget:
        cost += prices[i]
        i += 1

    return i

maximumToys([1,2,3,4], 7)

#Ransome Note
def checkMagazine(magazine, note):
    my_dict={}
    for word in list(set(magazine.split())):
        my_dict[word]=magazine.count(word)
    for w in note.split():
        if w in my_dict.keys():
            my_dict[w] -= 1
            if my_dict[w] <0:
                print('No')
                return
        else:
            print ('No')
            return
    print ('Yes')

checkMagazine('ive got a lovely bunch of coconuts', 'ive got some coconuts')
