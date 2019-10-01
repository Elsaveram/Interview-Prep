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
    test=np.random.choice(range(366), size=30, replace=True)
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
#Given two strings, check to see if they are anagrams. An anagram is when the two strings can be written using the exact same letters
#1
def anagram(s1,s2):
    w1=list(s1.replace(' ','').lower())
    w2=list(s2.replace(' ','').lower())

    for letter in set(w1+w2):
        c1=w1.count(letter)
        c2=w2.count(letter)
        if abs(c1-c2)!=0:
            return False

    return True
#2
def anagram2(s1,s2):

    s1 = s1.replace(' ','').lower()
    s2 = s2.replace(' ','').lower()

    return sorted(s1) == sorted(s2)
#3 Best O(n)
def anagram3(s1,s2):

    s1 = s1.replace(' ','').lower()
    s2 = s2.replace(' ','').lower()

    #check same number of letters
    if len(s1) != len(s2):
        return False

    count = {}

    for letter in s1:
        if letter in count:
            count[letter] += 1
        else:
            count[letter] = 1
    for letter in s2:
        if letter in count:
            count[letter] -= 1
        else:
            count[letter] =1
    for k in count:
        if count[k] != 0:
            return False
    return True

# Given an integer array, output all the * *unique ** pairs that sum up to a specific value k.

#1 O(n2)
def pair_sum(arr,k):
    l = list(range(len(arr)))
    lis_pairs = []

    for i in l:
        for j in l:
            if i != j:
                if arr[i] + arr [j] == k:
                    lis_pairs.append (tuple(sorted((arr[i],arr[j]))))
    print (set(lis_pairs))
    return len(set(lis_pairs))

#2 O(1) + 2 * O(log n) + O(n)
def pair_sum2(arr,k):
    sor_arr = sorted(arr)
    arr_rem =sorted(arr)
    count_pairs = 0
    pairs = []

    for i in list(range(len(sor_arr))):

        num = sor_arr[i]

        if num <= k/2:

            if k-num in arr_rem[i:]:
                count_pairs += 1
                pairs.append((num,k-num))
                arr_rem.remove(k-num)

    print(pairs)
    return count_pairs

# O(n) Preferred
def pair_sum3(arr, k):

    if len(arr) < 2:
        return "less than 2 elements in array"

    seen = set()
    output = set()

    for num in arr:

        target = k - num

        if target not in seen:
            seen.add(num)

        else:
            output.add( (min(num, target), max(num, target)))

    return len(output)
# Missing element: Consider an array of non-negative integers. A second array is formed by shuffling
# the elements of the first array and deleting a random element. Given these two arrays,
# find which element is missing in the second array.

#1 3*O(n)
def finder2(arr1, arr2):
    output = {}

    for ele in arr1:
        if ele in output:
            output[ele] += 1
        else:
            output[ele] = 1
    for ele in arr2:
        if ele in output:
            output[ele] -= 1
        else:
            output[ele] = 1
    for k in output:
        if output[k] > 0:
            return k
#2 O(nlogn)
def finder3(arr1, arr2):
    arr1.sort()
    arr2.sort()
    for a, b in zip(arr1, arr2):
        if a != b:
            return a
    return arr1[-1]

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

#Two String
def twoStrings(s1, s2):
    my_dict={}
    for letter in list(set(list(s1)))
        my_dict[letter]= s1.count(letter)
    for l in list(s2):
        if l in my_dict.keys():
            return 'YES'
    return 'NO'


def twoStrings(s1, s2):
    my_set = set( list(s1) )
    for l in list(s2):
        if l in my_set:
            return 'YES'
    return 'NO'

twoStrings('hi', 'world')

# Minimum Absolute Difference in an Array
def minimumAbsoluteDifference(arr):
    min_diff=[]
    for i in range(len(arr)-1):
        for j in range(i+1, len(arr)):
            min_diff.append(abs(arr[i]-arr[j]))
            if abs(arr[i]-arr[j]
    print(min(min_diff))

def minimumAbsoluteDifference(arr):
    min_diff=[]
    for i in range(len(arr)-1):
        for j in range(i+1, len(arr)):
            min_diff.append(abs(arr[i]-arr[j]))
    return(min(min_diff))

minimumAbsoluteDifference([-59, -36, -13, 1, -53, -92, -2, -96, -54, 75])

def minimumAbsoluteDifference(arr):
    arr.sort()
    min_diff=[]
    for i in range(len(arr)-1):
        min_diff.append(abs(arr[i+1]-arr[i]))
    return(min(min_diff))

def luckBalance(k, contests):
    luck_imp=[]
    luck_not_imp=[]
    for i in range(len(contests)):
        if contests[i][1]==1:
            luck_imp.append(contests[i][0])
        else:
            luck_not_imp.append(contests[i][0])
    luck_imp.sort()
    print(luck_not_imp)
    print(luck_imp)
    if len(luck_imp)<k or len(luck_imp)==k:
        win=0
        final_luck=sum(luck_not_imp)+sum(luck_imp)
    else:
        win=len(luck_imp)-k
    final_luck=sum(luck_not_imp)+sum(luck_imp[:win-1:-1])-sum(luck_imp[:win])
    return final_luck

#Greatest integer whose product with 5 is less than 67.
max([x for x in range(50) if x*5<67])

#Unique letters in a word
def count_letter(word):
    return len(list(set(word)))

def count_letter(word):
    my_unique_letters=[]
    for letter in word:
        if letter not in my_unique_letters:
            my_unique_letters.append(letter)
    return len(my_unique_letters)

count_letter('happy')

#Write a function that returns the common elements between two lists, each element only one time e.g. intersect([3,1,2,1], [1,4,2,2]) = [1,2]
def intersect(L1, L2):
    common=[]
    for element in list(set(L2)):
        if element in list(set(L1)):
            common.append(element)
    return(common)

def intersect(a, b):
    return list(set(a).intersection(set(b)))

#Write a function that returns all of the prime numbers within a given range.

def prime_num(x):
    if x>=2:
        for num in range(2, x):
            if not x%num!=0:
                return False
    else:
        return False
    return True

def get_primes(n):
    my_primes=[]
    for num in range(n+1):
        if prime_num(num):
            my_primes.append(num)
    return my_primes

get_primes(13)


#Write a function which gets the index of a given value in a sorted list:
def ret_index(lis, n):
    my_dict={}
    for i in range(len(lis)):
        my_dict[lis[i]]=i
    return my_dict[n]

lis = [1,2,4,5,11,19,95,116]
ret_index(lis, 4)

def ret_idx(lis, val):
    return lis.index(val)

#If x is a float and n is a integer, define a function to calculate x^n without using the built in Python method.

def my_power(x,n):
    positive=x
    negative=1
    if n==0:
        return 1
    elif n>0:
        for i in range(n-1):
            positive=positive*x
        return positive
    else:
        for i in range(abs(n+1)):
            negative=negative/x
        return negative

my_power(10,-3)

#Write code for all numbers up to 100 and print “fizz” if divisible by 3, “buzz” if divisible by 5, and “fizzbuzz” if divisible by 3 and 5.

for n in range(101):
    if n%3==0 and n%5==0:
        print('fizzbuzz')
    elif n%3==0:
        print('fizz')
    elif n%5==0:
        print('buzz')

#HackerRank

#Python ifelse
if n%2 !=0:
    print('Weird')
elif n in range(2, 6):
    print('Not Weird')
elif n in range(6, 21):
    print('Weird')
elif n>20:
    print('Not Weird')


check = {True: "Not Weird", False: "Weird"}

print(check[
        n%2==0 and (
            n in range(2,6) or
            n > 20)
    ])

#Arithmetic operators:
    print(a+b)
    print(a-b)
    print(a*b)

#Division
print(a//b)
print(a/b)

#Loops
for i in range(N):
    print(i**2)

#Function. Leap year boolean.
def is_leap(year):
    leap=True
    if year%100==0:
        if year%400 !=0:
            leap=False
    elif year%4 !=0:
        leap=False
    return leap

is_leap(1990)


#Print Function
print(int(''.join(map(str, list(range(n+1))))))
print(*list(range(1, n+1)), sep='')

#Solve Me First
def compareTriplets(a, b):
    score_a=0
    score_b=0
    for i in range(len(a)):
        if a[i]>b[i]:
            score_a += 1
        elif a[i]<b[i]:
            score_b += 1
    return (score_a, score_b)

compareTriplets([17, 28, 30], [99,16,8])

#Diagonal Diference

def diagonalDifference(arr):
    d1=0
    d2=0
    n=len(arr)
    for i in range(n):
        for j in range(n):
            if i==j: #Primary diagonal
                d1 += arr[i][j]
            if i==n-j-1: #Secondary diagonal
                d2 += arr[i][j]
    return abs(d1-d2)

diagonalDifference([[11,2,4],[4,5,6],[10, 8, -12]])

def plusMinus(arr):
    pos=0
    neg=0
    zer=0
    total=len(arr)
    for element in arr:
        if element ==0:
            zer += 1
        elif element <0:
            neg += 1
        else:
            pos += 1
    pos=pos/total
    neg=neg/total
    zer=zer/total
    print ('%.6f' %pos)
    print ('%.6f' %neg)
    print ('%.6f' %zer)


plusMinus([-4, 3, -9, 0, 4, 1])

def staircase(n):
    for stairs in range(1, n + 1):
        print (' ' * (n - stairs) + '#' * stairs)

staircase(6)

#Min-Max Sum

def miniMaxSum(arr):
    arr_sum=[]
    for i in arr:
        arr_sum.append(sum(arr)-i)
    print(min(arr_sum), max(arr_sum))

miniMaxSum([1,2,3,4,5])


#BirthdayCakeCandles
def birthdayCakeCandles(ar):
    num_candles=0
    for num in ar:
        if num==max(ar):
            num_candles += 1
    return num_candles

def birthdayCakeCandles(ar):
    return(len([i for i in ar if i==max(ar)]))

def birthdayCakeCandles(ar):
    return len(list(filter(lambda x: x==max(ar), ar)))

def birthdayCakeCandles(ar):
    return ar.count(max(ar))

birthdayCakeCandles([3,2,1,3])

#timeConversion
def timeConversion(s):
