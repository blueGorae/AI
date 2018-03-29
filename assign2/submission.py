#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    v=collections.defaultdict(lambda: 0)
    x=x.split()
    for i in range(len(x)):
        v[x[i]]=v[x[i]]+1;
    return v
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = collections.defaultdict(lambda :0) # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for i in range(numIters):
        for j in range(len(trainExamples)):
            increment(weights, eta*((1-(dotProduct( featureExtractor(trainExamples[j][0]),weights) * (trainExamples[j][1]))>0)*trainExamples[j][1]), featureExtractor(trainExamples[j][0]))

    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        example=random.choice(weights.keys())
        phi=extractWordFeatures(example)
        if dotProduct(phi, weights)>= 0:
            y=1
        else:
            y=-1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        v=collections.defaultdict(lambda:0)
        s=""
        w=""
        x=x.split()
        for i in x:
            s=s+i
        for i in range(len(s)-(n-1)):
            w=s[i:i+n]
            # for j in range(i, i+n):
            #     w=w+s[j]
            v[w]+=1
            w=""
        return v
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    def mean(l):
        v=collections.defaultdict(lambda:0)
        for i in range(len(l)):
            increment(v, 1, l[i])
        for k in v:
            v[k]=v[k]/float(len(l))
        return v
    def distance(v1, v2):
        increment(v1, -1, v2)
        v=dotProduct(v1, v1)
        increment(v1, 1, v2)
        return v
    def distance_for_list(v1, l):
        result=[]
        for i in range(len(l)):
            result.append(distance(v1, l[i]))
        return result

    initial_junk=len(examples)/K
    last=initial_junk*K
    pre_centroids=[collections.defaultdict(lambda : 0)]*K
    centroids=[collections.defaultdict(lambda : 0)]*K
    assignments=[None]*len(examples)
    n=0
    for i in range(0, len(examples), initial_junk):
        if i==last-initial_junk:
            centroids[n]=mean(examples[last-initial_junk:])
            break
        else:
            centroids[n]=mean(examples[i:i+initial_junk])
            n=n+1
    pre_centroids=centroids
    centroids_set=[[] for _ in range(K)]
    for _ in range(maxIters):
        for i in range(len(examples)):
            l=distance_for_list(examples[i], centroids)
            assignments[i]=l.index(min(l))
            centroids_set[assignments[i]].append(i)
        pre_centroids=centroids
        for i in range(K):
            temp=[]
            if centroids_set[i]==[]:
                centroids[i]=collections.defaultdict(lambda:0)
            else:
                for j in range(len(centroids_set[i])):
                    num=centroids_set[i][j]
                    temp.append(examples[num])
                centroids[i]=mean(temp)
        if centroids==pre_centroids:
            break
        centroids_set=[[]]*K
    loss=0
    for i in range(len(examples)):
        loss=loss+distance(examples[i],centroids[assignments[i]])

    return centroids, assignments, loss







    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    # END_YOUR_CODE
