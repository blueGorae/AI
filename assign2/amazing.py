#!/usr/bin/python

import graderUtil
import util
import time
from util import *

grader = graderUtil.Grader()
submission = grader.load('submission')

grader.addBasicPart('writeupValid', lambda : grader.requireIsValidPdf('sentiment.pdf'))

############################################################
# Problem 1: warmup
############################################################

grader.addManualPart('1a', 2) # simulate SGD
grader.addManualPart('1b', 2) # create small dataset

############################################################
# Problem 2: predicting movie ratings
############################################################

grader.addManualPart('2a', 2) # loss
grader.addManualPart('2b', 3) # gradient
grader.addManualPart('2c', 3) # smallest magnitude
grader.addManualPart('2d', 3) # largest magnitude
grader.addManualPart('2e', 3) # linear regression

############################################################
# Problem 3: sentiment classification
############################################################

### 3a

# Basic sanity check for feature extraction

### 3b


def test3b2():
    trainExamples = readExamples('polarity.train')
    devExamples = readExamples('polarity.dev')
    featureExtractor = submission.extractCharacterFeatures(10)
    weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    devError = evaluatePredictor(devExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print "Official: train error = %s, dev error = %s" % (trainError, devError)
    grader.requireIsLessThan(0.04, trainError)
    grader.requireIsLessThan(0.30, devError)
grader.addBasicPart('3b-2-basic', test3b2, maxPoints=2, maxSeconds=100, description="test classifier on real polarity dev dataset")



grader.grade()

