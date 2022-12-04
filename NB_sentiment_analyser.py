# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse

#For importing the tsv files
import pandas as pd
import numpy

#For the stoplist
#import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#For removing punctuation
import string


"""
IMPORTANT, modify this part with your details
"""
USER_ID = "aca19bcd" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    
    """

    
    trainingData = getFile('moviereviews/train.tsv')
    trainingData = removePunctuation(trainingData)
    trainingData = implementStoplist(trainingData)

    devData = getFile('moviereviews/dev.tsv')
    devData = removePunctuation(devData)
    devData = implementStoplist(devData)

    testData = getFile('moviereviews/test.tsv')
    testData = removePunctuation(testData)
    testData = implementStoplist(testData)

    if number_classes == 5:
        featureSentimentDict, sentimentFreqs = predictSentiment(trainingData, number_classes)
        predictedDevData = calculateSentimentVals(devData, featureSentimentDict, sentimentFreqs, number_classes)
        predictedTestData = calculateSentimentVals(testData, featureSentimentDict, sentimentFreqs, number_classes)

    elif number_classes == 3:
        #Note - Do trainingData & devData  still exist, or do deepcopies have to be made??
        trainingData = reduceClasses(trainingData)
        devData = reduceClasses(devData)

        featureSentimentDict, sentimentFreqs = predictSentiment(trainingData, number_classes)
        predictedDevData = calculateSentimentVals(devData, featureSentimentDict, sentimentFreqs, number_classes)
        predictedTestData = calculateSentimentVals(testData, featureSentimentDict, sentimentFreqs, number_classes)
    else:
        print("Error, enter 3 or 5 classes")


    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = 0
    

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

    if output_files == True and number_classes == 3:
        print("Saving to files")
        predictedDevData.to_csv('dev_predictions_3classes_aca19bcd.tsv', sep="\t")
        predictedTestData.to_csv('test_predictions_3classes_aca19bcd.tsv', sep="\t")
    elif output_files == True and number_classes == 5:
        print("Saving to files")
        predictedDevData.to_csv('dev_predictions_5classes_aca19bcd.tsv', sep="\t")
        predictedTestData.to_csv('test_predictions_5classes_aca19bcd.tsv', sep="\t")

def predictSentiment(data, numClasses):
    priorProbabilities = computePriorProbabilities(data)
    # print(priorProbabilities)
    # data = removePunctuation(data)
    # data = implementStoplist(data)
    # print("+++++ " + str(trainingData))

    # trainingData = splitUpSentences(trainingData)

    # print(str(trainingData) + " TRAINING DATA HERE")
    #print(str(trainingData.to_dict('dict')))

    #Returns e.g: {"cat": {0:1,1:0,2:0,3:4,4:0}, etc}
    featureSentimentDict = createSentimentDict(data, numClasses)
    #returns a dict of sentiment freqs {0:120, 1:.., 2:.., etc}
    sentimentFreqs = calculateSentimentFreq(featureSentimentDict)
    return featureSentimentDict, sentimentFreqs

def reduceClasses(data):
    for sentence in data:
        if sentence[2] == 1:
            sentence[2] = 0
        elif sentence[2] == 2:
            sentence[2] = 1
        elif sentence[2] == 3 or sentence[2] == 4:
            sentence[2] = 2
    return data

def getFile(fileLocation):
    data=pd.read_csv(fileLocation,sep='\t').to_numpy()
    #print("*******\n" + str(data.to_numpy()[0][0]) + "\n" + str(data.to_numpy()[0][1]) + "\n" + str(data.to_numpy()[0][2]) + "\n******")
    return data

#Calculates prior probability of a sentiment using sentences with sentiment (not features)
def computePriorProbabilities(trainingData):
    priorProbabilities = []
    totalCount = len(trainingData)
    for i in range(0, 5):
        sentimentCount = 0
        for sentence in trainingData:
            if sentence[2] == i:
                sentimentCount += 1
        priorProbabilities.append(sentimentCount/totalCount)
    return priorProbabilities

# #Done this way because it makes it easier to implement stoplisting, lowercase, etc vs splitting up words
# #whilst creating the sentiment dictionary
# def splitUpSentences(data):
#     word = ""
#     for j in range(0, len(data)): #sentence in trainingData:
#         sentenceList = []
#         for i in data[j][1]: #sentence[1]:
#             if i != " ":
#                 word += i
#             else:
#                 #NOTE - Can add word individually to the sentiment dictionary here in order to decrease the 
#                 #       amount of time the program takes to run.
#                 #sentimentDict = addWordToSentimentDict(word)
#                 sentenceList.append(word)
#                 word = ""
#         data[j][1] = sentenceList
#         # trainingData[2] = sentenceList
#     return data

def removePunctuation(data):
    for sentence in data:
        #Removes punctuation by creating a version of the string with the punctuation substituted
        sentence[1] = sentence[1].translate(str.maketrans('', '', string.punctuation))
    return data

#Function should implement a stoplist and remove any of the words which aren't relevant (e.g: 'it', 'and')
#Should then return the training data where trainingdata[1] consists of only the features     
def implementStoplist(data):
    for sentence in data:
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(sentence[1])
        wordsFiltered = []

        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)
        sentence[1] = wordsFiltered
    return data

def createSentimentDict(trainingData, numClasses):
    sentimentDict = dict(dict())
    for sentence in trainingData:
        for i in sentence[1]:
            if i not in sentimentDict:
                #The new dictionary added when a new word is found in the training data
                #(Containing each sentiment and word frequency of the sentiment)
                if numClasses == 5:
                    sentimentDict[i] = {0:0, 1:0, 2:0, 3:0, 4:0}
                elif numClasses == 3:
                    sentimentDict[i] = {0:0, 1:0, 2:0}
                else:
                    print("Error with number of classes")
            sentimentDict[i][sentence[2]] += 1
    # print("\n\n\n\n" + str(sentimentDict))
    return sentimentDict

# #Calculates likelihood of a given sentence
# #Uses 5 sentiment values (0,1,2,3,4)
# def calculateLikelihood(trainingData, sentimentDict):
#     sentimentFrequencies = calculateSentimentFreq(sentimentDict)
#     likelihoodVals = dict()


#     return likelihoodValues
#     # total = len(trainingData)
#     # for feature in 



# #Returns the amount of words which have each frequency, and the amount of times each word appears.
# #Returns these values as a list
# def calculateWordAndSentimentFreq(sentimentDict):
#     sentimentFreq = dict()
#     wordFreq = dict()
#     for word in sentimentDict:
#         totalWordFreq = 0
#         for sentiment in word:
#             sentimentFreq[sentiment] += sentimentDict[word][sentiment]
#             totalWordFreq += sentimentDict[word][sentiment]
#         wordFreq[word] = totalWordFreq
#     return sentimentFreq, wordFreq

def calculateSentimentFreq(sentimentDict):
    sentimentFreq = dict()
    for word in sentimentDict:
        for sentiment in sentimentDict[word]:
            if sentiment in sentimentFreq:
                sentimentFreq[sentiment] += sentimentDict[word][sentiment]
            else:
                sentimentFreq[sentiment] = sentimentDict[word][sentiment] #testKeys[int(sentiment)]]
    return sentimentFreq

def calculateSentimentVals(data, sentimentDict, sentimentFreqs, numClasses):
    sentimentVals = []
    sentimentOutput = pd.DataFrame(columns=['SentenceId', 'Sentiment'])
    for sentence in data:
        # print(str(sentence))
        sentimentChoice, sentimentVal = decideSentiment(sentence[1], sentimentDict, sentimentFreqs, numClasses)
        numpy.append(sentence, sentimentChoice)
        # print("Sentence: " + str(sentence))
        # print("sentence 2: " + str(numpy.append(sentence, sentimentChoice)))
        newSentiment = pd.DataFrame({'SentenceId':[sentence[0]], 'Sentiment':[sentimentChoice]})
        sentimentOutput = pd.concat([newSentiment, sentimentOutput.loc[:]]).reset_index(drop=True)
        #sentimentVals.append((sentence[0], sentimentChoice))

        #Adding the new sentiment values to the pandas dataframe
        if len(sentence) > 2:
            sentence[2] = sentimentChoice
    #print(str(sentimentOutput))
    return sentimentOutput
        # print("choice: " + str(sentimentChoice))

#Takes in a list of words (a sentence), uses the likelihood values (calculated using sentimentDict/sentimentFreqs from training data)
#to calculate the score for each sentiment, returns the highest score.
#NOTE - This doesn't take into account smoothing/values that aren't in the training data, leading to 0 vals for some sentiments
def decideSentiment(wordList, sentimentDict, sentimentFreqs, numClasses):
    sentimentLikelihoods = dict()
    # print(str(wordList) + "&&&")
    for word in wordList:
        #i = the sentiment (here would be 0,1,2,3,4)
        if word in sentimentDict:
            for i in range(0, len(sentimentDict[word])):
                if i in sentimentLikelihoods:
                    sentimentLikelihoods[i] *= (sentimentDict[word][i]/sentimentFreqs[i])
                else:
                    sentimentLikelihoods[i] = (sentimentDict[word][i]/sentimentFreqs[i])
    #Account for sentiments without any words
    for j in range(0, 4):
        if j not in sentimentLikelihoods:
            sentimentLikelihoods[j] = 0
    
    #Finding the max sentiment likelihood value:
    maxSentimentVal = 0
    bestSentiment = 0
    for k in range(0, len(sentimentLikelihoods)):
        if sentimentLikelihoods[k] > maxSentimentVal:
            maxSentimentVal = sentimentLikelihoods[k]
            bestSentiment = k
    if maxSentimentVal == 0:
        return 2, maxSentimentVal
    else:
        return bestSentiment, maxSentimentVal



            

if __name__ == "__main__":
    main()
