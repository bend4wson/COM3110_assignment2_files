# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse

#For importing the tsv files
import pandas as pd
import numpy

#For creating a heatmap of confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

#For the stoplist
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#For stemming
from nltk.stem import PorterStemmer

#For Lemmatizing
from nltk.stem import WordNetLemmatizer

#For feature selection, POS Tagging:
import nltk
from nltk import pos_tag
from nltk import RegexpParser

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
    

    trainingData = getFile('moviereviews/train.tsv')
    devData = getFile('moviereviews/dev.tsv')
    testData = getFile('moviereviews/test.tsv')

    trainingData = preProcess(trainingData)
    devData = preProcess(devData)
    testData = preProcess(testData)

    #Is features are being used
    if features != "all_words":
        for i in range(0, len(trainingData)):
            #Keep the words which are either adjectives (JJ) or adverbs (RB)
            trainingData[i][1] = [token for token, pos in pos_tag(trainingData[i][1]) if pos.startswith("JJ") or pos.startswith("RB")] 
        for j in range(0, len(devData)):
            devData[j][1] = [token for token, pos in pos_tag(devData[j][1]) if pos.startswith("JJ") or pos.startswith("RB")]
        for k in range(0, len(testData)):
            testData[k][1] = [token for token, pos in pos_tag(testData[k][1]) if pos.startswith("JJ") or pos.startswith("RB")]

    if number_classes == 3:
        trainingData = reduceClasses(trainingData)
        devData = reduceClasses(devData)

    featureSentimentDict, sentimentFreqs = predictSentiment(trainingData, number_classes)
    predictedDevData = calculateSentimentVals(devData, featureSentimentDict, sentimentFreqs, number_classes)
    predictedTestData = calculateSentimentVals(testData, featureSentimentDict, sentimentFreqs, number_classes)

    devSents = []
    devRealSents = []
    if number_classes == 3:
        devReal = reduceClasses(preProcess(getFile('moviereviews/dev.tsv')))
    else:
        devReal = preProcess(getFile('moviereviews/dev.tsv'))
    #Create a new, non-processed data set from the movie reviews to compare to the pre-processed one
    for i in range(0, len(devData)):
        devSents.append(devData[i][2])
        devRealSents.append(devReal[i][2])


    if number_classes == 5:
        confusionMatrix = calculateConfusionMatrices(devData, preProcess(getFile('moviereviews/dev.tsv')), number_classes)
    elif number_classes == 3:
        confusionMatrix = calculateConfusionMatrices(devData, reduceClasses(preProcess(getFile('moviereviews/dev.tsv'))), number_classes)
    else:
        print("Error")

    if confusion_matrix == True:
        confusionMatrixHM = sns.heatmap(confusionMatrix, annot=True, cmap='plasma')
        confusionMatrixHM.set(xlabel='Predicted Sentiment', ylabel='Actual Sentiment')
        plt.show()


    matrixVals = calculateMatrixVals(confusionMatrix)
    precisions = calculateClassPrecisions(matrixVals)
    recalls = calculateClassRecalls(matrixVals)
    f1s = calculateF1Vals(matrixVals)
    f1_score = calculateMacroF1(f1s)
    


    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

    if output_files == True and number_classes == 3:
        print("Saving to files")
        saveData(predictedDevData, 'predictions/dev_predictions_3classes_aca19bcd.tsv')
        saveData(predictedTestData, 'predictions/test_predictions_3classes_aca19bcd.tsv')
    elif output_files == True and number_classes == 5:
        print("Saving to files")
        saveData(predictedDevData, 'predictions/dev_predictions_5classes_aca19bcd.tsv')
        saveData(predictedTestData, 'predictions/test_predictions_5classes_aca19bcd.tsv')

"""This function saves the predicted classes into .tsv files"""
def saveData(predictions, fileName):
    file = open(fileName, 'w')
    file.write(f'SentenceID\tSentiment\n')
    for prediction in predictions:
        file.write(str(prediction[0]) + "\t" + str(prediction[1]) + "\n")
    file.close()

"""This function calculates the precision values for each class using the TP, TN, FP, FN values from the confusion matrix"""
def calculateClassPrecisions(matrixVals):
    #NOTE - Precision = TP / (TP+FP)
    precisionVals = []
    for classVals in matrixVals:
        classArr = matrixVals[classVals]
        precisionVals.append(classArr[0]/(classArr[0]+classArr[2]))
    return precisionVals

"""This function calculates the recall values for each class using the TP, TN, FP, FN values from the confusion matrix"""
def calculateClassRecalls(matrixVals):
    #NOTE - Recall = TP / (TP+FN)
    recallVals = []
    for classVals in matrixVals:
        classArr = matrixVals[classVals]
        recallVals.append(classArr[0]/(classArr[0]+classArr[3]))
    return recallVals

"""This function calculates the macro-F1 values for each class using the TP, TN, FP, FN values from the confusion matrix"""
def calculateF1Vals(matrixVals):
    f1Vals = []
    for classVals in matrixVals:
        classArr = matrixVals[classVals]
        f1Vals.append((2*classArr[0])/(2*classArr[0]+classArr[2]+classArr[3]))
    return f1Vals

"""This function calculates the average Macro-F1 value using the macro-F1 values for each class"""
def calculateMacroF1(f1Vals):
    macroF1 = 0.0
    for i in range(0, len(f1Vals)):
        macroF1 += f1Vals[i]
    macroF1 /= (len(f1Vals))
    return macroF1

#Calculates TP, TN, FP and FN values
"""This function calculates True Positive, True Negative, False Positive & False Negative values for each sentiment"""
def calculateMatrixVals(confusionMatrix):
    if len(confusionMatrix) == 3:
    #This matrix will contain the TP, TN, FP, FN values for each class calculated using the confusion matrix
        matrixValues = {0:[0, 0, 0, 0], 1:[0, 0, 0, 0], 2:[0, 0, 0, 0]}
    else:
        matrixValues = {0:[0, 0, 0, 0], 1:[0, 0, 0, 0], 2:[0, 0, 0, 0], 3:[0, 0, 0, 0], 4:[0, 0, 0, 0]}

    for className in range(0, len(matrixValues)):
        #Find the amount of true positives
        tp = confusionMatrix[className][className]
        
        matrixValues[className][0] += tp

        #Calculate the amount of true negatives
        tn = 0
        for j in range(0, len(confusionMatrix)):
            if j != className:
                for k in range(0, len(confusionMatrix[j])):
                    if k != className:
                        tn += confusionMatrix[j][k]

        #Calculate the amount of false positives
        fp = 0
        for l in range(0, len(confusionMatrix)):
            if l != className:
                fp += confusionMatrix[l][className]

        #Calculate the amount of false negatives
        fn = 0
        for i in range(0, len(confusionMatrix[className])):
            if i != className:
                fn += confusionMatrix[className][i]

        matrixValues[className] = [tp, tn, fp, fn]

    return matrixValues


"""This function processes the data which has been read in from the review files"""
def preProcess(data):
    # data = removeAllPunctuation(data)
    data = implementStoplist(data)
    data = implementLowercasing(data)
    data = removeSomePunctuation(data)
    data = implementStemming(data)
    # data = implementLemmatization(data)
    return data

"""This function removes patterns of punctuation from each sentence in the data structure given"""
def removeSomePunctuation(data):
    for i in range(0, len(data)):
        #Removes any punctuation that is defined as its own word in the data structure (e.g: ["cat", ",", "dog"], "," would be removed)
        data[i][1] = [word for word in data[i][1] if word != '.' and word != '.' and word != "'" and word != "-" and word != "//" and word != "*" and word != "--" and word != "..." and word != "-rrb-" and word != "-lrb-" and word != ";"]
        for j in range(0, len(data[i][1])):
            word = data[i][1][j]
            #Remove commas or full stops from each word in the data structure
            word.replace(",", "")
            word.replace(".","")

            # Some other replacements which were tried:
            # word.replace("'", "")
            # word.replace("-", "")
            # word.replace("//", "")
            # word.replace("*", "")
            # word.replace(",", "")
            # word.replace("--", "")
            # word.replace("...", "")
            # word.replace("-rrb-", "")
            # word.replace("-lrb-", "")
            # word.replace(";", "")
    return data

"""This function makes each letter in each word in the data structure lowercase"""
def implementLowercasing(data):
    for i in range(0, len(data)):
            for j in range(0, len(data[i][1])):
                data[i][1][j] = data[i][1][j].lower()
    return data

"""This function lemmatizes each word in the data structure"""
def implementLemmatization(data):
    wordnetLemmatizer = WordNetLemmatizer()
    for i in range(0, len(data)):
        for j in range(0, len(data[i][1])):
            data[i][1][j] = wordnetLemmatizer.lemmatize(data[i][1][j])
    return data

"""This function stems each word in the data structure"""
def implementStemming(data):
    ps = PorterStemmer()
    for i in range(0, len(data)):
        for j in range(0, len(data[i][1])):
            data[i][1][j] = ps.stem(data[i][1][j])
    return data

"""This function calculates the confusion matrix using the predicted sentiments and real sentiments"""
def calculateConfusionMatrices(predictedDevData, realDevData, numClasses):
    if numClasses == 3:
        confusionMatrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    elif numClasses == 5:
        confusionMatrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    else:
        print("Error with numClasses in calculateConfusionMatrices")
    #Add 1 to the correct cell of the confusion matrix in accordance to the predicted and real sentiment values
    for i in range(0, len(realDevData)):
        confusionMatrix[predictedDevData[i][2]][realDevData[i][2]] += 1
    return confusionMatrix

"""This function calculates the sentiments for each word across the dataset and puts them into a dictionary, 
    as well as calculating the frequency of each sentiment (on a word-level) and puts them into a list"""
def predictSentiment(data, numClasses):
    priorProbabilities = computePriorProbabilities(data)
    #Returns e.g: {"cat": {0:1,1:0,2:0,3:4,4:0}, etc}
    featureSentimentDict = createSentimentDict(data, numClasses)
    #returns a dict of sentiment freqs {0:120, 1:.., 2:.., etc}
    sentimentFreqs = calculateSentimentFreq(featureSentimentDict)
    return featureSentimentDict, sentimentFreqs

"""This function reduces the data set from 5 classes down to 3 classes"""
def reduceClasses(data):
    for sentence in data:
        if sentence[2] == 1:
            sentence[2] = 0
        elif sentence[2] == 2:
            sentence[2] = 1
        elif sentence[2] == 3 or sentence[2] == 4:
            sentence[2] = 2
    return data

"""This function reads in the file dataset and turns it into a pandas dataframe"""
def getFile(fileLocation):
    data=pd.read_csv(fileLocation,sep='\t').to_numpy()
    return data

"""This function calculates prior probability of a sentiment using sentences with sentiment (not features)"""
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

"""This function removes all punctuation from each sentence in the dataset"""
def removeAllPunctuation(data):
    for sentence in data:
        #Removes punctuation by creating a version of the string with the punctuation substituted for nothing
        sentence[1] = sentence[1].translate(str.maketrans('', '', string.punctuation))
    return data

"""This function implements a stoplist and removes any of the words which aren't relevant (e.g: 'it', 'and').
    It then return the training data where trainingdata[1] consists of only the features"""
def implementStoplist(data):
    for sentence in data:
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(sentence[1])
        wordsFiltered = []

        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)
            # wordsFiltered.append(w)
        sentence[1] = wordsFiltered
    return data

"""This function calculates the amount of times each sentiment of each word appears in the data structure,
    and returns the result as a nested dictionary"""
def createSentimentDict(trainingData, numClasses):
    sentimentDict = dict(dict())
    for sentence in trainingData:
        for i in sentence[1]:
            if i not in sentimentDict:
                #A new dictionary is added when a new word is found in the training data
                #(Containing each sentiment and word frequency for the sentiment)
                if numClasses == 5:
                    sentimentDict[i] = {0:0, 1:0, 2:0, 3:0, 4:0}
                elif numClasses == 3:
                    sentimentDict[i] = {0:0, 1:0, 2:0}
                else:
                    print("Error with number of classes")
            sentimentDict[i][sentence[2]] += 1
    return sentimentDict

"""This function calculates the amount of times each sentiment appears (per word as opposed to per sentence)
    using the given sentiment dictionary"""
def calculateSentimentFreq(sentimentDict):
    sentimentFreq = dict()
    for word in sentimentDict:
        for sentiment in sentimentDict[word]:
            if sentiment in sentimentFreq:
                sentimentFreq[sentiment] += sentimentDict[word][sentiment]
            else:
                sentimentFreq[sentiment] = sentimentDict[word][sentiment] #testKeys[int(sentiment)]]
    return sentimentFreq

"""This function predicts the sentiment values for each sentence in the given data, using the sentiment dictionary
    and the sentiment frequencies from the training data"""
def calculateSentimentVals(data, sentimentDict, sentimentFreqs, numClasses):
    sentimentOutput = []
    for sentence in data:
        #Creating a list containing sentiment predictions for writing to a file
        sentimentChoice, sentimentVal = decideSentiment(sentence[1], sentimentDict, sentimentFreqs, numClasses)
        sentimentOutput.append([sentence[0], sentimentChoice])
        if len(sentence) > 2:
            sentence[2] = sentimentChoice
    return sentimentOutput

"""This function calculates the likelihood values for each sentiment and returns the highest score"""
def decideSentiment(wordList, sentimentDict, sentimentFreqs, numClasses):
    sentimentLikelihoods = dict()
    for word in wordList:
        #\/ (If the word was in training data)
        if word in sentimentDict:
            #For each class for the word
            for i in range(0, len(sentimentDict[word])):
                #Note - This step implements laplace smoothing
                if i in sentimentLikelihoods:
                    #Multiplies the sentiment likelihood value by 
                    #(sentiment frequency for the word in the dataset/Amount of times a word appears of that sentiment)
                    #NOTE - ^ Equation doesn't take into account laplace smoothing as used below.
                    sentimentLikelihoods[i] *= ((sentimentDict[word][i]+1)/(sentimentFreqs[i]+len(sentimentDict)))
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
        #Return best sentiment = neutral if there isn't a max sentiment value
        return 2, maxSentimentVal
    else:
        return bestSentiment, maxSentimentVal


if __name__ == "__main__":
    main()