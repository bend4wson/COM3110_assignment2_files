# In order to run this code:

### First install the necessary toolkits. This can be done using the commands:

```python
pip install nltk
pip install pandas
pip install seaborn
pip install matplotlib
#Note - pip3 may be necessary instead
```

After this, you can install the necessary dependencies using:

```python
> python #Note - python3 may instead be necessary
>> import nltk
>> nltk.download('stopwords')
>> nltk.download('wordnet')
>> nltk.download('omw-1.4')
>> nltk.download('averaged_perceptron_tagger')
>> nltk.download('punkt')
>> exit()
```

In order to run the sentiment analyser, use the command:

```python
python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 5
#Note - May have to use python3 instead
```

Extra commands you may want to use:

```python
-output_files #Allows you to output your predicted values to .tsv files
-confusion_matrix #Allows you to output a confusion matrix using the predicted and actual values
-features all_words #Allows you to use all words instead of features
-features features #Allows you to use feature selection within your data set
-classes 3 #Allows you to use reduce your classes to 3 when classifying your reviews (negative, neutral and positive) as opposed to 5 classes (negative, somewhat negative, neutral, somewhat positive and positive)
```

For example:

```python
python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 3 -output_files -confusion_matrix -features all_words
```
