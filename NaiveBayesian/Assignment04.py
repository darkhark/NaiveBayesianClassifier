from sklearn.naive_bayes import CategoricalNB, GaussianNB
import pandas as pd
from sklearn.metrics import precision_score,f1_score, recall_score, accuracy_score

train = {'home': [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
         'top': [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
         'media': [1, 4, 1, 1, 4, 1, 1, 4, 4, 1, 1, 3, 4, 1, 1, 1, 2, 4, 1, 1, 5, 1, 1, 4],
         'Decision': [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]}

test = {'home': [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        'top': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
        'media': [1, 1, 2, 3, 1, 4, 1, 1, 1, 4, 1, 4]}

correctTest = [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]

trainDF = pd.DataFrame(data=train)
testDF = pd.DataFrame(data=test)

"""
Here, CategoricalNB was used because each value is a categorical data type. 
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB
"""
cnb = CategoricalNB()

x_Cat = trainDF.loc[:, trainDF.columns != 'Decision']
y_Cat = trainDF['Decision']
cnb.fit(x_Cat, y_Cat)
predictedCatValues = list(cnb.predict(testDF))

pscore = precision_score(correctTest, predictedCatValues)
f1score = f1_score(correctTest, predictedCatValues)
rscore = recall_score(correctTest, predictedCatValues)
ascore = accuracy_score(correctTest, predictedCatValues)

print("Categorical Values")
print('\tPrecision Score: ', pscore)
print('\tF1 Score: ', f1score)
print('\tRecall Score: ', rscore)
print('\tAccuracy Score: ', ascore)
print("Predicted Game Outcomes:\n\t" + str(predictedCatValues))
print("\t" + str(correctTest))
print("Actual outcomes ^\n")

"""
Gaussian was also used as it appears to be the standard Naive Bayes ran
"""

gnb = GaussianNB()

x_Gau = trainDF.loc[:, trainDF.columns != 'Decision']
y_Gau = trainDF['Decision']
gnb.fit(x_Gau, y_Gau)
predictedGauValues = list(gnb.predict(testDF))

pscore = precision_score(correctTest, predictedGauValues)
f1score = f1_score(correctTest, predictedGauValues)
rscore = recall_score(correctTest, predictedGauValues)
ascore = accuracy_score(correctTest, predictedGauValues)

print("Gaussian Values")
print('\tPrecision Score: ', pscore)
print('\tF1 Score: ', f1score)
print('\tRecall Score: ', rscore)
print('\tAccuracy Score: ', ascore)
print("\nPredicted Game Outcomes:\n\t" + str(predictedGauValues))
print("\t" + str(correctTest))
print("Actual outcomes ^\n")

