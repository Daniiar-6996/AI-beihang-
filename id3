import pandas as pd 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
import matplotlib.pyplot as plt 
import math
import numpy as np
import matplotlib.pyplot as plt

source = pd.read_csv('cpu_shop.csv') #import dataset 
repeated_sc = source.loc[np.repeat(source.index.values, source['Number of people'])].reset_index(drop=True) #transforming dataset
#the first column shows number of customers (not an attribute)
df = repeated_sc.drop(columns=['Number of people'])
features = df[['Age', 'Income', 'Bought before', 'Credit']] # choosing features
target = df['buying or not']

features_encoded = pd.get_dummies(features) #converting string features to int
model = DecisionTreeClassifier(criterion='entropy') #creating model
model.fit(features_encoded, target) # fitting dataset to the model

plt.figure(figsize=(12, 8))  
plot_tree(model, feature_names=features_encoded.columns, class_names=model.classes_, filled=True)
plt.title('Decision Tree Visualization')
plt.show()
new_customer = pd.DataFrame({
    'Age_middle': [0],
    'Age_old': [1],
    'Age_young': [0],
    'Income_average': [0],
    'Income_high': [0],
    'Income_low': [1],
    'Bought before_no': [1],
    'Bought before_yes': [0],
    'Credit_excellent': [1],
    'Credit_good': [0]
})

prediction = model.predict(new_customer)
print(prediction)
print("Will a newcomer buy the CPU?" , "Yes" if prediction[0] == 'buying' else "No")
