from pyexpat import model
import pandas
import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
from sklearn.linear_model import Ridge

"""
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
"""
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))
dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True)) 
dataset_1['Precipitation']  = pandas.to_numeric(dataset_1['Precipitation'].replace(',','', regex=True))
dataset_1['High Temp']  = pandas.to_numeric(dataset_1['High Temp'].replace(',','', regex=True))
dataset_1['Low Temp']  = pandas.to_numeric(dataset_1['Low Temp'].replace(',','', regex=True))

#print(dataset_1.to_string()) #This line will print out your data

"""The use of the code provided is optional, feel free to add your own code to read the dataset. The use (or lack of use) of this code is optional and will not affect your grade."""
Brooklyn = list(dataset_1['Brooklyn Bridge'])
Manhattan = list(dataset_1['Manhattan Bridge'])
Queensboro = list(dataset_1['Queensboro Bridge'])
Williamsburg = list(dataset_1['Williamsburg Bridge'])
total = list(dataset_1['Total'])
high_temp = list(dataset_1['High Temp']) 
low_temp = list(dataset_1['Low Temp'])
precipitation = list(dataset_1['Precipitation'])
total = list(dataset_1['Total'])


#Question 1
average_list = []
std_list = []
normlized_data = []
ones_array = len(total) * [1]
all_bridges = [ones_array, Brooklyn, Manhattan, Queensboro, Williamsburg]
all_bridges = list(np.transpose(np.array(all_bridges)))

for element in [Brooklyn, Manhattan, Queensboro, Williamsburg, total]:
    average = sum(element) / len(element)
    std = statistics.stdev(element)
    average_list.append(average)
    std_list.append(std)
    normed = []

    for number in element:
        normed_number = (number - average) / std
        normed.append(normed_number)
    
    normlized_data.append(normed)

normlized_inde = normlized_data[0 : 4]


constant_list = []

for element in normlized_inde[0]:
    constant_list.append(1)
normlized_inde.append(constant_list)

normlized_de = normlized_data[4]

normlized_inde = np.array(normlized_inde)
normlized_inde = np.transpose(normlized_inde)
normlized_inde_T = np.transpose(normlized_inde)
normlized_de = np.array(normlized_de)
First_term = np.matmul(normlized_inde_T, normlized_inde)
First_term_inverse = np.linalg.inv(First_term)
Second_term = np.matmul(First_term_inverse, normlized_inde_T)
model1 = list(np.matmul(Second_term, normlized_de))
print("The model for problem 1 is: ")
print(model1)
print("")

total_predict = []
normlized_inde = list(normlized_inde)

for element in normlized_inde:
    total_pred = 0
    i = 0

    for coefficient in element:
        adder = model1[i] * coefficient
        total_pred = total_pred + adder
    
    total_predict.append(total_pred)

normlized_de = list(normlized_de)

mse1 = mean_squared_error(normlized_de, total_predict)
r2_1 = r2_score(normlized_de, total_predict)
print("The MSE for the first model is ", mse1, ".")
print("The coefficient of determination for the first model is ", r2_1, ".")
print("")


#Question 2
def normalize_list(list):
    return_list = []
    for i in range(len(list)):
        return_list.append((list[i] - statistics.mean(list))/statistics.stdev(list))
    return return_list

norm_high = normalize_list(high_temp)
norm_low = normalize_list(low_temp)
norm_pre = normalize_list(precipitation)
norm_total = normalize_list(total)

data = []
data.append(norm_high)
data.append(norm_low)
data.append(norm_pre)
data = np.transpose(np.array(data))
norm_total = np.array(norm_total)

l = np.array(np.logspace(-5, 5, num = 51, base = 10))

def train_model(x,y,l):
    model = Ridge(alpha=l,fit_intercept=True)
    model.fit(x,y)
    y_pred = model.predict(x)
    mse =  mean_squared_error(y,y_pred)
    return model,mse

model_list = []
mse_list = []

for val in l:
    modell,mse = train_model(data,norm_total,val)
    mse_list.append(mse)
    model_list.append(modell)

min_val = min(mse_list)
min_idx = mse_list.index(min_val)

print("The model for Question 2 is: ")
print(model_list[min_idx].coef_)
print(model_list[min_idx].intercept_)
print("")
print("Best lambda is ", l[min_idx])
print("mse = ", min_val)
print("")

plt.plot(l,mse_list)
plt.title("MSE vs Lamda")
plt.xlabel("lamda")
plt.ylabel("MSE")
plt.show()

#Question 3
if_rain = []
do_rain = []
not_rain = []
i = 0

for element in precipitation:
    if element == 0:
        if_rain.append(0)
        not_rain.append(total[i])
    else:
        if_rain.append(1)
        do_rain.append(total
        [i])
    
    i = i + 1

logreg = LogisticRegression()
model_log = logreg.fit(all_bridges, if_rain)
model3 = model_log.coef_[0]
y_pred = model_log.predict(all_bridges)
score3 = metrics.accuracy_score(if_rain, y_pred)

print("The model for Question 3 is: ")
print(model3)
print("")
print("The accuracy score is ",score3,".")
print("")

