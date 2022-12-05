import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

## Initializations

dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv',parse_dates=True)
rowSize = dataset_1.shape[0]

for i in range(rowSize):
    if dataset_1["Precipitation"][i] == "T":
        dataset_1.at[i, "Precipitation"] = "0.00"
    else:
        onlyNum = dataset_1["Precipitation"][i]
        dataset_1.at[i, "Precipitation"] = onlyNum

for i in range (5, 10):
    for j in range(rowSize):
        rmComma = dataset_1.iat[j,i].split(",")
        toInt = ""
        for part in rmComma:
            toInt += part
        dataset_1.iat[j,i] = int(toInt)

for i in range (2,5):
    for j in range(rowSize):
        dataset_1.iat[j,i] = float(dataset_1.iat[j,i])

def Three_Bridge_Model(B1, B2, B3):
    X = dataset_1[[f"{B1} Bridge", f"{B2} Bridge", f"{B3} Bridge"]].values
    y = dataset_1[["Total"]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_prediction = model.predict(X_test)

    Predict_Score = model.score(X_test, y_test)

    return model, Predict_Score

## Problem 1
print("")
print("--Problem 1--")

Scores = [0,1,2,3]

BMW_Model = Scores[0] = Three_Bridge_Model("Brooklyn", "Manhattan", "Williamsburg")
QMB_Model = Scores[1] = Three_Bridge_Model("Queensboro", "Manhattan", "Brooklyn")
QMW_Model = Scores[2] = Three_Bridge_Model("Queensboro", "Manhattan", "Williamsburg")
QWB_Model = Scores[3] = Three_Bridge_Model("Queensboro", "Williamsburg", "Brooklyn")

a = Scores[0][1] * 100
b = Scores[1][1] * 100
c = Scores[2][1] * 100
d = Scores[3][1] * 100

print(f"Total Bike Traffic Model Part 1 = {BMW_Model[0].coef_[0][0]} * (Brooklyn Bridge) + {BMW_Model[0].coef_[0][1]} * (Manhattan Bridge) + {BMW_Model[0].coef_[0][2]} * (Williamsburg Bridge) + {BMW_Model[0].intercept_[0]}")
#print(QMB_Model[0].intercept_[0])
#print(QMW_Model[0].intercept_[0])
#print(QWB_Model[0].intercept_[0])
print("Brooklyn, Manhattan, and Williamsburg Bridges ->", "R-Squared Value: ",Scores[0][1],",", a,"%")
print("Brooklyn, Manhattan, and Queensboro Bridges ->", "R-Squared Value: ", Scores[1][1], b ,"%")
print("Brooklyn, Williamsburg, and Queensboro Bridges ->", "R-Squared Value: ", Scores[2][1], c ,"%")
print("Manhattan, Williamsburg, and Queensboro Bridges ->", "R-Squared Value: ", Scores[3][1], d ,"%")

## Problem 2
print("")
print("--Problem 2--")

X = dataset_1[["High Temp", "Low Temp", "Precipitation"]].values
y = dataset_1[["Total"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Column Names
col = (dataset_1[["High Temp", "Low Temp", "Precipitation"]]).columns

# Coefficients of Features
coeffs = pandas.DataFrame(regressor.coef_, ["coefficients"], columns=col)
y_pred = regressor.predict(X_test)
y_pred2 = regressor.predict(X)
days= [0] * 214
for i in range(214):
    days[i] = i + 1

plt.plot(days,y, color="orange")
plt.plot(days,y_pred2, color="blue")
plt.title("Total Bike Traffic vs. Temperature and Precipitation")
plt.xlabel("Days")
plt.ylabel("Total Bike Traffic")
plt.legend(["Actual", "Predicted"])
plt.show()

print("R-Squared Value: ", regressor.score(X_test, y_test),",",regressor.score(X_test, y_test) * 100,"%")
print(f"Total Bridge Traffic Model Part 2 = {regressor.coef_[0][0]} * (High Temp) + {regressor.coef_[0][1]} * (Low Temp) + {regressor.coef_[0][2]} * (Precipitation) + {regressor.intercept_[0]}")

## Problem 3
print("")
print("--Problem 3--")

df = dataset_1[['Day','Total']]
y = df['Day'].values
X = df['Total'].values
days_to_num = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
num_to_days = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
y = [days_to_num[day] for day in y]
y = np.array(y)

X = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
model = gnb.fit(X_train, y_train)
y_pred = model.predict(X_test)
    
print("Score: ", model.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)

monday = df[df['Day'] == 'Monday']
tuesday = df[df['Day'] == 'Tuesday']
wednesday = df[df['Day'] == 'Wednesday']
thursday = df[df['Day'] == 'Thursday']
friday = df[df['Day'] == 'Friday']
saturday = df[df['Day'] == 'Saturday']
sunday = df[df['Day'] == 'Sunday']

print("Monday Mean:",monday['Total'].mean())
print("Monday std:",monday['Total'].std())
print("\n")
print("Tuesday Mean:",tuesday['Total'].mean())
print("Tuesday std:",tuesday['Total'].std())
print("\n")
print("Wednesday Mean:",wednesday['Total'].mean())
print("Wednesday std:",wednesday['Total'].std())
print("\n")
print("Thursday Mean:",thursday['Total'].mean())
print("Thursday std:",thursday['Total'].std())
print("\n")
print("Friday Mean:",friday['Total'].mean())
print("Friday std:",friday['Total'].std())
print("\n")
print("Saturday Mean:",saturday['Total'].mean())
print("Saturday std:",saturday['Total'].std())
print("\n")
print("Sunday Mean:",sunday['Total'].mean())
print("Sunday std:",sunday['Total'].std())
print("\n")

plt.hist(monday['Total'], bins=10, alpha=0.5, label='Monday', color="red")
plt.hist(tuesday['Total'], bins=10, alpha=0.5, label='Tuesday', color="blue")
plt.hist(wednesday['Total'], bins=10, alpha=0.5, label='Wednesday', color="green")
plt.hist(thursday['Total'], bins=10, alpha=0.5, label='Thursday', color="yellow")
plt.hist(friday['Total'], bins=10, alpha=0.5, label='Friday', color="orange")
plt.hist(saturday['Total'], bins=10, alpha=0.5, label='Saturday', color="purple")
plt.hist(sunday['Total'], bins=10, alpha=0.5, label='Sunday', color="black")
plt.legend(loc='upper left')
plt.xlabel("Total")
plt.ylabel("Frequency")
plt.suptitle("Histogram of Total by Day")
plt.grid(True)
plt.show()
    