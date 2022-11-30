import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as stats

data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
high = list(data['High Temp (°F)'])
low = list(data['Low Temp (°F)'])
total = list(data['Total'])
total = [i.replace(",","") for i in total]
for i in range(0, len(total)):
    total[i] = float(total[i])

plt.scatter(high,total,color="blue")
plt.title('Total Riders vs High Temp')
plt.xlabel('High Temp')
plt.ylabel('Total Riders')
plt.grid(True)
plt.show()

plt.scatter(low,total,color="red")
plt.title('Total Riders vs Low Temp')
plt.xlabel('Low Temp')
plt.ylabel('Total Riders')
plt.grid(True)
plt.show()

X = data[['High Temp (°F)','Low Temp (°F)']]
Y = total

X1 = stats.add_constant(X)
model = stats.OLS(Y, X1)
results = model.fit()
print("params: \n", results.params)
print("R squared:", results.rsquared)

