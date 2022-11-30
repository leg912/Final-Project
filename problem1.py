import pandas 
from matplotlib import pyplot as plt
import statistics


dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
print(dataset_1.to_string()) #This line will print out your data


x = [0] * 214
for i in range(214):
    x[i] = i + 1


Brooklyn_Avg = statistics.mean(dataset_1['Brooklyn Bridge'])
Manhattan_Avg = statistics.mean(dataset_1['Manhattan Bridge'])
Queensboro_Avg = statistics.mean(dataset_1['Queensboro Bridge'])
Williamsburg_Avg = statistics.mean(dataset_1['Williamsburg Bridge'])

print('Brooklyn Bridge Average: ', Brooklyn_Avg,)
print('Manhattan Bridge Average: ', Manhattan_Avg,) 
print('Queensboro Bridge Average: ', Queensboro_Avg,)
print('Williamsburg Bridge Average: ', Williamsburg_Avg)
fig, pos = plt.subplots(2, 2)
fig.suptitle("Number of Bikes in Each City Per Day")
fig.legend(title='The Black Line is the Average Number of Bikes Per Day', loc='upper left')
pos[0,0].plot(x, dataset_1['Brooklyn Bridge'], color = 'orange')
pos[0,0].plot(x, [Brooklyn_Avg] * 214, color = 'black')
pos[0,0].set_title('Brooklyn',fontsize=10, fontweight='bold')
pos[0,0].set(xlim =(0,214), ylim=(0, 10000))
pos[0,0].set(xlabel = 'Number of Days', ylabel = 'Number of Bikes')


pos[1,0].plot(x, dataset_1['Manhattan Bridge'],color = 'purple')
pos[1,0].set_title('Manhattan',fontsize=10, fontweight='bold')
pos[1,0].plot(x, [Manhattan_Avg] * 214, color = 'black')
pos[1,0].set(xlim =(0,214), ylim=(0, 10000))
pos[1,0].set(xlabel = 'Number of Days', ylabel = 'Number of Bikes')


pos[0,1].plot(x, dataset_1['Williamsburg Bridge'], color = 'red')
pos[0,1].set_title('Williamsburg',fontsize=10, fontweight='bold')
pos[0,1].plot(x, [Williamsburg_Avg] * 214, color = 'black')
pos[0,1].set(xlim =(0,214), ylim=(0, 10000))
pos[0,1].set(xlabel = 'Number of Days', ylabel = 'Number of Bikes')


pos[1,1].plot(x, dataset_1['Queensboro Bridge'], color = 'green')
pos[1,1].set_title('Queensboro',fontsize=10, fontweight='bold')
pos[1,1].plot(x, [Queensboro_Avg] * 214, color = 'black')
pos[1,1].set(xlim =(0,214), ylim=(0, 10000))
pos[1,1].set(xlabel = 'Number of Days', ylabel = 'Number of Bikes')


plt.show()