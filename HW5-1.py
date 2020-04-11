import pandas as pd
import numpy as np

# read excel file
df = pd.read_excel("HW5-1.xls")

# declaration
input_data = df.iloc[:,:2].values
output_data = df.iloc[:,2:].values
input_data = np.mat(input_data)
output_data = np.mat(output_data)
tran = np.mat(input_data.transpose())

# minimize the following cost min_x_f(x) = || Ax - Y ||^2.
optimal_x = np.linalg.inv(tran * input_data) * tran * output_data

# print("input data: ")
# print(input_data)
# print("=================")
# print("output data: ")
# print(output_data)
# print("=================")
# print("input transpose data: ")
# print(tran)
print("\n=================")
print("optimal x that minimize the cost of linear regression model Y = Ax: ")
print(optimal_x)

optimal_x = np.mat(optimal_x.transpose())
df = pd.DataFrame(optimal_x)
df.columns = ['x1', 'x2']
df.to_excel("5-1_ans.xls", index = False)

