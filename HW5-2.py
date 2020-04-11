import pandas as pd
import numpy as np

df = pd.read_excel("HW5-2.xls")
rows, columns = df.shape[0], df.shape[1]

ss = np.mat(df.iloc[:,:])
q = np.random.random((4, 100))
q = np.mat(q)

EPOCHS = 50000
LR = 0.0005

for epoch in range(EPOCHS):
    loss = 0
    for i in range(100):
        q1, q2, q3, q4 = q[0,i], q[1, i], q[2, i], q[3, i]
        ax, ay, az = ss[i, 0], ss[i, 1], ss[i, 2]

        jac_tran = np.mat([[-2 * q3, 2 * q2,       0],
                           [ 2 * q4, 2 * q1, -4 * q2],
                           [-2 * q1, 2 * q4, -4 * q3],
                           [ 2 * q2, 2 * q3,       0]])

        f = np.array([[2 * (q2 * q4 - q1 * q3) - ax],
                      [2 * (q1 * q2 + q3 * q4) - ay],
                      [2 * (0.5 - q2 * q2 - q3 * q3) - az]])

        q[:, i] += -LR * (jac_tran * f)
        loss += np.sum(f)
    print("epoch: %d, loss: %d"%(epoch, loss))

print("\n====== gradient descent finished =====\n")

q = q.transpose()
q = np.mat(q)
df = pd.DataFrame(q)
df.columns = ['q1', 'q2', 'q3', 'q4']
df.to_excel("5-2_ans.xls", index = False)
