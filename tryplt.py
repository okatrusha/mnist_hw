import numpy as np
import plot_rnd

X1 = [i for i in range(10000)]
print(X1)
X1 = np.reshape(X1,newshape=[100,100])
print(X1.shape)

plot_rnd.rplt(X1, X1)

