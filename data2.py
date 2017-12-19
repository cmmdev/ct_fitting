import numpy as np
import math
import random
import sys
import pickle
import os
import sklearn


tot_values = 400
mean = 0
variance = 0.5
lower_limit = 0
upper_limit = 10

# simulate data.
# X: [ [x0(150-240), x1(x0~250)]
# y:  g(c0,c1) = |c0-c1|*e^|(|c1-150\)/255| * (1+random()*0.1)


def x0():
    return 150 + (240 - 150) * random.random()


def x1(x0):
    return x0 + (250 - x0) * random.random()


def g(c0, c1):
    print type(c0), type(c1)
    return abs(c0 - c1) * math.pow(math.e, abs(c1 - 150) / 110.) + 2.0 * np.random.normal(mean, variance)


def g_nd(c0, c1):
    result = []
    for i in range(len(c0)):
        row = []
        for j in range(len(c0[i,::])):
            c0f = c0[i,j]
            c1f = c1[i,j]
            row.append(g(c0f, c1f))
        result.append(row)
    return np.array(result)




X = []  # 1 x 2  c0 c1

os.remove('data.pkl')

output = open('data.pkl', 'wb')
dumped = []

for i in range(tot_values):
    c0 = x0()
    c1 = x1(c0)
    X.append([c0, c1])
for i in X:
    [c0, c1] = i
    y0 = g(c0, c1)
    dumped.append([c0, c1, y0])


pickle.dump(dumped, output)
output.close()

print "generate data [c0, c1, y0] done"

print sklearn.__version__, tot_values


# plot scatters
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

dd = np.array(dumped)

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m in [('r', 'o')]:
    xs = dd[::,0]
    ys = dd[::,1]
    zs = dd[::,2]
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# plt.show()


 ### plot mesh
fig = plt.figure()

x, y = np.mgrid[140:255:10, 140:255:10]
ax.plot_surface(x, y, g_nd(x,y), rstride=1, cstride=1)
ax.set_zlim(0, 0.2)

# savefig('../figures/plot3d_ex.png',dpi=48)
plt.show()


