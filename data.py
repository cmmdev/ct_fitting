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
# 198 210 205 215 => 150 180 155 190 <==> g(c0,c1) = |c0-c1|*e^|(|c1-150\)/255| * (1+random()*0.1) , f(c0,c1,t0,t1)=g(c0,c1)/t(c0,c1) + random()*0.05;
# X: [ [x0(150-240), x1(x0~250), x2( x0-10 ~ x0+10), x3(x2~250) ]


def x0():
    return 150 + (240 - 150) * random.random()


def x1(x0):
    return x0 + (250 - x0) * random.random()


def x2(x0):
    return x0 + 20 * (random.random() - 0.5)


def x3(x2):
    return x2 + (250 - x2) * random.random()


def g(c0, c1):
    return abs(c0 - c1) * math.pow(math.e, abs(c1 - 150) / 255.)


def f1(x_0, x1, x2, x3):
    return g(x_0, x1) / g(x2, x3) + 0.1 * np.random.normal(mean, variance)


def f(x0, x1, x2, x3):
    return abs(x0 - x1) / (abs(x2 - x3) + 0.0001)

X = []  # 1 x 4  c0 c1 t0 t1

os.remove('data.pkl')

output = open('data.pkl', 'wb')
dumped = []

for i in range(tot_values):
    c0 = x0()
    c1 = x1(c0)
    c2 = x2(c0)
    c3 = x3(c2)
    g0 = g(c0, c1)
    g1 = g(c2, c3)

    X.append([c0, c1, c2, c3, g0, g1])

for i in X:
    [c0, c1, c2, c3, g0, g1] = i
    y0 = f(c0, c1, c2, c3)
    dumped.append([c0, c1, c2, c3, g0, g1, y0])


pickle.dump(dumped, output)
output.close()

print "generate data [c0, c1, t0, t1, y0] done"

print sklearn.__version__






