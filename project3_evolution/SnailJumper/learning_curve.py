#!/usr/bin/python3

import matplotlib.pyplot as plt
import sys
import json

try:
    a = sys.argv[1]
    try:
        a = int(a)
        p = "./learning_curve_{}.json".format(a)
    except:
        p = a
    jf = open(p, 'r')
    j = dict(json.load(jf))
    jf.close()
except:
    print("usage: python3 learning_curve.py <json file number or file path>")
    exit()

g = [int(k) for k in j.keys()]
fmin = [int(j[k]['min']) for k in j.keys()]
fmax = [int(j[k]['max']) for k in j.keys()]
favr = [int(j[k]['avr']) for k in j.keys()]


plt.xlabel('Generation')
plt.ylabel('fitness')
plt.title('Learning curve')

plt.plot(g, fmin, label = "min")
plt.plot(g, fmax, label = "max")
plt.plot(g, favr, label = "avr")

plt.show()

