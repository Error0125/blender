import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from day2 import *

hbond= np.arange(len(parse1)**2).reshape(len(parse1),len(parse1))
hbond = hbond.astype(np.float64)

for i in range(f,len(parse1)+1):
    for j in range(f, len(parse1)+1):
        try:
            n = parse1[i]['N']
            o = parse1[j]['O']
        
            dist = np.sqrt((float(n[0])-float(o[0]))**2+(float(n[1])-float(o[1]))**2+(float(n[2])-float(o[2]))**2)
            hbond[i-1][j-1] = dist
        except KeyError:
            pass


idx = np.where(hbond<3.5)
nnum = idx[0]
onum = idx[1]

hbond_idx = pd.DataFrame({'N': nnum, 'O': onum })
hbond_idx += 1
print(hbond_idx)

nnum = list(hbond_idx['N'])
onum = list(hbond_idx['O'])

hh = {}
for i in range(1,len(nnum)+1):
    hh[i] = [nnum[i-1],onum[i-1]]

ll = {}
m = 1
for k,v in hh.items():
    if (v[0] != v[1]) and (v[0]+1 != v[1]) and (v[0]-1 != v[1]):
        ll[m] = v
        m+= 1
        
'''Hbond pair indices
print(ll)
'''

