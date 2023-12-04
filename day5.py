import numpy as np
import pandas as pd
from day4 import *

'''strand & coil 찾기'''

ss = {}
for i in range(1,len(parse1)+1):
    ss[i] = 'C'

for i in range(1,len(ll)+1):

    if [ll[i][1]+2, ll[i][0]-2] in ll.values():
        print(ll[i])
       
        ss[ll[i][1]] = 'E'
    if [ll[i][1]-2, ll[i][0]-2] in ll.values():
        print(ll[i])
         
        ss[ll[i][1]] = 'E'
    if [ll[i][1]+2, ll[i][0]+2] in ll.values():
        print(ll[i])
         
        ss[ll[i][1]] = 'E'
    if [ll[i][1]-2, ll[i][0]+2] in ll.values():
        print(ll[i])
         
        ss[ll[i][1]] = 'E' 
          
    else:
        continue
    


'''helix 찾기'''
for k,v in ll.items():
    if (v[1]+4 == v[0]):
        print(v)
        ss[v[1]] = 'H'



for i in range(2,len(ss)):
    if (ss[i-1] == 'E' and ss[i+1] == 'E') and ((-180 < phi_angle[i] < 0) and (50< psi_angle[i] <180)):
        ss[i] = 'E'
 
'''ss dictionary'''
print(ss)  
     

