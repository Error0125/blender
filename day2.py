import numpy as np
import matplotlib.pyplot as plt
from day1 import *

'''phi
이전 segment(c), 현 seg(n), ca, c
c; n ca c 
psi
현 seg(n), ca, c, 다음 seg(n)
n ca c ; n
'''

phi_angle = {}
psi_angle = {}

print(parse1)
f = next(iter(parse1))

for i in range(f, len(parse1)+1):
    try:    
        a = parse1[i-1]['C']
        b = parse1[i]['N']
        c = parse1[i]['CA']
        d = parse1[i]['C']
        
        u1 = [round(float(b[0])-float(a[0]),3), round(float(b[1])-float(a[1]),3), round(float(b[2])-float(a[2]),3)]
        u2 = [round(float(c[0])-float(b[0]),3), round(float(c[1])-float(b[1]),3), round(float(c[2])-float(b[2]),3)]
        u3 = [round(float(d[0])-float(c[0]),3), round(float(d[1])-float(c[1]),3), round(float(d[2])-float(c[2]),3)]
        u1 = np.array(u1)
        u2 = np.array(u2)
        u3 = np.array(u3)  
      
        
        phi = np.arctan2(np.dot(np.sqrt(u2.dot(u2))*u1,np.cross(u2,u3)),np.dot(np.cross(u1,u2),np.cross(u2,u3)))
        phi_angle[i] = np.rad2deg(phi)
        
        
    except KeyError:
        continue
    
    try:    
    
        b = parse1[i]['N']
        c = parse1[i]['CA']
        d = parse1[i]['C']
        e = parse1[i+1]['N']
        
        v1 = [round(float(c[0])-float(b[0]),3), round(float(c[1])-float(b[1]),3), round(float(c[2])-float(b[2]),3)]
        v2 = [round(float(d[0])-float(c[0]),3), round(float(d[1])-float(c[1]),3), round(float(d[2])-float(c[2]),3)]
        v3 = [round(float(e[0])-float(d[0]),3), round(float(e[1])-float(d[1]),3), round(float(e[2])-float(d[2]),3)]
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)
        
        psi = np.arctan2(np.dot(np.sqrt(v2.dot(v2))*v1,np.cross(v2,v3)),np.dot(np.cross(v1,v2),np.cross(v2,v3)))
        psi_angle[i] = np.rad2deg(psi)
    
    except KeyError:
        continue   


print('phi_angle: ')   
print(phi_angle)
print()
print('psi_angle: ')
print(psi_angle) 
 
 
        

x = list(phi_angle.values())
x.remove(x[-1])
y = list(psi_angle.values())

plt.title('Ramachandran plot')
plt.scatter(x,y)
plt.xlabel('phi angle')
plt.ylabel('psi angle')
plt.xlim([-180,180])
plt.ylim([-180,180])
plt.show()
