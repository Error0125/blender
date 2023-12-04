import numpy as np
import os

parse1 = {}
'''AMINO ACID DICTIONARY'''

amino = {'ALA':'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 
         'GLU' : 'E', 'GLN' : 'Q', 'GLY' : 'G', 'HIS' : 'H', 'ILE': 'I', 'LEU': 'L', 
         'LYS': 'K', 'MET' : 'M', 'PHE': 'F', 'PRO' : 'P', 'SER' : 'S',
         'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL': 'V'}

k = []
j = []


files = os.listdir("/Users/kistintern7/Desktop/test_")

for file_name in files:
    with open(os.path.join("/Users/kistintern7/Desktop/test_",file_name),'r') as pdbfile:
        try:
            for line in pdbfile:
                if line[:4] == 'ATOM':
                    a = line 
                    i = a[21:22].strip()
                    f = open(os.path.join('/Users/kistintern7/Desktop/test_', i + '_'+ file_name),'a')
                    f.write(a[:])
                    f.close()
        except UnicodeDecodeError:
            continue
            
        os.remove('/Users/kistintern7/Desktop/test_/'+str(file_name))
       
            
        
 
files = os.listdir("/Users/kistintern7/Desktop/test_")

for file_name in files:
    with open(os.path.join("/Users/kistintern7/Desktop/test_",file_name),'r') as pdbfile:
        try:
            for line in pdbfile:
                if line[:4] == 'ATOM':
                    a = line 
                    i = int(a[23:26])
                    if i in parse1:
        
                        parse1[i][a[13:16].strip()] = [a[32:39].strip(), a[40:47].strip(), a[48:55].strip()] 
            
                    else:
                        parse1[i] = {}
                        parse1[i][a[13:16].strip()] = [a[32:39].strip(), a[40:47].strip(), a[48:55].strip()] 
            
        
                    k.append(amino[a[17:20].strip()])
        except UnicodeDecodeError:
            continue
        
        
        seq = []            
        for i in range(1,len(k)):
            if k[i-1] !=  k[i]:
                seq.append(k[i-1])
        seq.append(k[len(k)-1])
        seq = ''.join(seq)
        print(seq)   
        print()
 
 
 
 


'''
klist = list(parse1.keys())
for i in range(1,klist[-1]+1):
    if i not in parse1:
        parse1[i] = 'X'
'''



     
'''task 1
print(parse1[3]['N'])
'''

'''task 2 
   print(seq) to see 1-letter sequence'''

                
       


