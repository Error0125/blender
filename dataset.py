import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



#AMINO ACID DICTIONARY

amino = {'ALA':'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 
         'GLU' : 'E', 'GLN' : 'Q', 'GLY' : 'G', 'HIS' : 'H', 'ILE': 'I', 'LEU': 'L', 
         'LYS': 'K', 'MET' : 'M', 'PHE': 'F', 'PRO' : 'P', 'SER' : 'S',
         'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL': 'V'}



files = os.listdir("/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/ps4_pdblist")

for file_name in files:
    with open(os.path.join("/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/ps4_pdblist",file_name),'r') as pdbfile:
        try:
            for line in pdbfile:
                if line[:4] == 'ATOM':
                    a = line 
                    i = a[21:22].strip()
                    f = open(os.path.join('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/ps4_pdblist', i + '_'+ file_name),'a')
                    f.write(a[:])
                    f.close()
        except UnicodeDecodeError:
            continue
            
        os.remove('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/ps4_pdblist/'+str(file_name))
       
            

 
files = os.listdir("/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/ps4_pdblist")

for file_name in files:
    with open(os.path.join("/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/ps4_pdblist",file_name),'r') as pdbfile:
        print(file_name)
        k = {}  
        parse1 = {}
        try:
            for line in pdbfile:
                if line[:4] == 'ATOM':
                    a = line 
                    if len(a[17:20].strip()) <3:
                        try:
                            os.remove('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/ps4_pdblist/'+str(file_name))
                        except FileNotFoundError:
                            k = 0
                            pass
                        
                    else:
                        i = int(a[22:26])
                        if i in parse1:
        
                            parse1[i][a[13:16].strip()] = [a[31:38].strip(), a[38:46].strip(), a[46:54].strip()] 
            
                        else:
                            parse1[i] = {}
                            parse1[i][a[13:16].strip()] = [a[31:38].strip(), a[38:46].strip(), a[46:54].strip()] 
                        try:
                            k[i] = amino[a[17:20].strip()]
                        except (KeyError, TypeError):
                            k = 0
                            pass
        except UnicodeDecodeError:
            continue
        
        if k == 0:
            pass
    
        else:    
            seq = list(k.values())
            seq = ''.join(seq)
         
        
            phi_angle = {}
            psi_angle = {}
        
            p = list(parse1.keys())

            for i in range(0, len(p)):
                try:    
                    a = parse1[p[i]-1]['C']
                    b = parse1[p[i]]['N']
                    c = parse1[p[i]]['CA']
                    d = parse1[p[i]]['C']
            
                    u1 = [round(float(b[0])-float(a[0]),3), round(float(b[1])-float(a[1]),3), round(float(b[2])-float(a[2]),3)]
                    u2 = [round(float(c[0])-float(b[0]),3), round(float(c[1])-float(b[1]),3), round(float(c[2])-float(b[2]),3)]
                    u3 = [round(float(d[0])-float(c[0]),3), round(float(d[1])-float(c[1]),3), round(float(d[2])-float(c[2]),3)]
                    u1 = np.array(u1)
                    u2 = np.array(u2)
                    u3 = np.array(u3)  
        
            
                    phi = np.arctan2(np.dot(np.sqrt(u2.dot(u2))*u1,np.cross(u2,u3)),np.dot(np.cross(u1,u2),np.cross(u2,u3)))
                    phi_angle[p[i]] = np.rad2deg(phi)
            
            
                except IndexError:
                    continue   
                except KeyError:
                    continue
                except ValueError:
                    pass
        
                try:    
        
                    b = parse1[p[i]]['N']
                    c = parse1[p[i]]['CA']
                    d = parse1[p[i]]['C']
                    e = parse1[p[i+1]]['N']
            
                    v1 = [round(float(c[0])-float(b[0]),3), round(float(c[1])-float(b[1]),3), round(float(c[2])-float(b[2]),3)]
                    v2 = [round(float(d[0])-float(c[0]),3), round(float(d[1])-float(c[1]),3), round(float(d[2])-float(c[2]),3)]
                    v3 = [round(float(e[0])-float(d[0]),3), round(float(e[1])-float(d[1]),3), round(float(e[2])-float(d[2]),3)]
                    v1 = np.array(v1)
                    v2 = np.array(v2)
                    v3 = np.array(v3)
            
                    psi = np.arctan2(np.dot(np.sqrt(v2.dot(v2))*v1,np.cross(v2,v3)),np.dot(np.cross(v1,v2),np.cross(v2,v3)))
                    psi_angle[p[i]] = np.rad2deg(psi)
        
                except IndexError:
                    continue   
                except KeyError:
                    continue
                except ValueError:
                    pass
                
                
            hbond= np.arange(len(parse1)**2).reshape(len(parse1),len(parse1))
            hbond = hbond.astype(np.float64)
           
            for i in range(0,len(p)):
                for j in range(0, len(p)):
                    try:
                        n = parse1[p[i]]['N']
                        o = parse1[p[j]]['O']
                    except KeyError:
                        continue
            
                    dist = np.sqrt((float(n[0])-float(o[0]))**2+(float(n[1])-float(o[1]))**2+(float(n[2])-float(o[2]))**2)
                    hbond[i][j] = dist
                

            nnum = []
            onum = []
            idx = np.where(hbond<3.5)
            n1num = idx[0]
        
            for i in range(0,len(n1num)):
                nnum.append(p[n1num[i]])
            o1num = idx[1]
            for i in range(0,len(o1num)):
                onum.append(p[o1num[i]])

            hbond_idx = pd.DataFrame({'N': nnum, 'O': onum })
        
            nnum = list(hbond_idx['N'])
            onum = list(hbond_idx['O'])

            hh = {}
            for i in range(1,len(nnum)+1):
                hh[i] = [nnum[i-1],onum[i-1]]

            ll = {}
            m = 1
            for q,v in hh.items():
                if (v[0] != v[1]) and (v[0]+1 != v[1]) and (v[0]-1 != v[1]):
                    ll[m] = v
                    m+= 1


            ss = {}
            for i in range(0,len(p)):
                ss[p[i]] = 'C'

            for i in range(1,len(ll)+1):

                if [ll[i][1]+2, ll[i][0]-2] in ll.values():
                    ss[ll[i][1]] = 'E'
                if [ll[i][1]-2, ll[i][0]-2] in ll.values():
                    ss[ll[i][1]] = 'E'
                if [ll[i][1]+2, ll[i][0]+2] in ll.values():
                    ss[ll[i][1]] = 'E'
                if [ll[i][1]-2, ll[i][0]+2] in ll.values():
                    ss[ll[i][1]] = 'E' 
                else:
                    continue
        
            #helix 
            for q,v in ll.items():
                if (v[1]+4 == v[0]):
                
                    ss[v[1]] = 'H'


            for i in range(0,len(ss)):
                try:
                    if (ss[p[i]-1] == 'E' and ss[p[i]+1] == 'E') and ((-180 < phi_angle[p[i]] < 0) and (50< psi_angle[p[i]] <180)):
                        ss[p[i]] = 'E'
                except KeyError:
                    continue
    
            #ss dictionary
            SS = []
            dict = {}
            for q,v in ss.items():
                SS.append(str(v))
            SS = ''.join(SS)
            
            dict['seq'] = seq
            dict['SS'] = SS
          
            np.savez(os.path.join('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/sss_struct',  file_name), **dict)
            



files = os.listdir("/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/sss_struct")
num = len(files)
train = []
valid = []
test = []


i=0
for file in files:
    i += 1
    if file == '.DS_Store':
        pass
    else: 
        if i <= int(18731*0.8):
            train.append(file)
        else: #i <= 18731:
            valid.append(file)
        #else:
        #    test.append(file)
    
        
train = np.array(train)
valid = np.array(valid) 
#test = np.array(test)     
            
np.save('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/train.npy', train)
np.save('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/valid.npy', valid)     
#np.save('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/test.npy', test)   

