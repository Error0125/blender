import torch
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#import models
import numpy as np
import os

#model
import torch.nn as nn

MAXEPOCH=100
BATCH=1

class CNN(nn.Module):
    def __init__(self, nlayer=4, dropout=0.1): #드롭아웃의 비율을 랜덤으로 0.1로 함. 학습                                            시에만 사용하고 예측시엔 사용하지 x
        super().__init__()
        layers = []

        drop = torch.nn.Dropout(p=dropout)
        conv1 = torch.nn.Conv1d(21,32,3,padding=1) # aa1hot,channel,
        layers = [drop,conv1]

        for k in range(nlayer):
            conv2 = torch.nn.Conv1d(32,32,3,padding=1) # aa1hot,channel,
            layers.append(conv2)
            layers.append(nn.BatchNorm1d(32))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.ModuleList(layers)

        # 1 x 32 x nres
        self.outlayer = nn.Linear(32,3)

    def forward(self, seq):
        #pred = seq # should B x 20 x nres
        for layer in self.layers:
            seq = layer(seq)

        seq = torch.transpose(seq,1,2) # put channel at the last

        pred = self.outlayer(seq)
        pred = torch.transpose(pred,2,1)
        return pred

class DataSet(torch.utils.data.Dataset): #사용자 정의 데이터셋. 반드시 3가지 포함할것!
    def __init__(self, datalist):
        super().__init__()
        self.tags = datalist

    def __len__(self):
        return len(self.tags) #데이터셋의 샘플 개수 반환

    def __getitem__(self,index): #인덱스에 해당하는 샘플을 데이터셋에서 불러오고 변형거쳐 반환.
        npz = 'data/'+self.tags[index]

        data = np.load(npz,allow_pickle=True) #true로 안해주면 파일이 로드 안될 수도 있음.

        aas = 'ACDEFGHIKLMNPQRSTVWYX'
        SS3 = 'HEC'

        seqs = [aas.index(a) for a in data['sequence'].tolist()[0]]
        SSs  = [SS3.index(a) for a in data['SS'][0]]
        seq1hot = np.transpose(np.eye(21)[seqs],(1,0)) # 21xnres
        #SS1hot = np.transpose(np.eye(3)[SSs],(1,0))
        #print(seq1hot.shape[1])
        return seq1hot, SSs,seq1hot.shape[1]

def collate(samples):
    try:
        seq,SS,nres = map(list, zip(*samples))
        nres = max(nres)
        B = len(seq)

        # map into maxres
        seqs = torch.zeros(B,21,nres)
        SSs  = torch.zeros(B,nres,dtype=torch.long)
        for i,s in enumerate(seq): seqs[i][:len(s[1])] = torch.tensor(s)
        for i,s in enumerate(SS):  SSs[i][:len(s)] = torch.tensor(s)

    except:
        print("collate fail")
        return False, False

    return seqs, SSs

model = CNN()
model.to(device) #cnn 모델을 gpu로 이동.

## load dataset
trainlist = np.load('data/train.npy')
validlist = np.load('data/valid.npy')

trainset = DataSet(trainlist)
validset = DataSet(validlist)

generator_params = {
    'shuffle': False, #학습 순서 셔플 여부
    'num_workers': 1, #참여시킬 cpu 코어 개수. (multi-loader, 튜닝을 통해 속도가 최대한
                      #안느려지는 선에서 조절 가능. default = 0)
    'pin_memory': True, #트루의 경우, 텐서를 cuda 메모리에 올림. 즉, cpu에서 돌아가던 데이터셋을
                        #gpu로 옮겨서 계산을 진행하도록 한다.
    'collate_fn': collate, #데이터 사이즈를 맞추기 위해 사용.?
    'batch_size': BATCH,
    'worker_init_fn' : np.random.seed() #어떤 worker를 불러올것인가를 리스트로 전달
                                        #np.random.seed()는 괄호안의 숫자를 시작으로 예측 가능한 난수를 형성하는 것. 시작 숫자를 고정한다!
                                            #고정 숫자가 동일하면 생성되는 난수도 동일함. (패턴이 동일하니까)
}
train_generator = torch.utils.data.DataLoader(trainset, **generator_params)
valid_generator = torch.utils.data.DataLoader(validset, **generator_params)

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
#torch.optim은 경사하강법인 SGD를 중심으로 구성되어 있음. 여기서는 최적화함수를 Adam 이용

lossfunc = torch.nn.CrossEntropyLoss()
for epoch in range(MAXEPOCH):
    if epoch != MAXEPOCH-1:
        loss_t = []
        for i,(seq,SS) in enumerate(train_generator):
            optimizer.zero_grad()
            # get prediction
            #if not SS: continue
            SSpred = model(seq.to(device))
            # calculate loss
            SS = SS.to(device)
            loss = lossfunc(SSpred,SS)
            loss.backward(retain_graph=True)
            optimizer.step()

            loss_t.append(loss.cpu().detach().numpy())
        #print("TRAIN:", epoch, float(np.mean(loss_)))
        
        loss_v = []
        for i,(seq,SS) in enumerate(valid_generator):
            # get prediction
            SSpred = model(seq.to(device))
            # calculate loss
            SS = SS.to(device)
            loss = lossfunc(SSpred,SS)
            loss_v.append(loss.cpu().detach().numpy())
        
        accuracy = []    
        for i,(seq,SS) in enumerate(valid_generator):
            # get prediction
            SSpred = model(seq.to(device))
            SS = SS.to(device)
            big = []
            for j in range(SS[0][1]):
                big.append(max([SSpred[0][0][j],SSpred[0][1][j],SSpred[0][2][j]]))
            big = torch.tensor(big)
            c = big==SS[0]
            accuracy.append(len(SS[0][c])/len(SS[0][1]))
                           
     
            
        
        print("Train/Valid: %3d %8.4f %8.4f"%(epoch, float(np.mean(loss_t)), float(np.mean(loss_v))))
        print("accuracy:", np.mean(accuracy))
    
  

                
                
