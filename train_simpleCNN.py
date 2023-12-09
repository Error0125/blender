import torch
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#import models
import numpy as np
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"]= "3"
import matplotlib.pyplot as plt

#model
import torch.nn as nn

MAXEPOCH=1
BATCH=1

class CNN2(nn.Module):
    def __init__(self, nlayer=4, dropout=0.1): 
        super().__init__()
        layers = []

        drop = torch.nn.Dropout(p=dropout)
        conv1 = torch.nn.Conv1d(20,32,3,padding=1) # aa1hot,channel,
        layers = [drop,conv1]

        for k in range(nlayer):
            conv2 = torch.nn.Conv1d(32,32,3,padding=1) # aa1hot,channel,
            layers.append(conv2)
            layers.append(nn.BatchNorm1d(32)) #relu보다 더 좋은게 정규화하는 것. 정규화한 값을 활성함수의 입력값으로 넣고, 최종 출력값을 다음 레이어의 입력값으로 넣는다. 자세한건 메모장
            layers.append(nn.ReLU(inplace=True)) #활성화 함수. max(0,input) (linear                                    변환 후 비선형성 도입해서 신경망 복잡성                                     추가.
                                                 #inplace : 더이상 추가 메모리 소비하
                                                 #지 않고 돌리기
                                                 #gradient vanishing/exploding
                                                 #문제를 해결하기 위한 방법 중 1

        self.layers = nn.ModuleList(layers)#파이토치에게 파이썬 리스트에 모듈이 저장되어있음을 알려주기 위해 nn.modulist로 다시 래핑해주는 과정.
        # 1 x 32 x nres
        self.outlayer = nn.Linear(32,3)

    def forward(self, seq):
        #pred = seq # should B x 20 x nres
        for layer in self.layers:
            seq = layer(seq)

        seq = torch.transpose(seq,1,2) # put channel at the last

        pred = self.outlayer(seq)
        pred = torch.transpose(pred,2,1)
        #print(pred.size())
        return pred

class DataSet(torch.utils.data.Dataset): #사용자 정의 데이터셋. 반드시 3가지 포함할것!
    def __init__(self, datalist):
        super().__init__()
        self.tags = datalist

    def __len__(self):
        return len(self.tags) #데이터셋의 샘플 개수 반환

    def __getitem__(self,index): #인덱스에 해당하는 샘플을 데이터셋에서 불러오고 변형거쳐 반환.
        npz = '/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/sss_struct/'+ self.tags[index]
        
        data = np.load(npz,allow_pickle=True) #true로 안해주면 파일이 로드 안될 수도 있음.

        aas = 'ACDEFGHIKLMNPQRSTVWY' 
        SS3 = 'HEC'

        #print('😋hello')

        seqs = [aas.index(a) for a in data['seq'].tolist()]
        #print(f'seqs: {seqs}')
        SSs  = [SS3.index(a) for a in data['SS'].tolist()]
        #print(f'SSs: {SSs}')
        seq1hot = np.transpose(np.eye(20)[seqs],(1,0)) # 20xnres
        #SS1hot = np.transpose(np.eye(3)[SSs],(1,0))
        #print('!!!!!!!!!!')
        #print(seq1hot.shape[1])

    

        return seq1hot, SSs, seq1hot.shape[1]

def collate(samples):
    try:
        seq,SS,nres = map(list, zip(*samples))
        #print(f'what is seq: {seq}')
        #print(f'what is SS: {SS}')
        nres = max(nres)
        #print(f'maxnres: {nres}')
        B = len(seq)
        #print(f'b: {B}')

        # map into maxres
        seqs = torch.zeros(B,20,nres)
        SSs  = torch.zeros(B,nres,dtype=torch.long)
        for i,s in enumerate(seq):
            #print(f'seqs: {seqs}')
            #print(seqs[i,:,:len(s[1])]
            seqs[i,:,:len(s[1])] = torch.tensor(s)
        for i,s in enumerate(SS):
            #print(f'SSs: {SSs}')
            #print(f'ss[i]: {SS[i]}')
            #print(f'ss[i][:3]: {SS[i][:3]}')
            SSs[i][:len(s)] = torch.tensor(s)

        #print(f'seqs: {seqs}')
        #print(f'SSs: {SSs}')

    except:
        print("collate fail")
        return False, False

    return seqs, SSs

model = CNN2()
model.to(device) #cnn 모델을 gpu로 이동.

## load dataset
trainlist = np.load('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/train.npy')
validlist = np.load('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/valid.npy')

trainset = DataSet(trainlist)
validset = DataSet(validlist)

generator_params = {
    'shuffle': False, #모든 배치 순회 후 데이터셋의 셔플 여부
    'num_workers': 0, #참여시킬 cpu 코어 개수. (multi-loader, 튜닝을 통해 속도가 최대한
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

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4) #옵티마이저 초기화
#torch.optim은 경사하강법인 SGD를 중심으로 구성되어 있음. 여기서는 최적화함수를 Adam 이용
#hyperparameter. optimization 조절 파라미터 : epoch 수, batch size, learning rate존재.
#lr은 작을수록 학습속도 느려지고 커지면 예측불가 동작 발생 ㄱㄴ

valid_accuracies = [] 
train_losses = []
valid_losses = []

lossfunc = torch.nn.CrossEntropyLoss() #손실함수 초기화
for epoch in range(MAXEPOCH):
    if epoch != MAXEPOCH-1:
        loss_t = []
        for i,(seq,SS) in enumerate(train_generator): #enumerate func : 여러 객체를 숫자 매겨 셀 수 있도록 해줌.
            optimizer.zero_grad()
            # get prediction
            #print(f'SSpred seq: {seq.size(2)}')
            if seq.size(2) < 15: continue
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
            if seq.size(2) < 15: continue
            SSpred = model(seq.to(device))
            # calculate loss
            SS = SS.to(device)
            #print(f'what is SS: {SS}')
            #print(f'ss[0]: {SS[0]}')
            #print(f'what is SSpred: {SSpred}')
            #print(f'sspred[0]: {SSpred[0]}')
            loss = lossfunc(SSpred,SS)
            loss_v.append(loss.cpu().detach().numpy())

        accuracy = []  # List to store accuracy for each batch
        for i, (seq, SS) in enumerate(valid_generator):
            # get prediction
            if seq.size(2) < 15: continue
            SSpred = model(seq.to(device))
            SS = SS.to(device)

            # Calculate accuracy for each batch
            _, predicted = torch.max(SSpred, 1)  # Find the index of the maximum value along the last dimension
            acc = torch.sum(predicted == SS).item() / (SS.size(1))  # Compute accuracy for the batch
            accuracy.append(acc)  # Append accuracy for the batch to the list

        # Compute average accuracy for the epoch and append it to the list of validation accuracies

        train_loss = float(np.mean(loss_t))
        valid_loss = float(np.mean(loss_v))
        avg_accuracy = np.mean(accuracy)

        valid_accuracies.append(avg_accuracy)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        
       

        print("Train/Valid: %3d %8.4f %8.4f" % (epoch, train_loss, valid_loss))
        print("accuracy:", avg_accuracy)


model_info = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'valid_losses': valid_losses,
    'valid_accuracies': valid_accuracies
}

# Define the file path to save the model
model_save_path = '/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/save/mymodel.pth'  # Change this path to your desired location and filename

# Save the model
torch.save(model_info, model_save_path)



# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plotting Accuracy
plt.figure(figsize=(10, 5))
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')
plt.show()

