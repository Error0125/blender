import torch
from train_simpleCNN import CNN2, DataSet, collate  
import numpy as np
import matplotlib.pyplot as plt

# Define your device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
saved_model_path = '/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/save/mymodel2.pth'
model_info = torch.load(saved_model_path)

model = CNN2()  
model.load_state_dict(model_info['model_state_dict']) 
model.to(device)  # Move model to device (CPU or GPU)

# Load the test dataset
testlist = np.load('/Users/oyujeong/Desktop/2023 2학기/ps4-dataset/test.npy')
testset = DataSet(testlist)

test_generator_params = {
    'shuffle': False,
    'num_workers': 0,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': 1,
    'worker_init_fn': np.random.seed()
}
test_generator = torch.utils.data.DataLoader(testset, **test_generator_params)

loss_func = torch.nn.CrossEntropyLoss()

test_losses = []
test_accuracy = []

model.eval()
with torch.no_grad():
    for seq, SS in test_generator:
        seq, SS = seq.to(device), SS.to(device)
        SSpred = model(seq)

        # Calculate loss
        loss = loss_func(SSpred, SS)
        test_losses.append(loss.item())

        # Calculate accuracy
        _, predicted = torch.max(SSpred, 1)
        acc = torch.sum(predicted == SS).item() / SS.size(1)
        test_accuracy.append(acc)
        #print(loss, acc)

avg_test_loss = np.mean(test_losses)
avg_test_accuracy = np.mean(test_accuracy)

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy * 100:.2f}%")

'''
# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot([avg_test_loss], label='Test Loss')
plt.xlabel('Metrics')
plt.ylabel('Loss')
plt.legend()
plt.title('Test Loss')
plt.show()

# Plotting Accuracy
plt.figure(figsize=(10, 5))
plt.plot([avg_test_accuracy], label='Test Accuracy')
plt.xlabel('Metrics')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test Accuracy')
plt.show()
'''