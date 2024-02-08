from code.stage_3_code.CustomDataset import CustomDataset
from code.stage_3_code.Data import Data
from torchvision import transforms
from torch.utils.data import DataLoader
from code.stage_3_code.Method_CNN import BaseCNN
import torch

data_path = '../../data/stage_3_data/ORL'
data = Data(data_path)
X_train, X_test, y_train, y_test = data.get_train_test()

# img_size = (112, 91)
transform = transforms.Compose([
    # transforms.Resize(img_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = CustomDataset(X_train, y_train, transform=transform)
test_dataset = CustomDataset(X_test, y_test, transform=transform)
batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = BaseCNN(input_shape=3, hidden_units=10, output_shape=40)

img_batch, label_batch = next(iter(train_dataloader))
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}\n")
model.eval()
with torch.inference_mode():
    pred = model(img_single)

print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")