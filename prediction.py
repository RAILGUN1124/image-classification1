import os
import torch
from torchvision import transforms
import PIL.Image as Image
from natsort import natsorted 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('best_model.pth', map_location=device)
image_transforms = transforms.Compose([
    transforms.ToTensor(),
])
directory = './hw4_test'
file_names = os.listdir(directory)
sorted_file_names = natsorted(file_names)
with open('prediction.txt', 'w') as file:
    model = model.eval()
    for x in range(10000):
        f = os.path.join(directory, sorted_file_names[x])
        image = Image.open(f).convert("RGB")
        image = image_transforms(image).float()
        image = image.unsqueeze(0)
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        file.write(str(predicted.item()))
        file.write('\n')
        if(x%100==0):
            print(f"{x/100} done")
    print("Done")