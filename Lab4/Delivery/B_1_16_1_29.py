import os
import sys
import cv2

import torch
import torch.nn as nn

import torchvision
from torchvision.utils import save_image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader


FILTER_DIM=None

# IMAGE_WIDTH=148
# IMAGE_HEIGHT=222
# IMAGE_CHANNELS=3

def read_filter(file_path):
    """Load filter from a text file."""
    
    with open(file_path, 'r') as f:
        # Read the first line to get the filter dimension
        filter_dim = int(f.readline().strip())
        # Read the subsequent lines to get the filter values
        filter_values = []
        for _ in range(filter_dim):
            line = f.readline().strip()
            values = [float(val) for val in line.split()]
            filter_values.append(values)

    # Convert to tensor and reshape to 3D (3, 3) tensor
    filter_tensor = torch.tensor(filter_values)

    # Repeat the filter 3 times along the channel dimension
    repeated_filter = filter_tensor.unsqueeze(0).repeat(3,1, 1)

    return filter_dim,repeated_filter


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)        
        self.transform =transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = cv2.imread(image_path)  # Read image using cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        if self.transform:
            image = self.transform(image=image)["image"]
        return image/255.0,self.image_files[idx]
    
def input_data_loader(folder_path,batch_size):

    # Transform
    transform = A.Compose([
        # A.PadIfNeeded(min_height=IMAGE_HEIGHT+FILTER_DIM-1, min_width=IMAGE_WIDTH+FILTER_DIM-1, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0], p=1.0),  # Add padding if needed
        ToTensorV2(p=1.0),  # Convert images to PyTorch tensors
    ])

    images_data=ImageFolderDataset(folder_path=folder_path,transform=transform)

    # Data Loader
    data_loader= DataLoader(dataset=images_data,batch_size=batch_size, shuffle=False)

    return data_loader




class Conv2D(nn.Module):
    def __init__(self,filter_dim,mask):
        '''
        filter_dim: int, the dimension of the filter (filter_dim x filter_dim)
        '''
        super(Conv2D, self).__init__()
        # Conv2D Layer 
        # Add Padding so output same size as input
        self.conv=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=filter_dim,padding="same")

        # Reshape the mask to fit weight shape and set it as the weight of the conv layer
        mask=mask.reshape(self.conv.weight.shape)

        # Set the mask as the weight of the conv layer
        self.conv.weight=nn.Parameter(mask)

        # Turn OFF the Gradient for the Filter No Learning :D
        self.conv.weight.requires_grad=False

        return

    def forward(self,input_image):
        '''
        input_image: torch.Tensor, the input image tensor
        '''
        # Forward Pass
        return self.conv(input_image)
 

def main():
    # Check if correct number of command-line arguments is provided
    if len(sys.argv) != 5:
        print("Usage: python conv.py input_dir output_dir batch_size filter_file_path")
        sys.exit(1)

    # Parse command-line arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    batch_size = int(sys.argv[3])
    filter_file = sys.argv[4]


    # Set the directory where you want to save the images
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    # 1. Reading Filter
    filter_dim,filter=read_filter(file_path=filter_file)

    # 2. Processing Input as batches
    data_loader=input_data_loader(input_dir,batch_size=batch_size)


    # Define Convolutional Neural Network
    Conv2D_model=Conv2D(filter_dim=filter_dim,mask=filter)


    for _, (images,images_name) in enumerate(data_loader):
        # Each Batch
        # print(images.shape) #[2,3,256,512]

        # Apply convolution [Forward Pass]
        convolved_images=Conv2D_model(images).detach().numpy()*255


        # Check Output is the same size as input
        assert convolved_images.shape[2] == images.shape[2]
        assert convolved_images.shape[3] == images.shape[3]


        # Iterate over the batch
        for i in range(convolved_images.shape[0]):
            try:
                # Generate the image path
                image_path = os.path.join(output_dir, images_name[i])

                # Save the image
                cv2.imwrite(image_path, convolved_images[i][0])

                print("Saved Image Successfully:", image_path)
            except Exception as e:
                print(f"Error occurred while saving image {images_name[i]}:", e)

    print("All convolved images saved successfully.")
        


if __name__ == "__main__":
    main()

# python ./conv.py ./input ./output_python 2 ./filters/avg_3_3.txt