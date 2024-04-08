import os
import sys
import cv2

import torch
import torch.nn.functional as F

from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader

FILTER_DIM=3

IMAGE_WIDTH=148
IMAGE_HEIGHT=222
IMAGE_CHANNELS=3
BATCH_SIZE=1

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
    repeated_filter = filter_tensor.unsqueeze(0).repeat(3, 1, 1)

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
        return image,self.image_files[idx]
    
def input_data_loader(folder_path):

    # Transform
    transform = A.Compose([
        # A.PadIfNeeded(min_height=IMAGE_HEIGHT+FILTER_DIM-1, min_width=IMAGE_WIDTH+FILTER_DIM-1, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0], p=1.0),  # Add padding if needed
        ToTensorV2(p=1.0)  # Convert images to PyTorch tensors
    ])

    images_data=ImageFolderDataset(folder_path=folder_path,transform=transform)

    # Data Loader
    data_loader= DataLoader(dataset=images_data,batch_size=BATCH_SIZE, shuffle=False)

    return data_loader

  
def convolve(batches_images,filter):
    # Assuming image is a batched 3-channel image tensor
    # image shape: [batch_size, channels, height, width]
    # Ensure the filter tensor has the correct dimensions
    filter = filter.unsqueeze(0)  # Add a batch dimension if necessary

    # Convert input images to float tensors if necessary
    batches_images = batches_images.float()

    # Apply convolution
    result = F.conv2d(batches_images, filter,stride=1,padding=(FILTER_DIM-1)//2, groups=batches_images.shape[0])

    # Round the output to the nearest integer
    result = torch.round(result)

    # Convert the result back to integer type
    result = result.type(torch.uint8)

    return result



def main():
    # Check if correct number of command-line arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python conv.py input_dir output_dir filter_file_path")
        sys.exit(1)

    # Parse command-line arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    filter_file = sys.argv[3]

    # 1. Reading Filter
    filter_dim,filter=read_filter(file_path=filter_file)
    assert filter_dim==FILTER_DIM

    # 2. Processing Input as batches
    data_loader=input_data_loader(input_dir)

    for batch_idx, (images,images_name) in enumerate(data_loader):
        # # Each Batch

        # Convolve
        # Apply convolution
        result=convolve(batches_images=images,filter=filter)

        # Check on output Dime
        width=result.shape[3]
        height=result.shape[2]
        assert width==IMAGE_WIDTH
        assert height==IMAGE_HEIGHT
     

        # Iterate over images in the batch
        for i in range(result.shape[0]):

            result_numpy = result[i].numpy().astype('uint8')  # Convert to numpy array and cast to uint8

            print(result_numpy.shape)

            # Save result using OpenCV
            output_path = os.path.join(output_dir, os.path.basename(images_name[i]))
            cv2.imwrite(output_path, result_numpy[0])

            print(f"Processed {images_name[i]} and saved result to {output_path}")


    
    
        



if __name__ == "__main__":
    main()
