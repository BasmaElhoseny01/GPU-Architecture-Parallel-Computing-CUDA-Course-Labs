import cv2
import sys
import numpy as np

def compare_images(image1_path, image2_path, threshold=100):
    # Read the images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if images are loaded successfully
    if image1 is None or image2 is None:
        print("Error: Could not read the images. Please check the file paths.")
        return None
    
    # Resize images to the same size if they have different dimensions
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # Compute Mean Squared Error (MSE) between the two images
    mse = np.mean((image1 - image2) ** 2)
    
    # Check if images are similar or not based on the threshold
    if mse < threshold:
        return True
    else:
        return False

def main():
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py image1_path image2_path")
        return
    
    # Read the filenames from command-line arguments
    image1_filename = sys.argv[1]
    image2_filename = sys.argv[2]
    
    # Compare images
    similar = compare_images(image1_filename, image2_filename)
    if similar is not None:
        if similar:
            print("Images are similar.")
        else:
            print("Images are not similar.")

if __name__ == "__main__":
    main()



# python compare_images.py ./input/image_1.jpeg ./output_k2/image_1.jpeg