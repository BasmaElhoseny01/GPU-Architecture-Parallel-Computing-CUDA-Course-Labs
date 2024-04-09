import os
import sys
from PIL import Image

def resize_images(input_folder, output_folder, width, height):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            try:
                # Open the image file
                with Image.open(os.path.join(input_folder, filename)) as img:
                    # Convert image to RGB if it has 4 channels
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')

                    # Resize the image
                    resized_img = img.resize((width, height))

                    # Save the resized image to the output folder
                    resized_img.save(os.path.join(output_folder, filename))
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    # Set the input and output folders
    input_folder = "input_original"  # Change this to the path of your input folder
    output_folder = "input"  # Change this to the path of your output folder

    # Check if correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python resize.py width height")
        return

    width = int(sys.argv[1])
    height = int(sys.argv[2])

    resize_images(input_folder, output_folder, width, height)
    print("Images resized successfully.")

if __name__ == "__main__":
    main()


# python ./scripts/resize.py 512 256