import cv2
import os
from tqdm import tqdm
def make_video(image_folder, output_video, frame_rate=30):
    files = os.listdir(image_folder)
    png_files = [file for file in files if file.endswith('.png')]
    images = []
    for i in range(len(png_files)):
        filename = os.path.join(image_folder, f"snapshot_{i:03}.png")
        if os.path.isfile(filename):
            images.append(filename)
        else:
            print(f"Warning: {filename} does not exist.")
    
    if not images:
        print("No images found in the specified folder.")
        return
    
    # Read the first image to get the width and height
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for idx, image in enumerate(tqdm(images, desc="Processing images", unit="image")):
        try:
            frame = cv2.imread( image)
            if frame is None:
                print(f"Error reading image {image}. Skipping.")
                continue
            video.write(frame)
        except Exception as e:
            print(f"Error processing image {image}: {e}")
    video.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    # /nethome/asingh3064/flash/home-robot/datadump/images/eval_test_114/102816756_599
    image_folder = "/nethome/asingh3064/flash/home-robot/datadump/images/eval_test_114/102816756_599"  # Change this to the path of your image folder
    output_video = "output_video24.mp4"  # Change this to your desired output video file name
    frame_rate = 5  # Adjust the frame rate as needed

    make_video(image_folder, output_video, frame_rate)
