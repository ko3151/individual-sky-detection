import cv2
import numpy as np
import os
import time
import random
from matplotlib import pyplot as plt

# Function to detect the image is taken during daytime or nighttime
def is_daytime(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    avg_brightness = np.mean(img_gray)    
    threshold = 100 # The threshold value is used to determine daytime or nighttime   
    
    # Compare the average brightness with the threshold
    if avg_brightness > threshold:
        return True
    else:
        return False

# Function to convert the image into a gradient image based on detected edges using the Sobel operator
def return_gradient_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert BGR image into grayscale
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0) # horizontal edges
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1) # vertical edges
    gradientImg = np.sqrt(sobel_x**2 + sobel_y**2) # Combine horizontal and vertical edges
    
    return gradientImg


# Function to find or output the sky border position
def return_border_position(gradientImg, thresh):
    # Creating the sky array with size of input image's width and filling it with the image's height
    skyBorder = np.full(gradientImg.shape[1], gradientImg.shape[0])
    
    # Loop through the columns (x-coordinates) of the gradient image
    for x in range(gradientImg.shape[1]):
        # np.argmax returns the index with the maximum intensity value on the vertical axis,
        # highlighting the brightest point, which is most likely part of the boundary line
        borderPosition = np.argmax(gradientImg[:, x] > thresh)
        
        # If the detected boundary position is greater than 0, it is considered part of the sky or horizon line
        if borderPosition > 0:
            skyBorder[x] = borderPosition
            
    return skyBorder

# Function to create a binary mask for the sky region based on given border points
def create_mask(borderPoints, img):
    # Creates mask the size (length x width) of the input image
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    
    # Loop through the borderPoints list, which contains (x, y) coordinates
    # Assign intensity value 255 to pixels below the border points, indicating ground pixels
    for x, y in enumerate(borderPoints):
        mask[y:, x] = 255

    # Create a new binary mask representing the sky region by inverting the original mask
    sky_mask = cv2.bitwise_not(mask)
    return sky_mask

# Function to output energy optimisation
def calc_energy_optimization(border_points, img):
    # Create binary mask based on estimated border values
    sky_mask = create_mask(border_points, img)

    # Check if sky region is empty (no pixels detected as sky)
    if np.sum(sky_mask == 255) == 0:
        # If sky region is empty, return a very low energy value
        return 1e-20

    # Determine ground and sky section in the mask created, returned as a 1-D array
    ground_region = np.ma.array(img, mask=cv2.cvtColor(cv2.bitwise_not(sky_mask), cv2.COLOR_GRAY2BGR)).compressed()
    ground_region.shape = (ground_region.size // 3, 3)
    sky_region = np.ma.array(img, mask=cv2.cvtColor(sky_mask, cv2.COLOR_GRAY2BGR)).compressed()
    sky_region.shape = (sky_region.size // 3, 3)

    # Check if there are enough samples for covariance calculation
    if sky_region.size < 3:
        # If there are not enough samples, return a very low energy value
        return 1e-20

    # Calculate covariance matrix of ground and sky regions of the image,
    # returning covariance value and average RGB values of the two regions
    covar_ground, average_ground = cv2.calcCovarMatrix(ground_region, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    covar_sky, average_sky = cv2.calcCovarMatrix(sky_region, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)

    # Energy optimization function Jn, function/equation (1)
    gamma = 2
    energyVal = 1 / (
        (gamma * np.linalg.det(covar_sky) + np.linalg.det(covar_ground)) +
        (gamma * np.linalg.det(np.linalg.eig(covar_sky)[1]) +
         np.linalg.det(np.linalg.eig(covar_ground)[1])))

    return energyVal

# Function to compute the optimal sky border position based on the previous computed sky border position by returnBorderPosition
def border_optimisation(img, gradientImg):
    minThresh=5
    maxThresh=600
    searchStep=5
    
    num_samples = ((maxThresh - minThresh) // searchStep) + 1 # number of sample points in the search space
    borderOptimal = None # Initialise optimal border list  
    energyMax = 0 # Initialise maximum energy function value

    for c in range(1, num_samples + 1):
        # Threshold used to compute optimal sky border position function
        thresh = minThresh + ((maxThresh - minThresh) // num_samples - 1) * (c - 1)
        
        # Obtain values signifying the sky-land border points
        bordertmp = return_border_position(gradientImg, thresh)
        
        # Calculate energy difference at that border point to determine if it's the optimal point in the border line
        # If it has more energy than the current energy maximum, the border point is recognized as the optimal border point,
        # and the energy maximum is updated
        energyVal = calc_energy_optimization(bordertmp, img)

        if energyVal > energyMax:
            energyMax = energyVal
            borderOptimal = bordertmp
            
    return borderOptimal

# Function to detect the skyline using Sobel edge detection
def detect_skyline(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Apply thresholding to create a binary mask for the detected skyline
    threshold = 0
    img = (gradient_mag > threshold).astype(np.uint8)
    
    # Create a copy of the original image to draw the edges
    detected_skyline = np.copy(img)
    detected_skyline[img != 0] = 255

    return detected_skyline

# Function to post-process the mask
def postprocess_mask(mask):
    kernel = np.ones((20,20),np.uint8) * 255
    inverted_mask = cv2.bitwise_not(mask)
    closed_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.bitwise_not(closed_mask)
    
    return final_mask

# Function to calculate the accuracy and mse
def evaluation(ground_truth_mask, detected_sky_mask):
    # Check if both masks have the same shape
    if ground_truth_mask.shape != detected_sky_mask.shape:
        raise ValueError("The ground truth mask and detected mask must have the same shape.")
    
    # Calculation for confusion matrix
    TP = np.sum(np.logical_and(ground_truth_mask, detected_sky_mask))
    FP = np.sum(np.logical_and(detected_sky_mask, np.logical_not(ground_truth_mask)))
    TN = np.sum(np.logical_and(np.logical_not(detected_sky_mask), np.logical_not(ground_truth_mask)))
    FN = np.sum(np.logical_and(np.logical_not(detected_sky_mask), ground_truth_mask))
       
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    mse = np.mean((ground_truth_mask - detected_sky_mask) ** 2)
    
    return accuracy, mse

# Function to visualize the result and save into result folder
def visualize_and_save_results(img, sky_region, skyline, postprocessed_mask, filename, accuracy, mse, daytime_or_nighttime, result_folder):
    # Text output to display on the figure
    text_output = f"Filename: {filename}; {daytime_or_nighttime}; Accuracy: {accuracy * 100:.2f}% Mean Squared Error: {mse:.4f}"

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original image")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("Detected Sky Region")
    plt.imshow(sky_region)

    plt.subplot(1, 3, 3)
    plt.title("Detected Skyline")
    plt.imshow(skyline, cmap="gray")

    plt.figtext(0.5, 0.95, text_output, ha='center', va='center', fontsize=12)
    plt.subplots_adjust(top=0.85)
    
    # Create the 'result' folder if it doesn't exist
    if not os.path.exists("result"):
        os.makedirs("result")
        print("SUCCESS: The 'Results' directory was created successfully.")

    # Save the figure in the 'result' folder with the filename as the image filename
    result_filepath = os.path.join(result_folder, f"{filename}_result.png")
    plt.savefig(result_filepath)
    plt.close()  # Close the figure after saving

def main():
    # Process images from four folders
    folders = ['images/623','images/684','images/9730','images/10917' ]
    
    # Initialize variables
    all_accuracy = []
    all_mse = []
    
    start_time = time.time()
    for folder in folders:
        # Create a subfolder inside 'result' to store results for each folder
        result_folder = os.path.join("result", os.path.basename(folder))
        os.makedirs(result_folder, exist_ok=True)
        
        # Determine the correct sample mask path for the current folder
        sample_mask_path = os.path.join("images", f"{os.path.basename(folder)}_mask.png")    
    
        # Get the list of files in the current folder
        files = [file for file in os.listdir(folder) if file.endswith(".jpg")]
        num_images_to_select = 5 # the number can be changed
        selected_files = random.sample(files, num_images_to_select)
        
        # Initialize variables
        total_daytime_images = 0
        total_accuracy = 0.0
        total_mse = 0.0
        average_accuracy = []
        average_mse = []
        nighttime_images = []
        best_daytime_mask = None
        best_daytime_accuracy = 0.0
        best_daytime_filename = "" 
        
        # Iterate through the images in the folder
        for file in selected_files:
            image_path = os.path.join(folder, file)
            img = cv2.imread(image_path)[:,:,::-1]
    
            # Determine daytime or nighttime
            daytime_or_nighttime = "daytime" if is_daytime(img) else "nighttime"
    
            if daytime_or_nighttime == "daytime":
                gradientImg = return_gradient_img(img)
                borderOptimal = border_optimisation(img, gradientImg)
                mask = create_mask(borderOptimal, img)
                postprocessed_mask = postprocess_mask(mask)
                skyline = detect_skyline(postprocessed_mask)
    
                # Apply mask to the original image to obtain the sky region
                sky_region = cv2.bitwise_and(img, img, mask=postprocessed_mask)
                sample_mask = cv2.imread(sample_mask_path, 0)
                accuracy, mse = evaluation(postprocessed_mask, sample_mask)
    
                # Call the visualize_results function to display the results and save the figure
                visualize_and_save_results(img, sky_region, skyline, postprocessed_mask, file, accuracy, mse, daytime_or_nighttime, result_folder)
    
                # Store the accuracy and mse in the lists along with the folder name
                average_accuracy.append((folder, file, accuracy))
                average_mse.append((folder, file, mse))
    
                # Calculate total accuracy and mse for the folder
                total_accuracy += accuracy
                total_mse += mse
                total_daytime_images += 1
                
                # Check if the accuracy of this daytime image is higher than the current best
                if accuracy > best_daytime_accuracy:
                    best_daytime_accuracy = accuracy
                    best_daytime_mask = postprocessed_mask
                    best_daytime_filename = file  # Update the best daytime filename
    
            else:
                # Store the nighttime image for later use
                nighttime_images.append((img, file))
                
            # Apply the best daytime mask to the nighttime images
        for nighttime_img, nighttime_file in nighttime_images:
            # Apply best daytime mask to the nighttime image to obtain the sky region   
            best_daytime_mask = cv2.resize(best_daytime_mask, (nighttime_img.shape[1], nighttime_img.shape[0]))
            nighttime_sky_region = cv2.bitwise_and(nighttime_img, nighttime_img, mask=best_daytime_mask)
            nighttime_postprocessed_mask = best_daytime_mask
            nighttime_skyline = detect_skyline(nighttime_postprocessed_mask)  
            nighttime_accuracy, nighttime_mse = evaluation(nighttime_postprocessed_mask, sample_mask)    
            visualize_and_save_results(nighttime_img, nighttime_sky_region, nighttime_skyline, nighttime_postprocessed_mask, nighttime_file, nighttime_accuracy, nighttime_mse, "nighttime", result_folder)
        
            
    
        # Calculate average accuracy and mse for the folder (excluding nighttime images)
        if total_daytime_images > 0:
            folder_average_accuracy = total_accuracy / total_daytime_images
            folder_average_mse = total_mse / total_daytime_images
    
            print(f"\nFolder: {os.path.basename(folder)}")
            print(f"Average Accuracy: {folder_average_accuracy * 100:.2f}%")
            print(f"Average Mean Squared Error: {folder_average_mse:.4f}")
            print(f"Total daytime images: {total_daytime_images}")
            print(f"Total nighttime images: {len(nighttime_images)}")
    
        # Store the accuracy and mse of each image in the all_accuracy and all_mse lists
        all_accuracy.extend(average_accuracy)
        all_mse.extend(average_mse)
    
    # Calculate overall average accuracy and mse for all folders
    if len(all_accuracy) > 0:
        overall_average_accuracy = sum(acc for _, _, acc in all_accuracy) / len(all_accuracy)
        overall_average_mse = sum(mse for _, _, mse in all_mse) / len(all_mse)
    
        print("\nOverall Average Accuracy (daytime only): {:.2f}%".format(overall_average_accuracy * 100))
        print("Overall Average Mean Squared Error (daytime only): {:.4f}".format(overall_average_mse))
    
    # Calculate the total running time of the algorithm
    running_time = time.time() - start_time
    print(f"\nTotal running time: {running_time:.4f} seconds")

if __name__ == "__main__":
    main()