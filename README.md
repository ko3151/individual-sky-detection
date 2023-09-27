# Sky Pixel (Region) Detection using Computer Vision Algorithm

This project aims to develop a computer vision algorithm that can automatically identify pixels that belong to the sky region using Python and OpenCV. The proposed computer vision algorithm processes images sourced from various folders (623, 684, 9730, 10917), each containing images captured during different times of the day and from diverse cameras and locations, utilizing a dataset from SkyFinder. The algorithm commences by conducting image preprocessing to categorize each image as either daytime or nighttime based on its brightness level. Subsequently, the sky region detection mechanism employs Sobel edge detection and energy optimization techniques to determine the optimal sky border position. Post-processing techniques are then applied to refine the segmentation outcome, followed by skyline detection using edge detection on the post-processed mask. For nighttime images, the algorithm seamlessly applies the best daytime mask to extract the sky region. Lastly, the algorithm quantitatively evaluates the accuracy and Mean Squared Error (MSE) of the detected sky region concerning the ground truth mask provided in the dataset. The output is visualized and saved in a dedicated result folder. Additionally, it computes the overall average accuracy and MSE specifically for daytime images across all folders, while also gauging the execution time to evaluate the effectiveness of the sky region detection algorithm.

## Steps
1. Download the images from camera **623**, **684**, **9730**, **10917** [here](https://cs.valdosta.edu/~rpmihail/skyfinder/images/index.html)

2. **Organize the Dataset**: Ensure that the downloaded image dataset is placed in the same folder as the algorithm code.

3. **Random Image Selection**: The algorithm will randomly select a specified number of images from each folder for evaluation. This random selection allows for diverse image testing.

4. **Visualization with Matplotlib**: The output of the algorithm is presented visually using the Matplotlib library. The generated figure will display three critical elements: the original image, the detected sky region, and the identified skyline. Additionally, an informative text line will be overlaid on the figure, presenting details such as the image filename, whether it's a daytime or nighttime image, the accuracy of the detected sky region, and the Mean Squared Error (MSE) compared to the ground truth mask. This visual representation facilitates easy reference and assessment of results.

5. **Result Output**: The algorithm saves the output figures in a designated "results" folder. The filenames are constructed based on the original folder name, followed by "_result.png". If the "results" folder does not exist, the algorithm will create it in the same directory as the code. This organized storage ensures that the algorithm's results are readily accessible and structured for analysis.

6. **Sample Results**: You can find a sample of the algorithm's output results in the "results" folder for reference and evaluation.
