# Annotated-Hand-Dataset-Generation

## Create a new conda environment using environment.yml

- conda env create --file environment.yml --name {env_name}

## Generate mediapipe keypoints

- Create new folders mediapipe_results/annotated_mediapipe_images and mediapipe_results/generated_keypoints
- Put the dataset in the datasets folder. Change the dataset location in the hand_keypoint_detection.py file (around line 10)
- Run "python hand_keypoint_detection.py"

## Generated the final dataset using mediapipe keypoints

- Create two folders 6_channels/ and 3_channels/.
- Change the datasets path in the generate_dataset.py (around line 29). 
- Run "python generate_dataset_openpose_color.py".
- 3_channels dataset would be in 3_channels/ folder and 6_channels npy dataset would be in 6_channels/.

## Notes by Yue
6 channels images will all be saved in `./6_channels` folder. You could first check whether the generated 3 channels are correct by checking `./lst_3_channels` :)