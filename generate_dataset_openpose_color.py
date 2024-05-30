import cv2
import time
import numpy as np
from PIL import Image
import os
import json


"""
Description: generate the dataset in preparation for the Generative model training
TODO: draw fingers far away first, then draw closer fingers
"""

# Define finger colors
finger_colors = {
    "thumb": ["#ff0a00", "#fe4d00", "#ff9900", "#ffe502"],
    "index": ["#ccff02", "#80ff00", "#33ff00", "#00ff19"],
    "middle": ["#01FF66", "#01FFB3", "#01FFFF", "#00B2FF"],
    "ring": ["#0866FF", "#0F19FF", "#3300FF", "#8001FF"],
    "pinky": ["#CC01FF", "#FF00E6", "#FF0099", "#FF054D"]
}

color_ls = [c for _, cs in finger_colors.items() for c in cs]


# Function to convert hex color to BGR for OpenCV
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))




POSE_PAIRS = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]])

CORRESPOND_PAIRS = {0: 0, 1: 5, 2: 6, 3: 7, 4: 9, 5: 10, 6: 11, 7: 17, 8: 18, 9: 19, 10: 13, 11: 14, 12: 15, 13: 1,
                    14: 2, 15: 3, 16: 8, 17: 12, 18: 20, 19: 16, 20: 4}

UP_DOWN_PARIS = np.array([[3, 4], [7, 8], [11, 12], [15, 16], [19, 20], [2, 3], [6, 7], [10, 11], [14, 15], [18, 19],
                 [1, 2], [5, 6], [9, 10], [13, 14], [17, 18], [0, 1], [0, 5], [0, 9], [0, 13], [0, 17]])


SMAPLE_NUM, THRE = 10000, 0.00001
cnt = 0

img_folder_pth = './datasets/10_imgs/'  # yy: dataset path containing the rgb images
# keypoints_folder_pth = './datasets/CMU_keypoints_5000/'  # yy: keypoints path containing the keypoints info
# colored_mh_pth = './mh_one_hand/'  # yy: path to the generated minimal hands (defined in generate_mh_info.py)
# skeleton_pth = './sign_language_multiple_hands_6_channels_results/'  # yy: path to save the generated dataset
dataset_6_channels_pth = './6_channels/'
dataset_3_channels_pth = './3_channels/'
dataset_lst_3_channels_pth = './lst_3_channels/'

for i, img in enumerate(sorted(os.listdir(img_folder_pth))[:SMAPLE_NUM]):
    print(img)
    file = img

    if '.png' not in file and '.jpg' not in file:
        continue
    print('here')

    if(i > 10000):
        continue
    try:
        print(i)
        mh_img_kp_img = np.zeros((1081, 1081, 3))
        colored_mh_img = np.zeros((1081, 1081,3 ))

        # iterate kp img
        kp_positions = {}
        z_positions = {}
        mh_img_kp_img = mh_img_kp_img / 255
    

        hand_points = np.load('./mediapipe_results/generated_keypoints/fingertips_' + file[:-4]  + '.npy', allow_pickle=True)
        hand_handedness = np.load('./mediapipe_results/generated_keypoints/handedness_' + file[:-4] + '.npy')
        
        ori_img = cv2.imread(os.path.join(img_folder_pth, img))
        ori_img = cv2.resize(ori_img, (256, 256))
        print(ori_img.shape)
        one_channel = ori_img[:, :, -1:]
        # img_c1, img_c2, img_c3 = np.zeros_like(one_channel), np.zeros_like(one_channel), np.zeros_like(one_channel)
        img_c1 = np.zeros_like(ori_img)

        for i, points in enumerate(hand_points):
            print(np.max(points[:, 1]))
            print(np.min(points[:, 1]))
            points = points*255
            points = points.astype(int)
            points = np.array([[255 - p[0], p[1], p[2]] for p in points])

            # calculate average color for each line
            pose_pair_joint_locations = np.array([points[p[0]][2] + points[p[1]][2] for p in POSE_PAIRS])
            if(hand_handedness[i] == 0):
                channel_3_color = 100
            elif(hand_handedness[i] == 1):
                channel_3_color = 200

            THICKNESS = 1
            for i in np.flip(np.argsort(pose_pair_joint_locations)):
                pair = POSE_PAIRS[i]
                # color = color_c1[i // 4]
                color = color_ls[i]
                color = hex_to_bgr(color)
                partA = pair[0]
                partB = pair[1]

                if partA < points.shape[0] and partB < points.shape[0]:
                    cv2.line(img_c1, (points[partA][0], points[partA][1]), (points[partB][0], points[partB][1]), color, THICKNESS * 2)
                    cv2.circle(img_c1, (points[partA][0], points[partA][1]), THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(img_c1, (points[partB][0], points[partB][1]), THICKNESS, color, thickness=-1, lineType=cv2.FILLED)


        # Save the 6-channel numpy array
        img_final = np.squeeze(np.stack(
            [ori_img, img_c1],
            axis=2))
        img_final_name = f'{dataset_6_channels_pth}{cnt:06d}' + "_img_final.npy"
        np.save(img_final_name, img_final)

        # Save last 3-channel numpy array
        img_lst_3c_pth = f'{dataset_lst_3_channels_pth}{cnt:06d}' + "_lst_3c.jpg"
        cv2.imwrite(img_lst_3c_pth, img_c1)

        cnt += 1
        print(cnt)
            # except:
            #     continue
    except Exception as e:
        print(e)
        continue