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



POSE_PAIRS = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]])

CORRESPOND_PAIRS = {0: 0, 1: 5, 2: 6, 3: 7, 4: 9, 5: 10, 6: 11, 7: 17, 8: 18, 9: 19, 10: 13, 11: 14, 12: 15, 13: 1,
                    14: 2, 15: 3, 16: 8, 17: 12, 18: 20, 19: 16, 20: 4}

UP_DOWN_PARIS = np.array([[3, 4], [7, 8], [11, 12], [15, 16], [19, 20], [2, 3], [6, 7], [10, 11], [14, 15], [18, 19],
                 [1, 2], [5, 6], [9, 10], [13, 14], [17, 18], [0, 1], [0, 5], [0, 9], [0, 13], [0, 17]])


SMAPLE_NUM, THRE = 10000, 0.00001
cnt = 0

img_folder_pth = './datasets/sign_language_dataset/'  # yy: dataset path containing the rgb images
# keypoints_folder_pth = './datasets/CMU_keypoints_5000/'  # yy: keypoints path containing the keypoints info
# colored_mh_pth = './mh_one_hand/'  # yy: path to the generated minimal hands (defined in generate_mh_info.py)
# skeleton_pth = './sign_language_multiple_hands_6_channels_results/'  # yy: path to save the generated dataset
dataset_6_channels_pth = './6_channels/'
dataset_3_channels_pth = './3_channels/'

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
        img_c1, img_c2, img_c3 = np.zeros_like(one_channel), np.zeros_like(one_channel), np.zeros_like(one_channel)

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
            
            blur_level, THICKNESS = 1, 1 
            color_c1 = [50, 150, 250, 100, 200]
            color_c2 = [100, 200, 50, 150, 250]

            updown_pair_joint_locations = np.array([points[p[0]][2] + points[p[1]][2] for p in UP_DOWN_PARIS])

            
            blur_level, THICKNESS = 1, 1
            color_c1 = [50, 150, 250, 100, 200]
            color_c2 = [100, 200, 50, 150, 250]
            for i in np.flip(np.argsort(pose_pair_joint_locations)):
                pair = POSE_PAIRS[i]
                color = color_c1[i // 4]
                partA = pair[0]
                partB = pair[1]

                if partA < points.shape[0] and partB < points.shape[0]:
                    cv2.line(img_c1, (points[partA][0], points[partA][1]), (points[partB][0], points[partB][1]), color, THICKNESS * 2)
                    cv2.circle(img_c1, (points[partA][0], points[partA][1]), THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(img_c1, (points[partB][0], points[partB][1]), THICKNESS, color, thickness=-1, lineType=cv2.FILLED)


                img_c1 = cv2.blur(img_c1, (blur_level, blur_level))


            for i in np.flip(np.argsort(updown_pair_joint_locations)):
                pair = UP_DOWN_PARIS[i]
                color = color_c2[i // 5]
                partA = pair[0]
                partB = pair[1]

                if partA < points.shape[0] and partB < points.shape[0]:
                    cv2.line(img_c2, tuple(np.array(points[pair[0]][:2], dtype=np.int)),
                            tuple(np.array(points[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)
                    cv2.circle(img_c2, tuple(np.array(points[pair[0]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                            lineType=cv2.FILLED)
                    cv2.circle(img_c2, tuple(np.array(points[pair[1]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                            lineType=cv2.FILLED)

                img_c2 = cv2.blur(img_c2, (blur_level, blur_level))


            for i in np.flip(np.argsort(pose_pair_joint_locations)):
                pair = POSE_PAIRS[i]
                color = channel_3_color
                partA = pair[0]
                partB = pair[1]

                if partA < points.shape[0] and partB < points.shape[0]:
                    cv2.line(img_c3, tuple(np.array(points[pair[0]][:2], dtype=np.int)),
                            tuple(np.array(points[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)
                    cv2.circle(img_c3, tuple(np.array(points[pair[0]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                            lineType=cv2.FILLED)
                    cv2.circle(img_c3, tuple(np.array(points[pair[1]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                            lineType=cv2.FILLED)

                img_c3 = cv2.blur(img_c3, (blur_level, blur_level))


            # stack 6 channels
            
        img_c1, img_c2, img_c3, ori_img = Image.fromarray(img_c1), Image.fromarray(img_c2), Image.fromarray(img_c3), Image.fromarray(ori_img)
        img_c1 = img_c1.resize((256, 256))
        img_c2 = img_c2.resize((256, 256))
        img_c3 = img_c3.resize((256, 256))
        ori_img = ori_img.resize((256, 256))

        img_c1, img_c2, img_c3, ori_img = np.asarray(img_c1), np.asarray(img_c2), np.asarray(img_c3), np.asarray(ori_img) 

        img_final = np.stack([ori_img[:, :, 0], ori_img[:, :, 1], ori_img[:, :, 2], img_c1, img_c2, img_c3], axis=2)
        img_final_name = f'{dataset_6_channels_pth}{cnt:06d}' + "_img_final.npy"
        np.save(img_final_name, img_final)

        real_image_with_skeleton_name = f'{dataset_3_channels_pth}{cnt:06d}' + "_real_img.jpg"
        cv2.imwrite(real_image_with_skeleton_name, ori_img)

        cnt += 1
        print(cnt)
            # except:
            #     continue
    except Exception as e:
        print(e)
        continue