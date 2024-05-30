import cv2
import os
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
file_path = './datasets/10_imgs' # Path to the folder containing the images
IMAGE_FILES = os.listdir(file_path)
max_image_count = 10000
print(IMAGE_FILES[0])

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 0.6
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 1

with mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0,
    min_tracking_confidence = 0) as hands:
  img_count = 0
  for idx, file in enumerate(IMAGE_FILES):
    
    if '.png' not in file and '.jpg' not in file:
      continue
    if(img_count == max_image_count):
      break
    
    image = cv2.flip(cv2.imread(f'{file_path}/{file}'), 1)
    
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      print(file)
      continue
    
    img_count += 1
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    j = 0
    hand_in_image = []
    for hand_landmarks in results.multi_hand_landmarks:
      print(f'Hand {j + 1}')
      j += 1
    
      fingertips = np.zeros((21, 3))
      
      c = 0
      for landmark in mp_hands.HandLandmark:
        # print(landmark)
        fingertips[c][0] = hand_landmarks.landmark[landmark].x
        fingertips[c][1] = hand_landmarks.landmark[landmark].y
        fingertips[c][2] = hand_landmarks.landmark[landmark].z
        c += 1

      hand_in_image.append(fingertips)
      
      np.set_printoptions(suppress=True)  

      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      
      c = 0
      for landmark in mp_hands.HandLandmark:
        org = (int(255*hand_landmarks.landmark[landmark].x), int(255*hand_landmarks.landmark[landmark].y))
        c += 1
      
    cv2.imwrite(
        './mediapipe_results/annotated_mediapipe_images/' + file[:-4] + '_annotated_image.png', cv2.flip(annotated_image, 1))
    
    np.save('./mediapipe_results/generated_keypoints/fingertips_' + file[:-4]  + '.npy', np.asarray(hand_in_image))

    hands_in_image_handedness = []
    for i, hand_handedness in enumerate(results.multi_handedness):
      handedness =  hand_handedness.classification[0].label
      # print(handedness)
      if handedness == 'Right':
        hands_in_image_handedness.append(np.asarray(1))
      elif handedness == 'Left':
        hands_in_image_handedness.append(np.asarray(0))
    
    # print(hands_in_image_handedness)
    # Save handedness
    np.save('./mediapipe_results/generated_keypoints/handedness_' + file[:-4]  + '.npy', np.asarray(hands_in_image_handedness))
    # Draw hand world landmarks.
    # print('Line 122')
    if not results.multi_hand_world_landmarks:
      print(file)
      continue
   