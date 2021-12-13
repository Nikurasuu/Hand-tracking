import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    image_hight, image_width, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #print('hand_landmarks:', hand_landmarks)

            Index_x = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * image_width
            Index_y = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * image_hight

            Middle_x = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
            Middle_y = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y * image_hight

            Thumb_x = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x * image_width 
            Thumb_y = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].y * image_width - 60

            print(
                f'INDEX_FINGER_TIP: (',
                f'{int(Index_x)}, '
                f'{int(Index_y)})'
            )

            print(
                f'MIDDLE_FINGER_TIP: (',
                f'{int(Middle_x)}, '
                f'{int(Middle_y)})'
            )

            print(
                f'THUMB_TIP: (',
                f'{int(Thumb_x)}, '
                f'{int(Thumb_y)})'
            )

            cv2.circle(img, (int(Index_x),int(Index_y)), 5, (255, 0, 0), thickness=3, lineType=8,shift=0)
            cv2.circle(img, (int(Middle_x),int(Middle_y)), 5, (255, 50, 50), thickness=3, lineType=8,shift=0)
            cv2.circle(img, (int(Thumb_x),int(Thumb_y)), 5, (255, 100, 100), thickness=3, lineType=8,shift=0)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

    

    cv2.imshow("Image", img)
    cv2.waitKey(1)