import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cv2.namedWindow("Finger Count", cv2.WINDOW_NORMAL)

# Initialize variables for smoothing
previous_finger_counts = [0, 0]

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    finger_counts = [0, 0]  # Initialize finger counts for both hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Count the number of open fingers for each hand
            finger_count = 0
            tips = [4, 8, 12, 16, 20]  # Landmark IDs for the tips of five fingers

            for tip_id in tips:
                y = hand_landmarks.landmark[tip_id].y
                x = hand_landmarks.landmark[tip_id].x
                if y < hand_landmarks.landmark[tip_id - 1].y and x > hand_landmarks.landmark[tip_id - 2].x:
                    finger_count += 1

            # Account for the thumb
            thumb_y = hand_landmarks.landmark[4].y
            thumb_x = hand_landmarks.landmark[4].x
            if thumb_y < hand_landmarks.landmark[3].y and thumb_x < hand_landmarks.landmark[2].x:
                finger_count += 1

            # Smoothing: Use the previous frame's result if it's more stable
            if abs(finger_count - previous_finger_counts[results.multi_hand_landmarks.index(hand_landmarks)]) <= 1:
                finger_count = max(0, finger_count)

            # Update finger counts for each hand
            finger_counts[results.multi_hand_landmarks.index(hand_landmarks)] = finger_count
            previous_finger_counts[results.multi_hand_landmarks.index(hand_landmarks)] = finger_count

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for i, finger_count in enumerate(finger_counts):
        cv2.putText(frame, f'Hand {i + 1}: {finger_count}', (10, 30 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Finger Count", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
