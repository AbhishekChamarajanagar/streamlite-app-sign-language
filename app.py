import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label dictionary for predictions
labels_dict = {i: chr(48 + i // 2) if i < 20 else chr(65 + (i - 20) // 2) for i in range(74)}

def process_frame(img):
    H, W, _ = img.shape
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        if len(data_aux) == model.n_features_in_:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return img

# Streamlit UI
st.title("Hand Gesture Recognition using Streamlit")
st.write("Show a hand gesture, and the app will predict the character!")

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture frame. Exiting...")
        break
    processed_frame = process_frame(frame)
    stframe.image(processed_frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
