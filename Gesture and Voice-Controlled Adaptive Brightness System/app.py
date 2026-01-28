# import cv2
# import mediapipe as mp
# import numpy as np
# import screen_brightness_control as sbc
# import speech_recognition as sr
# import datetime

# # Initialize webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# # Global Variables
# brightness = 50
# adaptive_brightness_enabled = True
# night_mode_enabled = False
# grayscale_mode_enabled = False
# extra_dim_mode_enabled = False
# invert_mode_enabled = False
# min_distance, max_distance = 10, 200
# recognizer = sr.Recognizer()

# def is_night_time():
#     current_hour = datetime.datetime.now().hour
#     return current_hour >= 19 or current_hour <= 6

# def apply_night_filter(frame):
#     overlay = np.full(frame.shape, (30, 30, 100), dtype=np.uint8)
#     return cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

# def find_distance(p1, p2):
#     return np.linalg.norm(np.array(p1) - np.array(p2))

# def count_fingers(hand_landmarks):
#     tips = [4, 8, 12, 16, 20]
#     count = sum(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y for tip in tips[1:])
#     thumb_tip, thumb_ip = hand_landmarks.landmark[4], hand_landmarks.landmark[3]
#     if thumb_tip.x < thumb_ip.x:
#         count += 1
#     return count

# def is_thumbs_up(hand_landmarks):
#     thumb_tip, thumb_ip, index_tip = hand_landmarks.landmark[4], hand_landmarks.landmark[3], hand_landmarks.landmark[8]
#     return thumb_tip.y < thumb_ip.y and thumb_tip.y < index_tip.y

# def update_brightness(brightness_level):
#     brightness_level = max(0, min(100, brightness_level))
#     try:
#         sbc.set_brightness(brightness_level)
#     except Exception as e:
#         print("Error updating brightness:", e)

# def get_adaptive_brightness(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)
#     return int(np.interp(avg_brightness, [0, 255], [0, 100]))

# def recognize_voice_command():
#     global brightness, adaptive_brightness_enabled, night_mode_enabled, grayscale_mode_enabled, extra_dim_mode_enabled, invert_mode_enabled
#     with sr.Microphone() as source:
#         print("Listening for command... (speak clearly)")
#         recognizer.adjust_for_ambient_noise(source, duration=1)
#         try:
#             audio = recognizer.listen(source, timeout=7, phrase_time_limit=5)
#             command = recognizer.recognize_google(audio).lower()
#             print("Recognized:", command)

#             if "increase brightness" in command:
#                 brightness = min(brightness + 10, 100)
#             elif "decrease brightness" in command:
#                 brightness = max(brightness - 10, 0)
#             elif "set brightness" in command:
#                 try:
#                     level = int(''.join(filter(str.isdigit, command)))
#                     if 0 <= level <= 100:
#                         brightness = level
#                 except ValueError:
#                     print("Could not extract brightness level.")
#             elif "enable adaptive mode" in command:
#                 adaptive_brightness_enabled = True
#             elif "disable adaptive mode" in command:
#                 adaptive_brightness_enabled = False
#             elif "enable night mode" in command:
#                 night_mode_enabled = True
#             elif "disable night mode" in command:
#                 night_mode_enabled = False
#             elif "enable bedtime mode" in command:
#                 grayscale_mode_enabled = True
#             elif "disable bedtime mode" in command:
#                 grayscale_mode_enabled = False
#             elif "enable extra dim mode" in command:
#                 extra_dim_mode_enabled = True
#                 brightness = 10
#                 update_brightness(brightness)
#             elif "disable extra dim mode" in command:
#                 extra_dim_mode_enabled = False
#             elif "enable color inversion" in command:
#                 invert_mode_enabled = True
#             elif "disable color inversion" in command:
#                 invert_mode_enabled = False

#             update_brightness(brightness)

#         except sr.WaitTimeoutError:
#             print("Listening timed out. No speech detected.")
#         except sr.UnknownValueError:
#             print("Could not understand the audio. Try again.")
#         except sr.RequestError as e:
#             print(f"Could not request results; {e}")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)

#     if adaptive_brightness_enabled:
#         brightness = get_adaptive_brightness(frame)

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             finger_count = count_fingers(hand_landmarks)
#             thumb_tip, index_tip = hand_landmarks.landmark[4], hand_landmarks.landmark[8]
#             x1, y1, x2, y2 = int(thumb_tip.x * w), int(thumb_tip.y * h), int(index_tip.x * w), int(index_tip.y * h)
#             distance = find_distance((x1, y1), (x2, y2))

#             if distance < 20 or finger_count == 0:
#                 brightness = 0
#             elif finger_count == 5:
#                 brightness = 90
#             elif is_thumbs_up(hand_landmarks):
#                 brightness = 50
#             else:
#                 brightness = int(np.interp(distance, [min_distance, max_distance], [0, 100]))

#             if not extra_dim_mode_enabled:
#                 update_brightness(brightness)

#     if night_mode_enabled:
#         brightness = min(brightness, 40)
#         frame = apply_night_filter(frame)
#         cv2.putText(frame, "Night Mode ON", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
#     else:
#         cv2.putText(frame, "Day Mode", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

#     if grayscale_mode_enabled:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#         cv2.putText(frame, "Grayscale Mode ON", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 255), 2)

#     if extra_dim_mode_enabled:
#         update_brightness(10)
#         cv2.putText(frame, "Extra Dim Mode ON", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

#     if invert_mode_enabled:
#         frame = cv2.bitwise_not(frame)
#         cv2.putText(frame, "Color Inversion ON", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 150), 2)

#     cv2.putText(frame, f"Brightness: {brightness}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     mode_text = "Adaptive ON" if adaptive_brightness_enabled else "Adaptive OFF"
#     cv2.putText(frame, mode_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Brightness Control", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('a'):
#         adaptive_brightness_enabled = not adaptive_brightness_enabled
#     elif key == ord('n'):
#         night_mode_enabled = not night_mode_enabled
#     elif key == ord('g'):
#         grayscale_mode_enabled = not grayscale_mode_enabled
#     elif key == ord('d'):
#         extra_dim_mode_enabled = not extra_dim_mode_enabled
#     elif key == ord('i'):
#         invert_mode_enabled = not invert_mode_enabled
#     elif key == ord('v'):
#         recognize_voice_command()

# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
import speech_recognition as sr
import datetime

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

brightness = 50
adaptive_brightness_enabled = True
night_mode_enabled = False
grayscale_mode_enabled = False
extra_dim_mode_enabled = False
invert_mode_enabled = False
face_mode_enabled = False  
min_distance, max_distance = 10, 200
recognizer = sr.Recognizer()

def is_night_time():
    current_hour = datetime.datetime.now().hour
    return current_hour >= 19 or current_hour <= 6

def apply_night_filter(frame):
    overlay = np.full(frame.shape, (30, 30, 100), dtype=np.uint8)
    return cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

def find_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    count = sum(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y for tip in tips[1:])
    thumb_tip, thumb_ip = hand_landmarks.landmark[4], hand_landmarks.landmark[3]
    if thumb_tip.x < thumb_ip.x:
        count += 1
    return count

def is_thumbs_up(hand_landmarks):
    thumb_tip, thumb_ip, index_tip = hand_landmarks.landmark[4], hand_landmarks.landmark[3], hand_landmarks.landmark[8]
    return thumb_tip.y < thumb_ip.y and thumb_tip.y < index_tip.y

def update_brightness(brightness_level):
    brightness_level = max(0, min(100, brightness_level))
    try:
        sbc.set_brightness(brightness_level)
    except Exception as e:
        print("Error updating brightness:", e)

def get_adaptive_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return int(np.interp(avg_brightness, [0, 255], [0, 100]))

def get_face_distance_brightness(results, width):
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            face_width = bbox.width * width
            return int(np.interp(face_width, [50, 250], [100, 30]))
    return None

def recognize_voice_command():
    global brightness, adaptive_brightness_enabled, night_mode_enabled, grayscale_mode_enabled
    global extra_dim_mode_enabled, invert_mode_enabled, face_mode_enabled
    with sr.Microphone() as source:
        print("Listening for command... (speak clearly)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=5)
            command = recognizer.recognize_google(audio).lower()
            print("Recognized:", command)

            if "increase brightness" in command:
                brightness = min(brightness + 10, 100)
            elif "decrease brightness" in command:
                brightness = max(brightness - 10, 0)
            elif "set brightness" in command:
                try:
                    level = int(''.join(filter(str.isdigit, command)))
                    if 0 <= level <= 100:
                        brightness = level
                except ValueError:
                    print("Could not extract brightness level.")
            elif "enable adaptive mode" in command:
                adaptive_brightness_enabled = True
            elif "disable adaptive mode" in command:
                adaptive_brightness_enabled = False
            elif "enable night mode" in command:
                night_mode_enabled = True
            elif "disable night mode" in command:
                night_mode_enabled = False
            elif "enable bedtime mode" in command:
                grayscale_mode_enabled = True
            elif "disable bedtime mode" in command:
                grayscale_mode_enabled = False
            elif "enable extra dim mode" in command:
                extra_dim_mode_enabled = True
                brightness = 10
                update_brightness(brightness)
            elif "disable extra dim mode" in command:
                extra_dim_mode_enabled = False
            elif "enable color inversion" in command:
                invert_mode_enabled = True
            elif "disable color inversion" in command:
                invert_mode_enabled = False
            elif "enable face mode" in command:
                face_mode_enabled = True
            elif "disable face mode" in command:
                face_mode_enabled = False

            update_brightness(brightness)

        except sr.WaitTimeoutError:
            print("Listening timed out.")
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if adaptive_brightness_enabled:
        brightness = get_adaptive_brightness(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb)
    face_result = face_detection.process(rgb)

    if face_mode_enabled:
        face_brightness = get_face_distance_brightness(face_result, w)
        if face_brightness is not None:
            brightness = face_brightness
            update_brightness(brightness)
            cv2.putText(frame, f"Face Brightness: {brightness}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)

    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(hand_landmarks)
            thumb_tip, index_tip = hand_landmarks.landmark[4], hand_landmarks.landmark[8]
            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
            distance = find_distance((x1, y1), (x2, y2))

            if distance < 20 or finger_count == 0:
                brightness = 0
            elif finger_count == 5:
                brightness = 90
            elif is_thumbs_up(hand_landmarks):
                brightness = 50
            else:
                brightness = int(np.interp(distance, [min_distance, max_distance], [0, 100]))

            if not extra_dim_mode_enabled and not face_mode_enabled:
                update_brightness(brightness)

    if night_mode_enabled:
        brightness = min(brightness, 40)
        frame = apply_night_filter(frame)
        cv2.putText(frame, "Night Mode ON", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)

    if grayscale_mode_enabled:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "Grayscale Mode ON", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 255), 2)

    if extra_dim_mode_enabled:
        update_brightness(10)
        cv2.putText(frame, "Extra Dim Mode ON", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

    if invert_mode_enabled:
        frame = cv2.bitwise_not(frame)
        cv2.putText(frame, "Color Inversion ON", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 150), 2)

    cv2.putText(frame, f"Brightness: {brightness}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    mode_text = "Adaptive ON" if adaptive_brightness_enabled else "Adaptive OFF"
    cv2.putText(frame, mode_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Brightness Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        adaptive_brightness_enabled = not adaptive_brightness_enabled
    elif key == ord('n'):
        night_mode_enabled = not night_mode_enabled
    elif key == ord('g'):
        grayscale_mode_enabled = not grayscale_mode_enabled
    elif key == ord('d'):
        extra_dim_mode_enabled = not extra_dim_mode_enabled
    elif key == ord('i'):
        invert_mode_enabled = not invert_mode_enabled
    elif key == ord('f'):
        face_mode_enabled = not face_mode_enabled
    elif key == ord('v'):
        recognize_voice_command()

cap.release()
cv2.destroyAllWindows()