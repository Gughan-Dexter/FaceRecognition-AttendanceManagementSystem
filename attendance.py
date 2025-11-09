import os, csv, cv2, numpy as np, face_recognition
from PIL import Image
from datetime import datetime

if input("Username: ") != "admin" or input("Password: ") != "1234":
    print("Access denied."); exit()
print("Login successful.\n")

known_faces, known_names = [], []
for f in os.listdir("photo_clean"):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        img = np.array(Image.open(os.path.join("photo_clean", f)))
        enc = face_recognition.face_encodings(img)
        if enc:
            known_faces.append(enc[0])
            known_names.append(os.path.splitext(f)[0])

if not known_faces:
    print("No registered faces found."); exit()

log_file = f"attendance_{datetime.now():%Y-%m-%d}.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Time"])

cam = cv2.VideoCapture(0)
print("Camera active. Press 's' to quit.")

while True:
    ok, frame = cam.read()
    if not ok: break

    locs = face_recognition.face_locations(frame)
    encs = face_recognition.face_encodings(frame, locs)

    name, color = "Detecting...", (255, 255, 255)
    for (t, r, b, l), face in zip(locs, encs):
        match = face_recognition.compare_faces(known_faces, face, 0.5)
        if True in match:
            name = known_names[match.index(True)]
            color = (0, 255, 0)
            with open(log_file, "r") as f:
                marked = [line.split(",")[0] for line in f]
            if name not in marked:
                with open(log_file, "a", newline="") as f:
                    csv.writer(f).writerow([name, datetime.now().strftime("%H:%M:%S")])
                print(f"Marked: {name}")
        else:
            name, color = "Unrecognized", (0, 0, 255)

        cv2.rectangle(frame, (l, t), (r, b), color, 2)

    h, w = frame.shape[:2]
    size = cv2.getTextSize(name, cv2.FONT_HERSHEY_TRIPLEX, 1, 2)[0]
    x, y = (w - size[0]) // 2, 50
    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cam.release()
cv2.destroyAllWindows()
