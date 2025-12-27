# face_system_v2.py
# Menu-driven Face Capture + LBPH Train/Save/Load + Real-time Recognition
# Auto-loads model (lbph_model.yml) and labels (labels.json) if present.

import os, sys, json
import cv2
import numpy as np

# ---- Config ----
DATASET_PATH = "dataset"
HAAR_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "lbph_model.yml"
LABELS_PATH = "labels.json"
IMG_SIZE = (200, 200)
SAMPLES_PER_PERSON = 10
UNKNOWN_THRESHOLD = 75  # Lower = stricter (Unknown more often)

# ---- Globals (filled at runtime) ----
recognizer = None
label_map = None  # {int_label: "name"}

# ---- Utils ----
def ensure_paths():
    if not os.path.exists(HAAR_PATH):
        print(f"[ERROR] Haar cascade not found at: {HAAR_PATH}")
        sys.exit(1)
    os.makedirs(DATASET_PATH, exist_ok=True)

def load_haar():
    return cv2.CascadeClassifier(HAAR_PATH)

def save_labels(_label_map):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(_label_map, f, ensure_ascii=False, indent=2)

def load_labels():
    if not os.path.exists(LABELS_PATH):
        return None
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}

def try_autoload_model():
    """Auto-load model/labels if files exist; return (recognizer, label_map) or (None, None)."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return None, None
    try:
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(MODEL_PATH)
        labels = load_labels()
        if not labels:
            return None, None
        print(f"[INFO] Auto-loaded model '{MODEL_PATH}' and labels '{LABELS_PATH}'.")
        return rec, labels
    except Exception as e:
        print(f"[WARN] Failed to auto-load model: {e}")
        return None, None

def iter_images(folder):
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    for root, _, files in os.walk(folder):
        for fn in files:
            if os.path.splitext(fn.lower())[1] in valid_ext:
                yield os.path.join(root, fn)

# ---- Features ----
def menu_capture_faces(face_cascade):
    """Capture SAMPLES_PER_PERSON faces into dataset/<name>/ via webcam."""
    name = input("Enter name for dataset folder (e.g., mohammed): ").strip().lower()
    if not name:
        print("[ERROR] Empty name.")
        return
    save_dir = os.path.join(DATASET_PATH, name)
    os.makedirs(save_dir, exist_ok=True)

    # Determine next index (supports adding more later)
    existing = [int(os.path.splitext(f)[0]) for f in os.listdir(save_dir)
                if os.path.splitext(f)[0].isdigit()]
    start_idx = max(existing) + 1 if existing else 1

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print(f"[INFO] Capture for '{name}'. Press 'C' to save, 'Q' to cancel.")
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame grab failed.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)

            cv2.putText(frame, f"Captured: {count}/{SAMPLES_PER_PERSON} - Press 'C'",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Capture Faces", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and len(faces) > 0:
                # take largest face (more robust if multiple)
                (x, y, w, h) = max(faces, key=lambda r: r[2]*r[3])
                roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(roi, IMG_SIZE)
                idx = start_idx + count
                out_path = os.path.join(save_dir, f"{idx}.jpg")
                cv2.imwrite(out_path, face_resized)
                count += 1
                print(f"[INFO] Saved {os.path.relpath(out_path)}")

                if count >= SAMPLES_PER_PERSON:
                    print(f"[INFO] Finished capturing {SAMPLES_PER_PERSON} images for '{name}'.")
                    break

            elif key == ord('q'):
                print("[INFO] Capture cancelled.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def train_and_save_model():
    """Train LBPH from all dataset/* folders, save model + labels."""
    persons = [d for d in os.listdir(DATASET_PATH)
               if os.path.isdir(os.path.join(DATASET_PATH, d))]
    if not persons:
        print("[ERROR] No people found in dataset/. Use option [1] to capture faces first.")
        return None, None

    persons.sort()
    label_map_local = {}
    faces, labels = [], []
    next_label = 0

    print("[INFO] Gathering training images…")
    for person in persons:
        folder = os.path.join(DATASET_PATH, person)
        imgs = list(iter_images(folder))
        if not imgs:
            print(f"[WARN] No images in {folder}, skipping.")
            continue
        label_map_local[next_label] = person
        for p in imgs:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Unreadable image skipped: {p}")
                continue
            if img.shape != IMG_SIZE:
                # Normalize size (capture already saves resized, but just in case)
                img = cv2.resize(img, IMG_SIZE)
            faces.append(img)
            labels.append(next_label)
        next_label += 1

    if len(faces) == 0:
        print("[ERROR] No valid training images found.")
        return None, None

    faces_np = np.array(faces, dtype=np.uint8)
    labels_np = np.array(labels, dtype=np.int32)

    print("[INFO] Training LBPH (this is quick)…")
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(faces_np, labels_np)

    # Save model + labels
    rec.write(MODEL_PATH)
    save_labels(label_map_local)
    print(f"[INFO] Saved model to '{MODEL_PATH}' and labels to '{LABELS_PATH}'.")
    return rec, label_map_local

def start_recognition(rec, labels, face_cascade):
    """Run webcam recognition using loaded/ trained model."""
    if rec is None or labels is None:
        print("[ERROR] Model not loaded. Use [2] to train or ensure auto-load worked.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Recognition running. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, IMG_SIZE)
                label, confidence = rec.predict(roi)
                # LBPH: lower confidence = better match
                if confidence <= UNKNOWN_THRESHOLD:
                    name = labels.get(label, "Unknown")
                    color = (0, 200, 0)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("LBPH Recognition", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def force_reset_model():
    """Delete saved model + labels (does not touch dataset)."""
    removed_any = False
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH); removed_any = True
        print(f"[INFO] Deleted {MODEL_PATH}")
    if os.path.exists(LABELS_PATH):
        os.remove(LABELS_PATH); removed_any = True
        print(f"[INFO] Deleted {LABELS_PATH}")
    if not removed_any:
        print("[INFO] No saved model/labels were present.")
    else:
        print("[INFO] Reset complete. You can retrain with option [2].")

# ---- Menu ----
def main():
    global recognizer, label_map
    ensure_paths()
    face_cascade = load_haar()

    # AUTO-LOAD at startup (no prompt)
    recognizer, label_map = try_autoload_model()

    while True:
        print("\n=============================")
        print("  FACE RECOGNITION SYSTEM V2")
        print("=============================")
        print("[1] Capture new face dataset")
        print("[2] Train & Save Model (overwrites)")
        print("[3] Start Recognition (auto-loads if available)")
        print("[4] Force Reset Saved Model")
        print("[Q] Quit")
        choice = input("\nChoose an option: ").strip().lower()

        if choice == "1":
            menu_capture_faces(face_cascade)

        elif choice == "2":
            recognizer, label_map = train_and_save_model()

        elif choice == "3":
            # If not loaded/trained in-memory, try auto-load again
            if recognizer is None or label_map is None:
                recognizer, label_map = try_autoload_model()
            if recognizer is None or label_map is None:
                print("[WARN] No saved model found. Train first with [2].")
            else:
                start_recognition(recognizer, label_map, face_cascade)

        elif choice == "4":
            force_reset_model()
            recognizer, label_map = None, None

        elif choice == "q":
            print("Goodbye!")
            break
        else:
            print("[ERROR] Invalid option.")

if __name__ == "__main__":
    # Pre-flight check: ensure contrib build is present
    try:
        _ = cv2.face.LBPHFaceRecognizer_create
    except Exception:
        print("[ERROR] OpenCV contrib not available. Install inside your conda env:")
        print("       pip install opencv-contrib-python")
        sys.exit(1)
    main()
