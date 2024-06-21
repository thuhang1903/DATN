

# class ChangeVector:
#     def __init__(self, facs=None, landmarkChange=None, emotion_label=0):
#         if landmarkChange is None:
#             landmarkChange = []
#         if facs is None:
#             facs = []
#         self.landmarkChange = landmarkChange
#         self.facs = facs
#         self.emotion_label = emotion_label


# class ImageData:
#     def __init__(self, facs=None, landmark=None, emotion=0):
#         if landmark is None:
#             landmark = []
#         if facs is None:
#             facs = {}
#         self.landmark = landmark
#         self.facs = facs
#         self.emotion = emotion


# def process_input_image(image, crop_proportion=0.2, max_diagonal=400):
#     if image.n_channels == 3:
#         image = image.as_greyscale()
#     image = image.crop_to_landmarks_proportion(crop_proportion)
#     d = image.diagonal()
#     if d > max_diagonal:
#         image = image.rescale(float(max_diagonal) / d)
#     return image


# class Emotion:
#     def __init__(self, name, facs_required, criteria):
#         self.name = name
#         self.facs_required = facs_required
#         self.criteria = criteria

#     def criteria(self, facs_input):
#         return True

#     def score(self, facs_input=None):
#         if facs_input is None:
#             facs_input = []
#         if (self.criteria(facs_input) == True):
#             max = 0
#             for required in self.facs_required:
#                 au_count = 0
#                 for facs in facs_input:
#                     if facs in required:
#                         au_count += 1
#                 if au_count / float(len(required)) >= max:
#                     max = au_count / float(len(required))
#             return max
#         else:
#             return 0


# def angry_criteria(facs_input):
#     if (23 in facs_input):
#         return True
#     return False


# def disgus_criteria(facs_input):
#     if (9 in facs_input or 10 in facs_input):
#         return True
#     return False


# def fear_criteria(facs_input):
#     if (1 in facs_input and 2 in facs_input and 3 in facs_input):
#         return True
#     return False


# def surprise_criteria(facs_input):
#     if (1 in facs_input and 2 in facs_input):
#         return True
#     if (5 in facs_input):
#         return True
#     return False


# def sadness_criteria(facs_input):
#     return True


# def happy_criteria(facs_input):
#     if (12 in facs_input):
#         return True
#     return False


# def contempt_criteria(facs_input):
#     if (14 in facs_input):
#         return True
#     return False


# happy = Emotion('happy', [[6, 12]], happy_criteria)
# sadness = Emotion('sadness', [[1, 4, 5], [6, 15], [1, 4, 15]], sadness_criteria)
# surprise = Emotion('surprise', [[1, 2, 5, 26]], surprise_criteria)
# fear = Emotion('fear', [[1, 2, 4, 5, 7, 20, 26]], fear_criteria)
# angry = Emotion('angry', [[4, 5, 7, 23]], angry_criteria)
# disgust = Emotion('disgust', [[9, 15, 16], [10, 15, 16]], disgus_criteria)
# contempt = Emotion('contempt', [[12, 14]], contempt_criteria)

# emotions = [happy, sadness, surprise, fear, angry, contempt, disgust, contempt]


import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import joblib

class ChangeVector:
    def __init__(self, facs=None, landmarkChange=None, emotion_label=0):
        if landmarkChange is None:
            landmarkChange = []
        if facs is None:
            facs = []
        self.landmarkChange = landmarkChange
        self.facs = facs
        self.emotion_label = emotion_label


class ImageData:
    def __init__(self, facs=None, landmark=None, emotion=0):
        if landmark is None:
            landmark = []
        if facs is None:
            facs = {}
        self.landmark = landmark
        self.facs = facs
        self.emotion = emotion


def process_input_image(image, crop_proportion=0.2, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image


class Emotion:
    def __init__(self, name, facs_required, criteria):
        self.name = name
        self.facs_required = facs_required
        self.criteria = criteria

    def criteria(self, facs_input):
        return True

    def score(self, facs_input=None):
        if facs_input is None:
            facs_input = []
        if (self.criteria(facs_input) == True):
            max = 0
            for required in self.facs_required:
                au_count = 0
                for facs in facs_input:
                    if facs in required:
                        au_count += 1
                if au_count / float(len(required)) >= max:
                    max = au_count / float(len(required))
            return max
        else:
            return 0


def angry_criteria(facs_input):
    if (23 in facs_input):
        return True
    return False


def disgus_criteria(facs_input):
    if (9 in facs_input or 10 in facs_input):
        return True
    return False


def fear_criteria(facs_input):
    if (1 in facs_input and 2 in facs_input and 3 in facs_input):
        return True
    return False


def surprise_criteria(facs_input):
    if (1 in facs_input and 2 in facs_input):
        return True
    if (5 in facs_input):
        return True
    return False


def sadness_criteria(facs_input):
    return True


def happy_criteria(facs_input):
    if (12 in facs_input):
        return True
    return False


def contempt_criteria(facs_input):
    if (14 in facs_input):
        return True
    return False


happy = Emotion('happy', [[6, 12]], happy_criteria)
sadness = Emotion('sadness', [[1, 4, 5], [6, 15], [1, 4, 15]], sadness_criteria)
surprise = Emotion('surprise', [[1, 2, 5, 26]], surprise_criteria)
fear = Emotion('fear', [[1, 2, 4, 5, 7, 20, 26]], fear_criteria)
angry = Emotion('angry', [[4, 5, 7, 23]], angry_criteria)
disgust = Emotion('disgust', [[9, 15, 16], [10, 15, 16]], disgus_criteria)
contempt = Emotion('contempt', [[12, 14]], contempt_criteria)

emotions = [happy, sadness, surprise, fear, angry, contempt, disgust, contempt]


# Adding necessary functions

def load_data(image_dir, emotion_dir):
    images = []
    labels = []
    for subject in os.listdir(image_dir):
        subject_path = os.path.join(image_dir, subject)
        if os.path.isdir(subject_path):
            for sequence in os.listdir(subject_path):
                sequence_path = os.path.join(subject_path, sequence)
                if os.path.isdir(sequence_path):
                    for img_file in os.listdir(sequence_path):
                        if img_file.endswith('.png'):
                            img_path = os.path.join(sequence_path, img_file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                emotion_file = get_emotion_path(emotion_dir, img_path)
                                if emotion_file and os.path.exists(emotion_file):
                                    with open(emotion_file, 'r') as f:
                                        label = int(float(f.readline().strip()))
                                        images.append(img)
                                        labels.append(label)
    return images, labels

def preprocess_images(images):
    processed_images = []
    for img in images:
        resized_img = cv2.resize(img, (48, 48))
        flattened_img = resized_img.flatten()
        processed_images.append(flattened_img)
    return np.array(processed_images)

def train_svm(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train_scaled, y_train)
    return svm_model, scaler

def evaluate_model(model, X_test, y_test, scaler=None):
    if scaler:
        X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

def save_model(model, model_path):
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path)

def get_facs_path(facs_dir, image_path):
    parts = image_path.split(os.sep)
    subject = parts[-3]
    sequence = parts[-2]
    facs_path = os.path.join(facs_dir, subject, sequence)
    if os.path.exists(facs_path):
        facs_files = [f for f in os.listdir(facs_path) if f.endswith('_facs.txt')]
        if facs_files:
            return os.path.join(facs_path, facs_files[0])
    return None

def get_emotion_path(emotion_dir, image_path):
    parts = image_path.split(os.sep)
    subject = parts[-3]
    sequence = parts[-2]
    emotion_path = os.path.join(emotion_dir, subject, sequence)
    if os.path.exists(emotion_path):
        emotion_files = [f for f in os.listdir(emotion_path) if f.endswith('_emotion.txt')]
        if emotion_files:
            return os.path.join(emotion_path, emotion_files[0])
    return None
