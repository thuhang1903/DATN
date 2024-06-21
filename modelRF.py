import os
import math
import menpo.io as mio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from Models import ImageData
import joblib

# Nhập Thư Viện và Định Nghĩa Các Hàm Hỗ Trợ
# Mục đích: Nhập các thư viện cần thiết cho việc xử lý ảnh, dữ liệu và xây dựng mô hình.
# Chi tiết:
'''os và math: Thư viện chuẩn Python dùng để xử lý các tác vụ hệ thống và các phép toán cơ bản.
menpo.io: Thư viện Menpo dùng để nhập và xử lý hình ảnh.
numpy: Thư viện dùng để xử lý mảng và các phép toán trên mảng.
sklearn: Thư viện scikit-learn dùng để xây dựng mô hình SVM, chuẩn hóa dữ liệu và phân tích thành phần chính.
joblib: Thư viện dùng để lưu và tải mô hình.'''

# Hàm xử lý ảnh
def process(image, crop_proportion=0.3, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image

def landmark_normalize(landmark):
    vector_max = 0
    for p in landmark:
        if (math.sqrt(float(p[0]) * float(p[0]) + float(p[1]) * float(p[1])) > vector_max):
            vector_max = math.sqrt(p[0] * p[0] + p[1] * p[1])
    for p in landmark:
        p[0] = p[0] / float(vector_max)
        p[1] = p[1] / float(vector_max)
    return landmark

def rolate(vector, angle):
    rx = vector[0] * (math.cos(angle)) - vector[1] * (math.sin(angle))
    ry = vector[0] * (math.sin(angle)) + vector[1] * (math.cos(angle))
    return [rx, ry]

def normalize_perk_landmark(landmark_perk, landmark_neutral):
    neutral_center = landmark_neutral[30]
    perk_center = landmark_perk[30]
    move_vector = [neutral_center[0] - perk_center[0], neutral_center[1] - perk_center[1]]
    for lm in landmark_perk:
        lm[0] += move_vector[0]
        lm[1] += move_vector[1]
    scale_neutral = [landmark_neutral[30][0] - landmark_neutral[27][0],
                     landmark_neutral[30][1] - landmark_neutral[27][1]]
    scale_perk = [landmark_perk[30][0] - landmark_perk[27][0], landmark_perk[30][1] - landmark_perk[27][1]]
    ratio = math.sqrt(scale_neutral[0] * scale_neutral[0] + scale_neutral[1] * scale_neutral[1]) / math.sqrt(
        scale_perk[0] * scale_perk[0] + scale_perk[1] * scale_perk[1])
    for lm in landmark_perk:
        lm[0] = (perk_center[0] - lm[0]) * (1 - ratio) + lm[0]
        lm[1] = (perk_center[1] - lm[1]) * (1 - ratio) + lm[1]

    sign_y = scale_perk[0] * scale_neutral[1] - scale_perk[1] * scale_neutral[0]
    sign_x = scale_perk[0] * scale_neutral[0] + scale_perk[1] * scale_neutral[1]
    angle = math.atan2(sign_y, sign_x)
    for lm in landmark_perk:
        tmp_vector = [lm[0] - landmark_perk[30][0], lm[1] - landmark_perk[30][1]]
        new_vector = rolate(tmp_vector, angle)
        lm[0] = new_vector[0] + landmark_perk[30][0]
        lm[1] = new_vector[1] + landmark_perk[30][1]
    return landmark_perk

def coopNormalize(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    merged = set1 + set2
    normalized = normalize(merged, norm='max', axis=0)
    result = []
    for i in range(len1, len1 + len2):
        result.append(normalized[i])
    return result

# import images
path_to_rf_training_database = 'CK+/train-images/**/**/**/*'
training_images = mio.import_images(path_to_rf_training_database, verbose=True)
training_images = training_images.map(process)

path_to_facs = 'CK+/FACS/'
path_to_emotions = 'CK+/Emotion/'

# create training data
# Kiểm tra dữ liệu đã được gán nhãn
labeled_subject = []
emotion_subject = os.listdir(path_to_emotions)
for subject in emotion_subject:
    session = os.listdir(path_to_emotions + "/" + subject)
    for s in session:
        labeled_subject.append(subject + "-" + s)
count = 0
rf_training_data = []
train_subject_name = []
while (count < len(training_images)):
    file_path = str(training_images[count].path).split("\\")
    if len(file_path) < 9:
        print(f"Unexpected file path structure: {file_path}")
        count += 2
        continue
    
    facs_path = path_to_facs + file_path[-3] + '/' + file_path[-2]
    gt_emotion = -1
    emotion_checker = file_path[-3] + "-" + file_path[-2]
    if emotion_checker in labeled_subject:
        emotion_path = path_to_emotions + file_path[-3] + '/' + file_path[-2]
        if len(os.listdir(emotion_path)) != 0:
            emotion_path = emotion_path + '/' + os.listdir(emotion_path)[0]
            with open(emotion_path) as fi:
                for line in fi:
                    if line.split():
                        gt_emotion = int(float(line.split()[0]))
    else:
        print(f"Emotion not found for: {emotion_checker}")

    if not os.path.exists(facs_path) or not os.listdir(facs_path):
        print(f"FACS data missing for: {emotion_checker}")
        count += 2
        continue
    
    facs_path = facs_path + '/' + os.listdir(facs_path)[0]
    with open(facs_path, 'r') as fi:
        data_facs = {}
        tmp = [line.split() for line in fi if line.split()]
        for f in tmp:
            data_facs[str(int(float(f[0])))] = int(float(f[1]))

    landmark = []
    landmark_neutral = training_images[count].landmarks['PTS'].points
    if count + 1 >= len(training_images):
        print(f"Missing paired image for: {file_path}")
        break

    landmark_perk = training_images[count + 1].landmarks['PTS'].points
    landmark_perk = normalize_perk_landmark(landmark_perk, landmark_neutral)

    for i in range(0, 68):
        landmark.append([landmark_perk[i][0] - landmark_neutral[i][0], landmark_perk[i][1] - landmark_neutral[i][1]])

    rf_training_data.append(ImageData(data_facs, landmark, gt_emotion))
    train_subject_name.append([emotion_checker, gt_emotion])
    count += 2

# create testing data
rf_testing_data = []
path_to_rf_testing_database = "CK+/test-images/**/**/**/*"
testing_images = mio.import_images(path_to_rf_testing_database, verbose=True)
testing_images = testing_images.map(process)

count = 0
test_subject_name = []
while (count < len(testing_images)):
    file_path = str(testing_images[count].path).split("\\")
    if len(file_path) < 9:
        print(f"Unexpected file path structure: {file_path}")
        count += 2
        continue

    facs_path = path_to_facs + file_path[-3] + '/' + file_path[-2]
    gt_emotion = -1
    emotion_checker = file_path[-3] + "-" + file_path[-2]
    if emotion_checker in labeled_subject:
        emotion_path = path_to_emotions + file_path[-3] + '/' + file_path[-2]
        if len(os.listdir(emotion_path)) != 0:
            emotion_path = emotion_path + '/' + os.listdir(emotion_path)[0]
            with open(emotion_path) as fi:
                for line in fi:
                    if line.split():
                        gt_emotion = int(float(line.split()[0]))
    else:
        print(f"Emotion not found for: {emotion_checker}")

    if not os.path.exists(facs_path) or not os.listdir(facs_path):
        print(f"FACS data missing for: {emotion_checker}")
        count += 2
        continue

    facs_path = facs_path + '/' + os.listdir(facs_path)[0]
    with open(facs_path, 'r') as fi:
        data_facs = {}
        tmp = [line.split() for line in fi if line.split()]
        for f in tmp:
            data_facs[str(int(float(f[0])))] = int(float(f[1]))

    landmark = []
    landmark_neutral = testing_images[count].landmarks['PTS'].points
    if count + 1 >= len(testing_images):
        print(f"Missing paired image for: {file_path}")
        break

    landmark_perk = testing_images[count + 1].landmarks['PTS'].points
    landmark_perk = normalize_perk_landmark(landmark_perk, landmark_neutral)

    for i in range(0, 68):
        landmark.append([landmark_perk[i][0] - landmark_neutral[i][0], landmark_perk[i][1] - landmark_neutral[i][1]])

    rf_testing_data.append(ImageData(data_facs, landmark, gt_emotion))
    test_subject_name.append([emotion_checker, gt_emotion])
    count += 2

# - build training data
x_training = []
y_label = []
for data in rf_training_data:
    if data.emotion != -1:
        tmp = []
        for vector in data.landmark:
            tmp.append(vector[0])
            tmp.append(vector[1])
        x_training.append(tmp)
        y_label.append(data.emotion)

models = []
clf = RandomForestClassifier(max_depth=8, random_state=0)
model = clf.fit(normalize(x_training, norm='max', axis=0), y_label)
models.append(model)
filename = 'models_RF.joblib'
joblib.dump(models, filename)

x_testing = []
y_label = []
for data in rf_testing_data:
    if data.emotion != -1:
        tmp = []
        for vector in data.landmark:
            tmp.append(vector[0])
            tmp.append(vector[1])
        x_testing.append(tmp)
        y_label.append(data.emotion)

# Predictions and evaluation
y_pred = clf.predict(coopNormalize(x_training, x_testing))

print("Dự đoán nhãn cảm xúc:")
print(y_pred)
print("Nhãn cảm xúc thực tế:")
print(y_label)
print("Độ chính xác")
print(clf.score(coopNormalize(x_training, x_testing), y_label))

# Print precision, recall, and f1-score
print("\nClassification Report:")
print(classification_report(y_label, y_pred))
