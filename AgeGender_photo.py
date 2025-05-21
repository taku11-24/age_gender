import cv2 as cv
import sys
import os

# モデルファイルのパスを設定
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
faceProto = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
faceModel = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(MODEL_DIR, "age_deploy.prototxt")
ageModel = os.path.join(MODEL_DIR, "age_net.caffemodel")
genderProto = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
genderModel = os.path.join(MODEL_DIR, "gender_net.caffemodel")

# その他設定
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

def getFaceBox(net, frame, conf_threshold=0.7):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104,117,123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3] * frameWidth)
            y1 = int(detections[0,0,i,4] * frameHeight)
            x2 = int(detections[0,0,i,5] * frameWidth)
            y2 = int(detections[0,0,i,6] * frameHeight)
            bboxes.append([x1,y1,x2,y2])
    return bboxes

# 実行引数チェック
if len(sys.argv) < 2:
    print("Usage: python AgeGender_photo.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

# モデルの読み込み
faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

# 画像読み込み
image = cv.imread(image_path)
if image is None:
    print("Image not found:", image_path)
    sys.exit()

bboxes = getFaceBox(faceNet, image)
if not bboxes:
    print("No face detected")
    sys.exit()

padding = 20
for bbox in bboxes:
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(bbox[2] + padding, image.shape[1] - 1)
    y2 = min(bbox[3] + padding, image.shape[0] - 1)

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        continue

    blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    # 性別と年齢の大まかなグループだけを表示
    print(f"{gender}, {age}")