from flask import Flask, request, render_template_string
import cv2 as cv
import numpy as np

app = Flask(__name__)

# モデルファイルパス
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

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

HTML_TEMPLATE = '''
<!doctype html>
<html>
<head>
  <title>Age & Gender Detection</title>
</head>
<body>
  <h2>Upload a photo to detect age and gender</h2>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="file" required>
    <input type="submit" value="Detect">
  </form>

  {% if results %}
    <h3>Results:</h3>
    <ul>
    {% for r in results %}
      <li>{{ r.gender }}, {{ r.age }}</li>
    {% endfor %}
    </ul>
  {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def detect():
    results = []
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            if img is None:
                return "Could not read the image", 400

            bboxes = getFaceBox(faceNet, img)
            if not bboxes:
                return render_template_string(HTML_TEMPLATE, results=[])

            padding = 20
            for bbox in bboxes:
                x1 = max(0, bbox[0] - padding)
                y1 = max(0, bbox[1] - padding)
                x2 = min(bbox[2] + padding, img.shape[1] - 1)
                y2 = min(bbox[3] + padding, img.shape[0] - 1)

                if x2 <= x1 or y2 <= y1:
                    continue

                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]

                results.append({
                    'gender': gender,
                    'age': age
                })

    return render_template_string(HTML_TEMPLATE, results=results)



if __name__ == "__main__":
    app.run(debug=True)
