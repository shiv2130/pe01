import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('vehicledetectmodel.h5')

class_names = ['City Car', 'Big Truck', 'Multi Purpose Vehicle', 'Sedan', 'sports', 'traffic', 'Truck', 'Van']  # example classes


image_path = '/Users/shivanshuprakash/Desktop/bigproject/Smart-Adaptive-Traffic-Management-System/traffic.jpg'
image = cv2.imread(image_path)

bounding_boxes = [
    (50, 50, 200, 150)  # box 1: x=50, y=50, width=200, height=150
]

input_size = 48

for (x, y, w, h) in bounding_boxes:
    vehicle_img = image[y:y+h, x:x+w]
    vehicle_resized = cv2.resize(vehicle_img, (input_size, input_size))
    vehicle_normalized = vehicle_resized / 255.0  # normalize pixel values
    vehicle_input = np.expand_dims(vehicle_normalized, axis=0)  # add batch dimension

    preds = model.predict(vehicle_input)
    predicted_class = class_names[np.argmax(preds)]

    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cv2.imshow('Vehicle Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.jpg', image)



