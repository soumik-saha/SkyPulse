import streamlit as st
from streamlit_option_menu import option_menu
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO
import base64
import numpy as np
import cv2
import onnxruntime as ort
import yaml
from yaml.loader import SafeLoader
import joblib

with open('data.yaml',mode='r') as f:
    data_yaml = yaml.load(f,Loader=SafeLoader)
    
labels = data_yaml['names']
#print(labels)

# Load the ONNX model , fault model
yolo = ort.InferenceSession('Model/weights/best.onnx')
Fault = joblib.load('Model/Wire_Fault.joblib')
def detect_dents_and_cracks(image):
    # Preprocess the image
    image = image.copy()
    row, col, d = image.shape

    # Convert image to a square image
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Prepare the image blob
    INPUT_WH_YOLO = 640
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)

    # Perform inference using the ONNX model
    yolo.set_providers(['CPUExecutionProvider'])
    yolo_input_name = yolo.get_inputs()[0].name
    yolo_output_name = yolo.get_outputs()[0].name
    preds = yolo.run([yolo_output_name], {yolo_input_name: blob})[0]

    # Debugging information
    st.write(f"Predictions shape: {preds.shape}")
    

    detections = preds[0]
    boxes = []
    confidences = []
    classes = []

    # widht and height of the image (input_image)
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WH_YOLO
    y_factor = image_h/INPUT_WH_YOLO

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detection an object
        if confidence > 0.4:
            class_score = row[5:].max() # maximum probability from 20 objects
            class_id = row[5:].argmax() # get the index position at which max probabilty occur
        
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                # construct bounding from four values
                # left, top, width and height
                left = int((cx - 0.5*w)*x_factor)
                top = int((cy - 0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
            
                box = np.array([left,top,width,height])
            
                # append values into the list
                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)
            
    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()
    for ind in index:
        # extract bounding box
        x,y,w,h = boxes_np[ind]
        bb_conf = int(confidences_np[ind]*100)
        classes_id = classes[ind]
        class_name = labels[classes_id]
    
        text = f'{class_name}: {bb_conf}%'
    
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,255,255),-1)
    
        cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)

    return image

if __name__ == "__main__":

    st.markdown("## Cracks & Dents Detection")
    with st.sidebar:
        selected = option_menu('SkyPulse Aero MultiTool',
            ['Crasks & Dents Detection', 'Wire Fault Detection','About'],
            icons =['activity','activity','info'],default_index=0)
    
    if selected == "Crasks & Dents Detection":
        st.info('Upload Image to check any dents or cracks are present in image', icon="ℹ️")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
        # Read the uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Display the original image
            st.image(image, caption="Original Image", use_column_width=True)

        # Detect dents and cracks
            marked_image = detect_dents_and_cracks(image)

        # Display the marked image
            st.image(marked_image, caption="Image with Marked Locations", use_column_width=True)

    if selected == "Wire Fault Detection" :
        v= st.text_input("Enter Voltage:")
        i = st.text_input("Enter Current:")
        r = st.text_input("Enter Resistance")

        def predict_wire_status(v, i, r):

            # Make prediction for the input data
            prediction = Fault.predict([[v, i, r]])

            if prediction[0] == 1:
                st.write('__Faulty Wire Detected__')
            else:
                st.write('__No Fault__')
        
        if st.button('Predict'):
            st.write("Prediction:")
            result = predict_wire_status(v, i, r)
        
       

    if selected == "About":
        st.markdown("<h2 style='text-align: center;'>ABOUT</h2>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<p style='text-align: center;'>Dual Risks of Faulty Wiring and Structural Damag in aircraft</p>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<h4 style='text-align: center;'>Developed and maintained by</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Soumik Saha</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Pankaj Goel</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Bhagyasri Ramarao</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Ayushi SomeTitle</p>", unsafe_allow_html=True)
        st.markdown("____")
