from flask import Flask, request, jsonify, render_template
import torch
from models.arc import ConvNet
import torch.nn as nn
from flask_cors import CORS
from torchvision.transforms import transforms
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

classes = ['TUBERCULOSIS', 'HEALTHY', 'PNEUMONIA',]  
model_path = 'models/xray_lung_disease_detection_model.pth'
model = ConvNet(num_classes=len(classes))
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
def predict_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image_tensor = transformer(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = classes[predicted.item()]
    
    return pred_class
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            image_bytes = file.read()
            prediction = predict_image(image_bytes)
            return jsonify({'prediction': prediction})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000,debug=True)