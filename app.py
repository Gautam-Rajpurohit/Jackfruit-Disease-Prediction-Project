from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
class DenseNetMAML(nn.Module):
    def __init__(self, num_classes=3):
        super(DenseNetMAML, self).__init__()
        # Use 'weights=None' instead of 'pretrained=False'
        self.densenet = models.densenet121(weights=None)  
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)

model = DenseNetMAML()
# Setting weights_only=True in torch.load isn't required for state_dict loading
model.load_state_dict(torch.load('jackfruit_disease_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

class_names = ['Algae Spot Disease ', 'Healthy', 'Black Spot Disease']

# Image transformation function
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Prediction function
def predict(image_path):
    image_tensor = transform_image(image_path)
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    accuracy = torch.softmax(output, dim=1).max().item() * 100  # Get confidence score
    return class_names[predicted.item()], accuracy

# Home route to serve the HTML
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Make prediction
        classification, accuracy = predict(image_path)
        return jsonify({'classification': classification, 'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)
