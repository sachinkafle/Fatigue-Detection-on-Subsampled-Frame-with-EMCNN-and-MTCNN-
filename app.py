import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from utils.model_utils import FatigueDetection  # Import your fatigue detection class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
ALLOWED_EXTENSIONS = {'mov', 'mp4', 'avi'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Call fatigue detection logic
            fatigue_detector = FatigueDetection(video_path)
            result = fatigue_detector.calculate()

            # Determine the fatigue status based on the result
            if result:
                fatigue_status = "Fatigue Detected"
            else:
                fatigue_status = "No Fatigue Detected"
            
            # Render the result.html template and pass the fatigue status
            return render_template('result.html', fatigue_status=fatigue_status)
    
    # Render the index.html template for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
