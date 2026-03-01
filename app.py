import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import predictor

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for the React frontend

# Configure Upload Folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
HEATMAP_FOLDER = os.path.join("static", "heatmaps")

# Ensure upload and heatmap directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER

# Route to serve created heatmaps to React frontend
@app.route('/static/heatmaps/<filename>')
def serve_heatmap(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename)

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "message": "Brain Tumor Detection API is running. Please access the React dashboard on port 3000.",
        "endpoints": {
            "predict": "/api/predict [POST]"
        }
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Get dict response from predictor
            result_dict = predictor.check(filepath)
            
            # Use the request's host for the heatmap URL to handle different environments
            host = request.host
            heatmap_url = f"http://{host}/{result_dict['heatmap_path']}"

            # Formulate API Response
            return jsonify({
                "predicted_class": result_dict["predicted_class"],
                "confidence": result_dict["confidence"],
                "probabilities": result_dict["probabilities"],
                "heatmap_url": heatmap_url
            }), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4555) # Custom API port to decouple from default web servers
