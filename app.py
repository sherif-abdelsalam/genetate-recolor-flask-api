from flask import Flask, request,send_file,jsonify # type:ignore 
from flask_cors import CORS # type:ignore
from color_assignment import assign_colors
from PIL import Image
import io
import numpy as np
import cv2

from color_processing import apply_dominant_color
from color_palette import preprocess_image, apply_mask, plot_extended_palette, U2NET_MODEL

app = Flask(__name__)
CORS(app)


### end point for generating color palette
######################################################################

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Access the raw image data from the request body
        image_data = request.data
        
        if not image_data:
            return jsonify({'error': 'No image data found in the request'}), 400

        # Convert the raw image data to a PIL Image
        image = Image.open(io.BytesIO(image_data))
        # Process the image as before
        image_size = 256
        input_array = preprocess_image(image, image_size)
        y_pred = U2NET_MODEL.predict(input_array)
        predicted_mask = y_pred[0]
        predicted_mask = cv2.resize(predicted_mask, (image_size, image_size))
        original_image = np.array(image.resize((image_size, image_size)))


        focal_object = apply_mask(original_image, predicted_mask)
        extended_palette = plot_extended_palette(focal_object, n_colors=3, n_new_colors=2)

        return jsonify({
            'color_palette': extended_palette,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# curl -X POST http://localhost:5150/api/process-image --data-binary "@/home/sherif-hodhod/Desktop/bag_design.jpg" -H "Content-Type: image/jpeg"

### end point for color assignment
######################################################################

@app.route('/assign_colors', methods=['POST'])
def assign_colors_endpoint():
    try:
        data = request.json
        if not data or 'layers' not in data or 'palette' not in data:
            return jsonify({"error": "Invalid data"}), 400
        
        layers = data['layers']
        palette = data['palette']
        
        assignment = assign_colors(palette, layers)
        return jsonify(assignment)
    
    except Exception as e:
        print("Error occurred:", str(e))  # Print error message
        return jsonify({"error": "Internal server error", "message": str(e)}), 500


###################################################################################################33

@app.route('/apply', methods=['POST'])
def upload_file():

    if 'file' not in request.files or 'color' not in request.form:
        return jsonify({'error': 'No file or color provided'}), 400

    file = request.files['file']
    color = request.form['color']
    print(color)
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image file
        image = Image.open(file.stream)
        image_rgb = np.array(image.convert('RGB'))
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Parse the color
        color = [int(c) for c in color.split(',')]
        if len(color) != 3:
            return jsonify({'error': 'Invalid color format. Use R,G,B'}), 400

        # Apply dominant color
        image_rgb_modified = apply_dominant_color(image_rgb, color, scale_factor=1.0)
        
        # Convert the result to a format Flask can send
        _, buffer = cv2.imencode('.png', image_rgb_modified)
        image_stream = io.BytesIO(buffer)

        return send_file(image_stream, mimetype='image/png', as_attachment=True, download_name='modified_image.png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

############################################################

from color_palette_ai_version import generate_response

@app.route('/process_prompt', methods=['POST'])
def process_prompt():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract the string from the JSON data
    input_string = data.get('input_string', '')

    generated_response = generate_response(input_string)

    # Prepare the response
    result = {
        'received_string': input_string,
        'palette': generated_response,  # The AI-generated response
        'length': len(input_string)  # Length of the input string
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
