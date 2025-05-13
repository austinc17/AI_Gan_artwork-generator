from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import send_file
import os
import random
from flask import send_file, make_response
import subprocess

app = Flask(__name__, static_folder='static')
CORS(app)



@app.route('/api/images/<filename>')
def serve_image(filename):
    filepath=os.path.join('static', 'images', filename)
    response = make_response(send_file(filepath))
    response.headers["conent-disposition"]=f"attachment; filename={filename}"
    return response


from flask import send_from_directory

from flask import send_from_directory

@app.route('/api/download/<filename>')
def download_image(filename):
    return send_from_directory('static/generated', filename, as_attachment=True)









@app.route('/generate-image', methods=['GET'])
def generate_image():
    try:
        print("Running GAN image generator script...")
        subprocess.run(['python3', 'generate_image.py'], check=True)
        print("Image generated successfully.")
        
        # Send the generated image back
        return send_file('static/generated/generated.png', mimetype='image/png')
    except subprocess.CalledProcessError:
        print("Image generation failed.")
        return {'error': 'Image generation failed'}, 500

    


if __name__ == '__main__':
    app.run(debug=True)
