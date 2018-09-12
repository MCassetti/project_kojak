import os
from flask import Flask, render_template, redirect, url_for, request, jsonify,send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    filename = file.filename
    f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(f)
    return redirect(url_for('uploaded_file', filename=filename))

@app.route('/show/<filename>')
def uploaded_file(filename):
    #filename = 'http://127.0.0.1:5000/uploads/' + filename

    return render_template('index.html', filename=filename)

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)

app.run(threaded=True, debug=True)
