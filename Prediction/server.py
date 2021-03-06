import os
import zipfile
from ProductionModel import suicide
from flask import Flask, render_template, request, redirect, url_for ,send_file

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html') 

upload_dir = os.getcwd() + "/upload"

@app.route("/upload", methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if len(os.listdir(upload_dir)) != 0:
            for fname in os.listdir(upload_dir):
                fpath = os.path.join(upload_dir, fname)
                os.remove(fpath)
        upload_file = request.files.get('file', None)
        if upload_file.filename != '':
            upload_file.save(os.path.join(upload_dir, upload_file.filename))
            return redirect(url_for('download'))
    return render_template('loading.html')

@app.route("/input")
def input_page():
    return render_template('upload.html')

@app.route("/download")
def download():
    suicide()
    return render_template('download.html')

@app.route("/load")
def loading_page():
    return render_template('loading.html')

@app.route("/about")
def about_page():
    return render_template('about.html')

@app.route("/download_result", methods=['GET', 'POST'])
def download_result():
    zipfolder = zipfile.ZipFile('output_result.zip', 'w', compression=zipfile.ZIP_STORED)
    for root,dirs, files in os.walk('output_result'):
        for file in files:
            zipfolder.write("output_result/"+file)
        zipfolder.close()
    return send_file('output_result.zip', as_attachment=True)

@app.route("/download_graph", methods=['GET', 'POST'])
def download_graph():
    zipfolder = zipfile.ZipFile('output_graph.zip', 'w', compression=zipfile.ZIP_STORED)
    for root,dirs, files in os.walk('output_graph'):
        for file in files:
            zipfolder.write("output_graph/"+file)
        zipfolder.close()
    return send_file('output_graph.zip', as_attachment=True)

@app.route('/login')
def login():
    return redirect('http://localhost:3000/login')

@app.route('/signup')
def signup():
    return redirect('http://localhost:3000/signup')

@app.route('/logout')
def logout():
    return redirect('http://localhost:3000/logout')

@app.route('/forgotpassword')
def forgotpassword():
    return redirect('http://localhost:3000/forgotpass')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')