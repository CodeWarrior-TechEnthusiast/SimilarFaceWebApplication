import processing, os
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads'

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/try/', methods=('GET', 'POST'))
def main_app():
    if request.method == "POST":
        file = request.files.getlist("file")
        #file = request.files['file']
        for x in file:
            x.save(os.path.join(app.config['UPLOAD_FOLDER'], x.filename))
        processing.process()
        return render_template('results.html')
    return render_template('try.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == "__main__":
    app.run()

