from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap

import os
# import model_test
import model_test
import you_dl
import soundfile as sf

app = Flask(__name__, template_folder='Template')
Bootstrap(app)

"""
Routes
"""
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        url = request.form['url_text']
        url = url.strip()

        if uploaded_file.filename != '':
            av_path = os.path.join('static', uploaded_file.filename)
            image_name = uploaded_file.filename.split('.')[0]
            image_path = os.path.join('static', image_name + '.png')
            uploaded_file.save(av_path)
            pred = model_test.get_prediction(av_path,image_path)

            result = {
                'class_name': pred,
                'image_path': image_path,
            }
            return render_template('result.html', result = result)
        elif url != '':
            av_path, title = you_dl.dwl_vid(url)
            # image_name = title.split('.')[0]
            # image_path = os.path.join('static', image_name + '.png')
            image_path = "static/out_graph.png"
            pred = model_test.get_prediction(av_path,image_path)
            result = {
                'class_name': pred,
                'image_path': image_path,
            }
            return render_template('result.html', result = result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True)
