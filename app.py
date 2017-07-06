# encoding: utf-8

import os
from flask import Flask, url_for, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import i2v

app = Flask(__name__)

with open('i2v.pid','w') as f:
    pid = os.getpid()
    f.write(str(pid)+"\n")
    with open('/proc/' + str(pid) + '/oom_adj', 'w') as f2:
        f2.write("-17\n")

illust2vec = i2v.make_i2v_with_chainer("illust2vec_tag_ver200.caffemodel", "tag_list.json")

@app.route("/i2v", methods=['POST'])
def i2v():
    f = request.files['the_file']
    fname = './static/images/'+secure_filename(f.filename)
    f.save(fname)
    img = Image.open(fname)
    preds = illust2vec.estimate_plausible_tags([img], threshold=0.7)
    return jsonify(preds)

@app.route("/feature2", methods=['POST'])
def i2v_feature2():
    f = request.files['the_file']
    #fname = './static/images/'+secure_filename(f.filename)
    #f.save(fname)
    with f.read() as img:
        feature = illust2vec.extract_feature([img])[0]
        os.remove(fname)
        j = jsonify({'feature': feature.tolist()})
    return j

@app.route("/feature", methods=['POST'])
def i2v_feature():
    f = request.files['the_file']
    fname = './static/images/'+secure_filename(f.filename)
    f.save(fname)
    with Image.open(fname) as img:
        feature = illust2vec.extract_binary_feature([img])[0]
        os.remove(fname)
        j = jsonify({'feature': feature.tolist()})
    return j

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

