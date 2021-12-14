# -*- coding: utf-8 -*-
# @Time : 2021/12/14 12:30
# @Author : Yam

from flask import Flask, render_template, request, redirect
import os
from model import Model
from config import ckpt_path, pretrained_dir

# 创建程序
# web应用程序
app = Flask(__name__)

# 准备模型
model = Model(pretrained_dir, ckpt_path)

@app.route('/',methods = ['POST','GET'])
def index():
    # 一般页面
    return render_template(
         "demo.html",
        msg = "这里展示预测结果。"
    )


@app.route('/predict',methods = ['POST','GET'])
def prediction():
    # 返回预测结果
    inputs = request.form.get('inputs')
    print("{:=^80}".format("[get input]"))
    print(inputs)
    print()
    print("{:-^80}".format("[return]"))
    outputs = model.run(inputs)
    print(outputs)
    print("{:=^80}".format("[end]"))
    return render_template(
        "demo.html",
        outputs = outputs
    )


app.run(
    host = "127.0.0.1",
    port = 8848,
    debug = True
)