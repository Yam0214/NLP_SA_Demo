# -*- coding: utf-8 -*-
# @Time : 2021/12/14 12:30
# @Author : Yam

from pathlib import Path
import re
import numpy as np
import tensorflow as tf
from zhconv import convert
from tensorflow import keras
from transformers import AutoTokenizer, TFAutoModel


TOKEN_LEN = 200  # text 转化成token后的长度
ckpt_path = Path("ckpt")  # 模型权重保存位置


class Model():
    def __init__(self, pretrained_dir, ckpt_path):
        # token
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
        # model
        bert_model = TFAutoModel.from_pretrained(pretrained_dir)

        # build model
        input_token = keras.Input(shape=(TOKEN_LEN,), dtype=tf.int32)
        embedding = bert_model(input_token)[0]
        embedding = keras.layers.GlobalAveragePooling1D()(embedding)
        output = keras.layers.Dense(
            3,
            activation=tf.nn.softmax,
            bias_regularizer=keras.regularizers.l1_l2(l1=0.005, l2=0.005)
        )(embedding)
        self.model = keras.Model(input_token, output)

        # 编译
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        # 加载权重
        self.model.load_weights(ckpt_path)
        print("Finished loading model.")

    def clean(self, text):
        # 数据清洗方法
        text = re.sub(r"(回复)?(//)?\s*@\S*?\s*:", "@", text)  # 去除正文中的@和回复/转发中的用户名
        # text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
        # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
        URL_REGEX = re.compile(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
            re.IGNORECASE)
        text = re.sub(URL_REGEX, "", text)  # 去除网址
        text = text.replace("转发微博", "")  # 去除无意义的词语
        text = text.replace("O网页链接?", "")
        text = text.replace("?展开全文c", "")
        text = text.replace("网页链接", "")
        text = text.replace("展开全文", "")
        text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
        return text.strip()

    def get_token(self, text):
        # TOKEN_LEN 输入长度
        token = self.tokenizer.encode(text)
        token = token[:TOKEN_LEN] + [0] * (TOKEN_LEN - len(token))
        return token

    def pre(self, text_list):
        # 数据预处理

        token_list = []
        for text in text_list:
            res = convert(text, locale='zh-cn')
            res = self.clean(text)
            res = self.get_token(text)
            token_list.append(res)
        return np.array(token_list)

    def run(self, msg):
        text_list = msg.split("\n")
        inputs = self.pre(text_list)  # 预处理
        probs = self.model.predict(inputs)  # 分类预测
        # 生成结果
        outputs = []
        for i in range(probs.shape[0]):
            label = probs[i].argmax() - 1
            outputs.append("【{:>2d}】: {}\n".format(label, text_list[i].strip()))
        return outputs



if __name__ == '__main__':
    from config import ckpt_path, pretrained_dir
    model = Model(pretrained_dir, ckpt_path)
    msg = """起得比鸡早，睡得比狗晚。黑眼圈一天天加重，累死了！
    我太难了别人怎么发烧都没事就我一检查甲型流感?
    同样的游轮，不同的命运。钻石公主号这是要全船覆灭的节奏，各国赶快把自家人都早早领回去，别沾光了日本的医疗资源。?
    昨晚亲眼看见了流星雨诶！我可太幸运了。
    小保底又歪了……"""
    outputs = model.run(msg)
    print(outputs)