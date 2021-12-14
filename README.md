# NLP_SA_Demo

1.   数据集来自[疫情期间网民情绪识别 竞赛 - DataFountain](https://www.datafountain.cn/competitions/423/datasets)。

2.   基于tensorflow框架和[huggingface 的 bert-base-chinese](https://huggingface.co/bert-base-chinese)与训练模型实现的情感分析。
3.   本地展示页面基于flask实现。设置的端口为`8848`
4.   加载模型需要从[bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main)处下载文件如下并放置在根目录

```
config.json
README.md
tf_model.h5
tokenizer.json  
tokenizer_config.json  
vocab.txt

============放置位置===========

C:.
├─bert-base-chinese
├─ckpt_2
├─templates 
├─config.py
├─main.py
├─model.py
├─main.py
└─__pycache__
```

