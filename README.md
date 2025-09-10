# 基于Clip多模态模型的数字分类器1️⃣2️⃣3️⃣4️⃣5️⃣

## 如何使用这个代码😊

首先打开src文件，里面的clip为模型结构，ViT模型为VisionTransformer（参考论文：AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE），text_encoder为文本编码器，两者通过对比学习在实现多模态感知（参考论文：Learning Transferable Visual Models From Natural Language Supervision）📖

train文件为训练代码文件，模型的参数我已经训练完毕，可以直接使用inference文件实现效果展示😃

test文件是我自己写的一些测试案例，可以跳过阅读🤓

注意！在使用inference文件的时候，会生成数字图片的样子，将图片关闭后可以看到预测结果😊


