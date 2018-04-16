# chinese_font_recognition
中文识别的流程，model部分可以自己更改。如果用的是IDE会出现错误，我这里是用文本编辑器写的，还请自行更改。

model部分有三个，centerloss的可以自己做适当修改

resnet50可以跑到val_acc = 0.94左右

# data_split 
ratio是一个浮动的参数，可以分配train和val的数据。

文件是按照ratio随机划分的，需要PIL包。不会的可以用pip install PIL

自己替换自己路径就可以了。
