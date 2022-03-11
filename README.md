# labelme2X
将labelme的标注转换为voc或者coco格式数据集

代码来源[paddlex](https://github.com/PaddlePaddle/PaddleX/tree/ddec840f3f084f7f41b70226e0767edf81e8f2c4/paddlex/tools)

详见[1. 数据准备](https://github.com/PaddlePaddle/PaddleX#1-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)

# 使用
现在假设你有一个labelme注释好的数据集labelmeset，其格式如下：
- images ：存放所有图片的文件夹
- json：存放所有json的文件夹

现在你想将这个labelme标注数据集转为voc格式的数据集，则执行：
> python convert.py 
