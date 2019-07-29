# FaceRecognition
学习使用`face_recognition`

主要基于`python3.7`，`face_recognition`，部分功能需要一些图像处理库

## 环境配置

主要记录`win10`下如何安装`face_recognition`

安装 https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/ 中`可再发行组件和生成工具`的`Microsoft 生成工具 2015 更新 3`

安装`CMake`（不再赘述）

```
pip install cmake
pip install dlib
pip install face_recognition
```

## 使用说明

### face_in.py

从`1/`文件夹中读取`.jpg`文件，并以文件名作为姓名录入到`face.dat`文件中

**Warning:** 运行程序将直接覆盖原有的`face.dat`，不保留原有数据

单个图片中需只有一个人脸

`num_jitters`参数表示将读入人脸做多少次微小扰动提取特征值，设置较高的值可以些微提升准确性，大幅提高处理时长

多进程处理，CPU会跑满，可能会造成死机，不建议同时运行其他程序

### face_in_ext.py

从`data/`文件夹中读取`.jpg`文件，并以文件名作为姓名添加到当前目录`face.dat`文件中，相同姓名将被覆盖

`num_jitters`参数同上

### recognize.py

从`2/`文件夹中读取`.jpg`文件，以`face.dat`中的人脸数据识别并标注之后输出到原文件夹中，文件名将加上`_with_boxes`后缀

`EPS`参数是识别为`Unknown`的阈值，该值越高越容易识别为`Unknown`

`Textsize`参数是输出图像中字号的最小值，字号将在此值以上自适应

### live.py

通过`opencv2`读取摄像头，进行实时人脸识别，多进程避免图像卡顿并提高人脸较多时的处理效率，一般约有1s左右延迟

窗口中按`Q`退出

窗口中按`P`将保存当前识别到的人脸至`data/`文件夹中，命名为当前识别信息

`EPS`与`Textsize`参数同上

`Rate`参数为图像缩小率，增加此值将提高速度，降低识别率，此值小于`1`无意义

`RecoInterval`参数为进行识别的间隔秒数，降低此值将降低处理负载

运行此程序时将在控制台输出当前的负载信息，便于监视