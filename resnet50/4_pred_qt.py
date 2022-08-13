# -*- coding: utf-8 -*-

import tensorflow.keras.models as models
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil


class MainWindow(QTabWidget):
    # 初始化
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('ResNet50图片分类')
        # 模型初始化
        self.model = models.load_model("models/temp.h5")
        self.to_predict_name = "images/show.png"
        #这里需要用的训练时model.summary()代码 打印的class_names 列表['negative', 'positive']与最后预测结果一一对应
        self.class_names =['Apple', 'Apricot', 'Banana', 'Blueberry', 'Cherry', 'Grape', 'Peach', 'Pear', 'Pineapple', 'Watermole']
        self.resize(888, 888)
        self.initUI()

    # 界面初始化，设置界面布局
    def initUI(self):
        cnn_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)

        # 主页面，设置组件并在组件放在布局上
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("请选择 \n相关图片图片并上传 点击开始识别即可")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        label_result = QLabel(' 识别结果 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('楷体', 10))
        self.result.setFont(QFont('楷体', 20))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        cnn_widget.setLayout(main_layout)

        # 关于页面，设置组件并把组件放在布局上
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用图片分类识别系统')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/bj.png'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel("作者：闲鱼zlf")
        label_super.setFont(QFont('楷体', 12))
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        main_layout.addWidget(about_title)
        main_layout.addStretch()
        main_layout.addWidget(about_img)
        main_layout.addStretch()
        main_layout.addWidget(label_super)
        main_widget.setLayout(main_layout)

        # 添加注释
        self.addTab(main_widget, '主页面')
        self.addTab(cnn_widget, '图片图片识别')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/图片分类.png'))


    # 上传并显示图片图片
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')  # 打开文件选择框选择文件
        img_name = openfile_name[0]  # 获取图片图片名称
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmp_up." + img_name.split(".")[-1]  # 将图片图片移动到当前目录
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)  # 打开图片图片
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)  # 将图片图片的大小统一调整到400的高，方便界面显示
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))  # 将图片图片大小调整到224*224用于模型推理
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))
            self.result.setText("等待识别")

    # 预测图片图片
    def predict_img(self):
        img = Image.open('images/target.png')  # 读取图片图片
        img = np.asarray(img)  # 将图片图片转化为numpy的数组
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))  # 将图片图片输入模型得到结果
        print(outputs)
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]  # 获得对应的名称
        self.result.setText(result)  # 在界面上做显示

    # 界面关闭事件，询问用户是否关闭
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
