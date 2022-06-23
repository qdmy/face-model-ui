from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import numpy as np
from util import readobj, getbs
import open3d as o3d
import threading
import sys
import os


PARAM_BS = np.zeros((52,), dtype=np.float32)

def get_bs_name():
    name_list = os.listdir("arkit_blendshape")
    name_list.sort()
    res = [ele.split('.')[0] for ele in name_list if 'Neutral' not in ele]
    return res


class Foo(QtWidgets.QSlider): 
    def __init__(self, widget, id, name): 
        super().__init__(widget) 
        self.timer_id = -1 
        # self.slider = QtWidgets.QSlider(widget) 
        self.setMinimum(0) 
        self.setMaximum(100) 
        self.valueChanged.connect(self.value_changed) 
        self.id = id

        self.setOrientation(QtCore.Qt.Horizontal)
        self.setObjectName(name)

    def timerEvent(self, event): 
        self.killTimer(self.timer_id) 
        self.timer_id = -1


    def value_changed(self): 
        if self.timer_id != -1: 
            self.killTimer(self.timer_id) 

        self.timer_id = self.startTimer(3000) 
        global PARAM_BS
        PARAM_BS[self.id] = self.value() / 100.0


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("SHOW BS")
        MainWindow.resize(1020, 750)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.layout_widgets = []
        self.layout_list = []
        layout_width = 250 # 控制两列滚动条间距
        layout_num = 4
        for i in range(layout_num):
            layoutWidget = QtWidgets.QWidget(self.centralwidget)
            layoutWidget.setGeometry(QtCore.QRect(20+layout_width*i, 10, 231, 700)) # left, top, width and height for each bar
            layoutWidget.setObjectName("verticalLayoutWidget_{}".format(i))
            self.layout_widgets.append(layoutWidget)
            layout = QtWidgets.QVBoxLayout(layoutWidget)
            layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
            layout.setContentsMargins(2, 0, 7, 0)
            layout.setObjectName("verticalLayout_{}".format(i))
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.layout_list.append(layout)
            

        self.label_list = []
        self.foo_list = []
        layout_deep = 13
        self.ele_num = 52
        current_layout = 0
        for i in range(self.ele_num):
            label = QtWidgets.QLabel(self.layout_widgets[current_layout])
            coeffBar = Foo(self.layout_widgets[current_layout], i, "coeff{}".format(i))
            label.setObjectName("label_{}".format(i))
            self.label_list.append(label)
            self.foo_list.append(coeffBar)

            self.layout_list[current_layout].addWidget(self.label_list[i])
            self.layout_list[current_layout].addWidget(self.foo_list[i])

            if (i+1) % layout_deep == 0:
                current_layout += 1


        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusba")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        name_list = get_bs_name()
        for i in range(self.ele_num):
            self.label_list[i].setText(_translate("MainWindow", name_list[i]))


from threading import Thread


# _sum = 0


def cal_sum(begin, end):
    # global _sum
    _sum = 0
    for i in range(begin, end + 1):
        _sum += i
    return  _sum

"""重新定义带返回值的线程类"""


class MyThread(Thread):
    def __init__(self, func, args=None):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        if self.args == None:
            self.result = self.func()
        else:
            self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return False



flag_exit = False

def show_face(app):
    global PARAM_BS
    print('Press [ESC] to exit.')
    
    global flag_exit
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480)

    neutral_path = "arkit_blendshape/Neutral.obj"
    bs_path = "arkit_blendshape/"

    nv, nf = readobj(neutral_path, num_v=1220)     # (1220, 3), (2304, 3)
    blendshape = getbs(bs_path)           # (52, 1220, 3)

    deltablendshape = 1.0*(blendshape-nv.reshape(1,-1,3))     #(52, 1220, 3) - (1, 1220, 3) = (52, 1220, 3)/
    deltablendshape = deltablendshape.reshape(52,-1).T  # (52, 3660)

    first_flag = True
    mesh = o3d.geometry.TriangleMesh()
    while not flag_exit:
        if first_flag:
            first_flag = False
            vertices = o3d.utility.Vector3dVector(nv +  \
                    np.matmul(deltablendshape, PARAM_BS.reshape(52,1)).reshape(-1,3))
            triangles = o3d.utility.Vector3iVector(nf-1)
            mesh.vertices = vertices
            mesh.triangles = triangles
            mesh.paint_uniform_color([0.3,0.5,0.5])
            mesh.compute_vertex_normals()
            vis.add_geometry(mesh)
        else:

            vertices = o3d.utility.Vector3dVector(nv + \
                    np.matmul(deltablendshape, PARAM_BS.reshape(52,1)).reshape(-1,3))
            mesh.vertices = vertices
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
    if flag_exit:
        app.closeAllWindows()



def main():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    t2 = MyThread(show_face, args=(app,)) 
    t2.start()
    app.exec_()

    global flag_exit
    flag_exit = True


if __name__ == "__main__":
    main()