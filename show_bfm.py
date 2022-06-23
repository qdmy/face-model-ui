from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import numpy as np
from util import readobj, getbs
import open3d as o3d
import threading
import sys
import os
import torch
from bfm.bfm import BFM_mat, Morphable
import argparse


params_out = {}
params_out["shape_params"] = torch.zeros(1, 80)
params_out['expression_params'] = torch.zeros(1, 64)

params_len = [80, 64]
params_pos = np.cumsum(params_len).tolist()

def get_param_name(pos):
    global params_len
    global params_pos
    assert len(params_pos)==2, 'wrong code'
    params_pos.append(pos)
    params_pos.sort()
    idx = params_pos.index(pos)
    if idx == 0 and pos < 80:
        idx_in = pos
    elif params_pos[idx] == params_pos[idx+1]:
        idx_in = 0
        idx = idx + 1
    else:
        idx_in = params_pos[idx] - params_pos[idx-1]
    params_pos.pop(idx)
    return list(params_out.keys())[idx], idx_in

def get_bs_name():
    res = []
    global params_out
    for k, v in params_out.items():
        new_k = k.split('_')[0]
        for i in range(v.shape[1]):
            res.append(f'{new_k}_{i}')
    return res


class Foo(QtWidgets.QSlider): 
    def __init__(self, widget, id, name): 
        super().__init__(widget) 
        self.timer_id = -1 
        self.name = name # params_name_idx
        # self.slider = QtWidgets.QSlider(widget) 
        self.setMinimum(-500) 
        self.setMaximum(500) 
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
        global params_out
        params_out[self.name][0, self.id] = self.value() / 100.0


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        global params_out

        MainWindow.setObjectName("SHOW FLAME")
        MainWindow.resize(1280, 1120)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.layout_widgets = []
        self.layout_list = []
        layout_width = 180
        layout_num = 7
        for i in range(layout_num):
            layoutWidget = QtWidgets.QWidget(self.centralwidget)
            layoutWidget.setGeometry(QtCore.QRect(1+layout_width*i, 0, 150, 1100))
            layoutWidget.setObjectName("verticalLayoutWidget_{}".format(i))
            self.layout_widgets.append(layoutWidget)
            layout = QtWidgets.QVBoxLayout(layoutWidget)
            layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setObjectName("verticalLayout_{}".format(i))
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.layout_list.append(layout)
            

        self.label_list = []
        self.foo_list = []
        layout_deep = 21
        self.ele_num = 144
        current_layout = 0
        for i in range(self.ele_num):
            param_name, idx_in = get_param_name(i)
            label = QtWidgets.QLabel(self.layout_widgets[current_layout])
            coeffBar = Foo(self.layout_widgets[current_layout], idx_in, param_name)
            label.setObjectName("{}_{}".format(param_name.split("_")[0], idx_in))
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

def show_face(app, face_model_config):
    global params_out
    print('Press [ESC] to exit.')
    
    global flag_exit
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)

    neutral_path = "bfm/template.obj"
    bfm_basis = BFM_mat({'dir': "bfm/", 'front': True}, geo_scale=1e-6)
    bfm_basis.reduce_num_shape_coeffs(face_model_config.shape_params) # 只有80个shape
    bfm_basis.reduce_num_expression_coeffs(face_model_config.expression_params) # 只有64个expression
    bfm_model = Morphable(bfm_basis)
    nv, nf = readobj(neutral_path, num_v=35709)     # (1220, 3), (2304, 3)

    first_flag = True
    mesh = o3d.geometry.TriangleMesh()
    while not flag_exit:
        if first_flag:
            first_flag = False

            vertices = bfm_model(torch.cat(list(params_out.values()), dim=-1))
            vertices = o3d.utility.Vector3dVector(vertices[0])
            triangles = o3d.utility.Vector3iVector(nf-1)
            mesh.vertices = vertices
            mesh.triangles = triangles
            mesh.paint_uniform_color([0.3,0.5,0.5])
            mesh.compute_vertex_normals()
            vis.add_geometry(mesh)
        else:
            vertices = bfm_model(torch.cat(list(params_out.values()), dim=-1))
            vertices = o3d.utility.Vector3dVector(vertices[0])
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

    
    parser = argparse.ArgumentParser(description='config for face model')
    parser.add_argument(
        '--shape_params',
        type = int,
        default = 80,
        help = 'the number of shape parameters'
    )
    parser.add_argument(
        '--expression_params',
        type = int,
        default = 64,
        help = 'the number of expression parameters'
    )

    config = parser.parse_args()
    t2 = MyThread(show_face, args=(app,config)) 
    t2.start()
    app.exec_()

    global flag_exit
    flag_exit = True


if __name__ == "__main__":
    main()