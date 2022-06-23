from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import numpy as np
from util import readobj, getbs
import open3d as o3d
import threading
import sys
import os
import torch
from flame.FLAME import FLAME
import argparse
from easydict import EasyDict as edict


params_out = {}
params_out["shape_params"] = torch.zeros(1, 300)
params_out['expression_params'] = torch.zeros(1, 100)
params_out["pose_params"] = torch.zeros(1, 6)
params_out["neck_pose"] = torch.zeros(1, 3)
params_out["eye_pose"] = torch.zeros(1, 6)

params_len = [300, 100, 6, 3, 6]
params_pos = np.cumsum(params_len).tolist()

def get_param_name(pos):
    global params_len
    global params_pos
    assert len(params_pos)==5, 'wrong code'
    params_pos.append(pos)
    params_pos.sort()
    idx = params_pos.index(pos)
    if idx == 0 and pos < 300:
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
        self.setMinimum(-5000) 
        self.setMaximum(5000) 
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
        MainWindow.resize(2280, 1280)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.layout_widgets = []
        self.layout_list = []
        layout_width = 180
        layout_num = 14
        for i in range(layout_num):
            layoutWidget = QtWidgets.QWidget(self.centralwidget)
            layoutWidget.setGeometry(QtCore.QRect(1+layout_width*i, 0, 160, 1300))
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
        layout_deep = 30
        self.ele_num = 415
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

def show_face(app, flame_config):
    global params_out
    print('Press [ESC] to exit.')
    
    global flag_exit
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)

    neutral_path = "flame/FLAME_sample.obj"
    flamelayer = FLAME(flame_config)
    nv, nf = readobj(neutral_path, num_v=5023)     # (1220, 3), (2304, 3)

    first_flag = True
    mesh = o3d.geometry.TriangleMesh()
    while not flag_exit:
        if first_flag:
            first_flag = False

            vertices, _ = flamelayer(**params_out)
            vertices = o3d.utility.Vector3dVector(vertices[0])
            triangles = o3d.utility.Vector3iVector(nf-1)
            mesh.vertices = vertices
            mesh.triangles = triangles
            mesh.paint_uniform_color([0.3,0.5,0.5])
            mesh.compute_vertex_normals()
            vis.add_geometry(mesh)
        else:
            vertices, _ = flamelayer(**params_out)
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

    
    parser = argparse.ArgumentParser(description='config for flame')
    # flame参数
    parser.add_argument(
        '--face_model_path',
        type = str,
        default = 'flame/generic_model.pkl',
        help = 'flame model path'
    )
    parser.add_argument(
        '--static_landmark_embedding_path',
        type = str,
        default = 'flame/flame_static_embedding.pkl',
        help = 'Static landmark embeddings path for FLAME'
    )
    parser.add_argument(
        '--dynamic_landmark_embedding_path',
        type = str,
        default = 'flame/flame_dynamic_embedding.npy',
        help = 'Dynamic contour embedding path for FLAME'
    )
    # FLAME hyper-parameters
    parser.add_argument(
        '--shape_params',
        type = int,
        default = 300, # 100
        help = 'the number of shape parameters'
    )
    parser.add_argument(
        '--expression_params',
        type = int,
        default = 100, # 50
        help = 'the number of expression parameters'
    )
    parser.add_argument(
        '--pose_params',
        type = int,
        default = 6,
        help = 'the number of pose parameters'
    )
    # Training hyper-parameters
    parser.add_argument(
        '--use_face_contour',
        default = False,
        type = bool,
        help = 'If true apply the landmark loss on also on the face contour.'
    )
    parser.add_argument(
        '--use_3D_translation',
        default = False, # Flase for RingNet project
        type = bool,
        help = 'If true apply the landmark loss on also on the face contour.'
    )
    parser.add_argument(
        '--optimize_eyeballpose',
        default = True, # False for For RingNet project
        type = bool,
        help = 'If true optimize for the eyeball pose.'
    )
    parser.add_argument(
        '--optimize_neckpose',
        default = True, # False For RingNet project
        type = bool,
        help = 'If true optimize for the neck pose.'
    )
    parser.add_argument(
        '--num_worker',
        type = int,
        default = 4,
        help = 'pytorch number worker.'
    )
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 1,
        help = 'Training batch size.'
    )
    flame_config = parser.parse_args()
    t2 = MyThread(show_face, args=(app,flame_config)) 
    t2.start()
    app.exec_()

    global flag_exit
    flag_exit = True


if __name__ == "__main__":
    main()