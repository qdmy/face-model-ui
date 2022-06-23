import numpy as np
import os

def readobj(objpath, num_v):
    with open(objpath, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip().split(' ')[1:4]
    f.close()
    vertices = np.array(lines[1:num_v+1], dtype=np.float32)
    faces = np.array(lines[num_v+1:], dtype=np.int32)
    return vertices, faces

def getbs(bsdir):
    names = os.listdir(bsdir)
    names.sort()
    bslist = [bsdir +  i for i in names]
    # bslist.sort()
    blendshape = [ ]
    for bs in bslist:
        if 'Neutral' in bs:
            continue
        vertices, _ = readobj(bs, num_v=1220)
        blendshape.append(vertices)
    return np.array(blendshape, dtype=np.float32)


if __name__ == '__main__':
    objpath = 'blendshape/browDownRight.obj'
    v, f = readobj(objpath)
    print(v.shape)
    print(f.shape)
    bsdir = 'blendshape/'
    bs = getbs(bsdir)
    print(bs.shape)
    print('aa')