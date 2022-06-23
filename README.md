# data
prepare data as written in each face model directory
## ARKIT
I have left these blendshapes in [arkit_blendshape/](https://github.com/qdmy/face-model-ui/tree/master/arkit_blendshape) since they cannot be found on the internet (I don't know why)
## BFM
- download BFM_2009 (search and apply for it yourself) and put `01_MorphableModel.mat` (in your downloaded BFM_2009) into [BFM/](https://github.com/qdmy/face-model-ui/tree/master/bfm)
- download `BFM_exp_idx.mat`, `BFM_front_idx.mat`, `std_exp.txt` from [here](https://github.com/microsoft/Deep3DFaceReconstruction/tree/master/BFM) and put them into [BFM/](https://github.com/qdmy/face-model-ui/tree/master/bfm)
- download `Exp_Pca.bin` from [here](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view) and put it into [BFM/](https://github.com/qdmy/face-model-ui/tree/master/bfm)
## FLAME
- download [FLAME](https://flame.is.tue.mpg.de/) and put `generic_model.pkl` (male or female may be supported, haven't tested) into [flame/](https://github.com/qdmy/face-model-ui/tree/master/flame)
- download `flame_dynamic_embedding.npy` and   `flame_static_embedding.pkl` from [here](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model) and put them into [flame/](https://github.com/qdmy/face-model-ui/tree/master/flame)

# dependency
## poetry
- install [poetry](https://python-poetry.org/docs/#installation)
- run this command, it will create a virtualenv that support this repository
    ```bash
    poetry install
    ```
- then you can use the created virtualenv to run the scripts (no need to activate it):
    ```bash
    poetry run python show_arkit.py
    poetry run python show_bfm.py
    poetry run python show_flame.py
    ```
## pip
- run this command to install dependencies
    ```bash
    pip install -r requirements.txt
    ```
- then run one of the following:
    ```bash
    python show_arkit.py
    python show_bfm.py
    python show_flame.py
    ```