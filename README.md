# data
prepare data as written in each face model directory
## ARKIT
I have left these blendshapes here since they cannot be found on the internet (I don't know why)
## BFM
- download BFM_2009 (search and apply for it yourself) and put `01_MorphableModel.mat` (in your downloaded BFM_2009) here
- download `BFM_exp_idx.mat`, `BFM_front_idx.mat`, `std_exp.txt` from [here](https://github.com/microsoft/Deep3DFaceReconstruction/tree/master/BFM) and put them here
- download `Exp_Pca.bin` from [here](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view) and put it here
## FLAME
- download [FLAME](https://flame.is.tue.mpg.de/) and put `generic_model.pkl` (male or female may be supported, haven't tested) here
- download `flame_dynamic_embedding.npy` and   `flame_static_embedding.pkl` from [here](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model) and put them here

# dependency
```bash
pip install -r requirements.txt
```

# run
```bash
python show_arkit.py
python show_bfm.py
python show_flame.py
```