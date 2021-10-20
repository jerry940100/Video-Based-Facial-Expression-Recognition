# Video Based Facial Expression Recognition
> This note shows the experimental results on Oulu-Casia dataset by using **R2plus1D**, **TSM**,  **2-layers C3D**, **TimeSFormer**, **ViViT**, **CNN+LSTM_Attention** models.
## :memo: Outline
* Experimental Model Introduction
    * R2plus1D
    * TSM
    * 2-layers C3D
    * TimeSFormer
    * ViViT
    * CNN+LSTM_Attention
* Dataset Introduction and Preprocessing
* Experimental Settings
* Experimental Results
## Experimental Model Introduction
* ### [R2plus1D](https://github.com/jerry940100/Video-Based-Facial-Expression-Recognition/blob/main/Model/R2plus1d.py)
    ![](https://i.imgur.com/f4cB6xZ.png)
    >R2plus1D separates 3d convolution into 2d concolution + 1d convolution. There are two advantages of R2plus1D compared with C3D. 
    >1. First, despite not changing the number of parameters, it doubles the number of nonlinearities in the network due to the additional ReLU between the 2D and 1D convolution in each block. 
    >2.  The second benefit is that forcing the 3D convolution into separate spatial and temporal components renders the optimization easier. This is manifested in lower training error compared to 3D convolutional networks of the same capacity.
* ### [TSM](https://github.com/mit-han-lab/temporal-shift-module)
    ![](https://i.imgur.com/9j7VMoq.png)
    >TSM learns the temporal information by shifting the feature map along the temporal dimension. It is computationally free on top of a 2D convolution.
    >The detail introduction and implementation can refer [official temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module)
* ### [2-layers C3D](https://github.com/jerry940100/Video-Based-Facial-Expression-Recognition/blob/main/Model/C3D.py)
    
    <img src=https://i.imgur.com/3nqoA8z.png alt="drawing" style="width:250px;vertical-align:middle;"/><br>

* ### TimeSFormer
    ![](https://i.imgur.com/O3l1yam.png)


* ### [ViViT](https://github.com/jerry940100/Video-Based-Facial-Expression-Recognition/blob/main/Model/ViViT_pretrained.py)
    ![](https://i.imgur.com/9Ju3Tuz.png)
    >ViViT first embeds each frame of a video sequence and        separates them into different **Vision Transformer**.  After getting each ViT output. ViViT sends each output of ViT to a temporal transformer to learn temporal information between different embeddings. 
* ### [CNN+LSTM_Attention](https://github.com/jerry940100/Video-Based-Facial-Expression-Recognition/blob/main/Model/CNNplusLSTM.py)
    <img src=https://i.imgur.com/b58jPh3.png alt="drawing" style="width:250px;vertical-align:middle;"/><br>
    >This model first uses CNN to extract spatial features in each frame and input these embeddings into LSTM model to learn the temporal features in the video sequence. After LSTM, we use the Attention here to learn the importance degree in the sequence. Then, we do the weighted sum on each embedding from LSTM and connect a linear layer to classify.


## Dataset Introduction and Preprocessing
![](https://i.imgur.com/uQxwoHZ.png)


## Experimental Settings
![](https://i.imgur.com/cddnwYg.png)


## Experimental Results
![](https://i.imgur.com/o7o8YDT.png)



