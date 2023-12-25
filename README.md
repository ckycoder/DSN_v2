# DSN_v2
The code of the paper ‘DSN-v2: Improving the Classification Ability to Man-Made and Natural Objects in SAR Images’

# Abstract
The traditional convolutional neural network (CNN)-based methods usually use the spatial information in the amplitude of complex synthetic aperture radar (SAR) images. Several studies have started to concentrate on merging the unique physical properties of SAR images, such as DSN-v1, extracting the backscattering characteristic from the frequency domain. Although DSN-v1 has obtained impressive classification ability, there is some room for improvement. In this letter, DSN-v2 is proposed to boost the classification ability of man-made and natural objects in SAR images. The improvement is reflected in two aspects. First, a multiscale subband feature extraction (MSFE) component is designed for natural objects. Since we observe their multiscale subband spectrum is significantly different, multiple encoders are used to extract effective features. Second, the additive angular margin (AAM) loss is introduced to distinguish man-made objects more clearly by manually adding a margin to the decision boundary. The experimental results on the Sentinel-1 (S1) dataset show that DSN-v2 achieves superior classification performance and model training speed compared with DSN-v1.

# Run
        run train_XXX.py will give the recognition accuracy on the test set in every epoch
        run test_xxx.py will give the F1_score
## dataset
        please follow and cite https://github.com/Alien9427/DSN

## model
        model is an empty folder that can be used to save the model
# Citation
## If you find this repository/work helpful in your research, welcome to cite the paper.
        @ARTICLE{10290990,
          author={Chen, Keyang and Pan, Zongxu and Niu, Ben and Hong, Wen and Hu, Yuxin and Ding, Chibiao},
          journal={IEEE Geoscience and Remote Sensing Letters}, 
          title={DSN-v2: Improving the Classification Ability to Man-Made and Natural Objects in SAR Images}, 
          year={2023},
          volume={20},
          number={},
          pages={1-5},
          doi={10.1109/LGRS.2023.3326605}}
