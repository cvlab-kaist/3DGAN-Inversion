mkdir pretrained_models
cd pretrained_models

wget 'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-128.pkl' # Pretrained EG3D Model
wget 'https://www.dropbox.com/s/eavcyyv7zhrmctv/pose_estimator.pt'
wget 'https://www.dropbox.com/s/ifq7ojavrvolglp/e4e_24K.pt'

cd ..