conda create -n os python=3.8 -y
conda activate os

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip3 install -U pip --user
pip3 install setuptools==59.5.0 --user
pip3 install 'pytest-runner<5.0' --user
pip3 install PyYAML --user
pip3 install matplotlib   --user
pip3 install opencv-python --user
pip3 install pandas  --user
pip3 install tqdm  --user
pip3 install timm==0.5.4  --user
pip3 install tensorboard  --user
pip3 install tensorboardX  --user
pip3 install jpeg4py  --user
pip3 install wandb  --user
pip3 install ipdb  --user
pip3 install imgcat  --user
pip3 install lmdb  --user
pip3 install einops  --user
pip3 install pycocotools  --user
pip3 install easydict  --user
pip3 install visdom  --user
pip3 install tikzplotlib --user
pip3 install memorizing-transformers-pytorch
pip3 install faiss
pip3 install einops_exts