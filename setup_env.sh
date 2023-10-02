conda env create -f environment.yml 
conda activate askBert 
conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
echo 'y' | pip install -r requirements.txt
export SQUAD_DIR=`pwd`/SQuAD-2.0
pip install jupyter
ipython kernel install --name "askBert" --user