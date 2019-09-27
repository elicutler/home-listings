
git config --global user.name 'Eli Cutler'
git config --global user.email cutler.eli@gmail.com

cp -rf user-settings/.ipython ~/
cp -rf user-settings/.jupyter/lab/user-settings ~/.jupyter/lab/

conda env create -f environment.yml
echo 'conda activate' >> ~/.bashrc

source ~/.bashrc
conda activate home-listings