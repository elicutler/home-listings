# For conda commands to work, need to execute this script using: . ./init_env.sh

git config --global user.name 'Eli Cutler'
git config --global user.email cutler.eli@gmail.com

cp -rf user-settings/.ipython ~/
cp -rf user-settings/.jupyter/lab/user-settings ~/.jupyter/lab/

conda env create -f environment.yml

echo ". /home/ec2-user/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
echo "conda activate" >> ~/.bashrc
source ~/.bashrc

conda deactivate
conda activate home-listings