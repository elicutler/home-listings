
# restore my custom user settings in place
cp user-settings/* ~/

# create conda env then activate it
conda env create -f environment.yml
sudo ln -s /home/ec2-user/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh
echo 'conda activate' >> ~/.bashrc
source ~/.bashrc
conda activate home-listings
