### Steps to initialize the jupyter environment
If setting up the environment for the first time, execute the following statements:
```
cd ~/SageMaker
git clone https://github.com/elicutler/home-listings.git
git config --global user.name <your name>
git config --global user.email <your email>
cd home-listings
sudo ln -s /home/ec2-user/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh
chmod +x ./init_env.sh
```

Run:
```
./init_env.sh
echo 'conda activate' >> ~/.bashrc
source ~/.bashrc
conda activate home-listings
```

Running `./init_env.sh` will, among other things, apply a dark theme to JupyterLab; however, you will need to refresh the browser tab for this setting to take effect.

After running the `init_env.sh` script, if you want to download data through Datafiniti's API you will need to create a `credentials.py` file with a string variable `DATAFINITI_API_TOKEN` containing your Datafiniti API token.