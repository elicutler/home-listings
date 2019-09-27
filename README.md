### Steps to initialize the jupyter environment
If setting up the environment for the first time, execute the following statements:
```
cd ~/SageMaker
git clone https://github.com/elicutler/home-listings.git

cd home-listings
chmod +x ./init_env.sh
```

After the first-time setup steps, or if restarting a stopped jupyter instance, run:
```
git config --global user.name <your name>
git config --global user.email <your email>

cp -rf user-settings/.ipython ~/
cp -rf user-settings/.jupyter/lab/user-settings ~/.jupyter/lab/ # This will apply a dark theme, but need to refresh the browser for it to take effect.

conda env create -f environment.yml
echo 'conda activate' >> ~/.bashrc

source ~/.bashrc
conda activate home-listings
```

C, apply a dark theme to JupyterLab; however, you will need to refresh the browser tab for this setting to take effect.

After running the `init_env.sh` script, if you want to download data through Datafiniti's API you will need to create a `credentials.py` file with a string variable `DATAFINITI_API_TOKEN` containing your Datafiniti API token.