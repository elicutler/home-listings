### Steps to initialize the jupyter environment
If setting up the environment for the first time, execute the following statements:
```
cd ~/SageMaker
git clone https://github.com/elicutler/home-listings.git

cd home-listings
chmod +x ./init_env.sh

vi credentials.py 
DATAFINITI_API_TOKEN = '<your Datafiniti API token>'
:wq
```

After the first-time setup steps, or if restarting a stopped jupyter instance, run:
```
. ./init_env.sh
```
Running `./init_env.sh` will, among other things, apply a dark theme to JupyterLab; however, you will need to refresh the browser tab for this setting to take effect.
