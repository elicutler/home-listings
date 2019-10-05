### Steps to initialize the jupyter environment
If setting up the environment for the first time, execute the following statements. Note that if you are not me you should first edit the `git config` lines of the `init_env.sh` file to take your own name and email. Note also that this will apply my setting preferences (dark theme for jupyterlab, material theme for text editor).
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
