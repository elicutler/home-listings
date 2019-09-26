### Steps to initialize the jupyter environment
If setting up the environment for the first time, execute the following statements:
```
cd ~/SageMaker
git clone https://github.com/elicutler/home-listings.git
git config --global user.name <your name>
git config --global user.email <your email>
cd home-listings
chmod +x ./init_env.sh
./init_env.sh
```

If restoring the environment after having previously stopped the notebook instance, only run:
```
./init_env.sh
```

