
# restore my custom user settings
cp -rf user-settings/.ipython ~/
cp -rf user-settings/.jupyter/lab/user-settings ~/.jupyter/lab/

# create conda env then activate it
conda env create -f environment.yml
