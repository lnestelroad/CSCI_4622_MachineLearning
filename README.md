# CSCI_4622_MachineLearning
Final project material for the CSCI 4622 Machine Learning course at CU Boulder.

**Authors:**  
+ Kaitlin Coleman _[kaitlin.coleman@colorado.edu]()_
+ Jot Kaur _[jot.kaur@colorado.edu]()_
+ Alex Ho _[alex.ho@colorado.edu]()_
+ Liam Nestelroad _[liam.nestelroad@colorado.edu]()_

## Summary

TODO: Decide on a project.

## Installation

To run this code, we highly recommend using a python virtual environment which can be set up using the following commands:

```bash
echo 'Shalom, World!'
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

Side note: the libraries for PyTorch and TensorFlow included are CPU versions. To install the GPU version, please consult with the sites for CUDA setup.

In addition, most code will be written in a Jupyter Notebook so we recommend getting that set up as well. For those using VScode for their Jupyter Notebooks, you will need to run the following command to add a ipython kernel that runs from the new virtual environment and then change it in the top right of the notebook itself.
```bash
ipython kernel install --user --name=env
```

To exit the python virtual environment, simply use the command `deactivate`. For more information, please refer to the following [here](https://realpython.com/python-virtual-environments-a-primer/)  


TODO: Include Docker support for Jupyter Notebooks

## Resources

+ Data sets and project ideas:
    + [Kaggle](https://www.kaggle.com/competitions)
+ Deep learning libraries:
    + [PyTorch Website](https://pytorch.org/)
    + [TensorFlow Website](https://www.tensorflow.org/)