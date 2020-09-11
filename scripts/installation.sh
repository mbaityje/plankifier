#!/bin/bash
#######################################################################
#
# A beginner's guide to set up a virtual environment and install the
# needed modules. The code in the repository is meant to run smoothly
# with the installations in this document.
#
#######################################################################



cd ~

#Install virtual env
sudo apt-get install -y python3-venv

# Make (if it doesn't exist) and access directory for virtual environments
mkdir .virtualenvs
cd .virtualenvs

# Create virtual environment
python3 -m venv plankifier

# Activate the environment (just type deactivate to deactivate)
source ~/.virtualenvs/plankifier/bin/activate

# Install the python modules into the environment
pip install --upgrade pip
pip3 install keras tensorflow-gpu pandas matplotlib numpy seaborn sklearn







