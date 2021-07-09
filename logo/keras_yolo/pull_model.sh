#!/bin/bash

# Exit on failure
set -e

# Retrieve the latest version of models.tar
wget https://storage.googleapis.com/7bb9263c6ea848ec43b8eae798677ae752a7c487/models.tar
#rm -rf models
tar -xvf models.tar
rm models.tar
