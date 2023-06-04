#!/bin/sh

# First argument is a .txt file with its content separated by a new line
# the contents are printed in one line and used as arguments for the
# training script
cat | tr '\n' ' ' < $1


