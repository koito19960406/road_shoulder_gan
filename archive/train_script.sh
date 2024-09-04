#!/bin/sh

# 1. Argument is a .txt file with its content separated by a new line
# 2. New lines are replaced by spaces and printed out
# 3. Printed content is read as arguments
python train.py `cat <$1 | tr '\n' ' '`
