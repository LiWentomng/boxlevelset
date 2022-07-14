#!/usr/bin/env bash

pip install -r requirements/build.txt

pip install -v -e . #Or  python setup develop

cd ./mmdet/ops/tree_filter/
python setup.py build develop
