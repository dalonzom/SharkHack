import sys
import glob 
import os
from pathlib import Path
import shutil


trainPath = '/home/marissa/Downloads/asl_alphabet_test_real_world_+_kaggle'

def up_one_dir(path):
    try:
    	p = Path(path).absolute()
	parent_dir = p.parents[1]
	p.rename(parent_dir / p.name)
    except IndexError:
        # no upper directory
        pass

for r, d, f in os.walk(trainPath):
	for file in f: 
		up_one_dir(r + '/'+ file) 


