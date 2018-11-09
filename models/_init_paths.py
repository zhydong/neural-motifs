import os
import sys
import os.path as osp

def add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)

this_dir = osp.dirname(__file__)

motifs_path = osp.abspath(osp.join(this_dir, '..'))
add_path(motifs_path)

