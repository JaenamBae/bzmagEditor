# -*- coding: utf-8 -*-

import sys, os
from PyQt5.QtGui import (QIcon)

# ------------------------------------------------------------------------------
class NOHImageProvider:
    def __init__(self, root_path):
        self.images = {}
        self.load(root_path)
        
    def load(self, root_path):
        # add default icon
        try:
            self.images['Default'] = QIcon(root_path + '/Default.gif')
        except:
            raise

        # add icons
        fnames = os.listdir(root_path)
        for fname in fnames:
            path = root_path + '/' + fname
            if not os.path.isfile(path):
                continue
            head, tail = os.path.split(path)
            name, ext = os.path.splitext(tail)
            ext = ext.lower()
#            print(name)
            self.images[name] = QIcon(path)
            
    def getImage(self, name):
        try:
            return self.images[name]
        except:
            raise

# self.fullscreen_button.setIcon(fullscreen_icon)