# -*- coding: utf-8 -*- 

import sys, os
sys.path.append(os.getcwd() + '/..')
sys.path.append('QtProperty')
sys.path.append('libqt5')


from PyQt5.QtWidgets import *
from bzmagEditor import *

def main():
    app = QApplication(sys.argv)
    myWindow = bzMagWindow(None)
    app.exec_()
    
if __name__ == '__main__':
    main()
