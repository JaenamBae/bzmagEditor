# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QCoreApplication, QMetaObject, pyqtSlot)
# from PyQt5.QtGui
from PyQt5.QtWidgets import (QWidget, QDockWidget, QHBoxLayout, QListWidget, QSizePolicy, QPushButton)
import bzmagPy as bzmag

class Ui_ConsoleWidget(object):
    def setupUi(self, ConsoleWidget):
        ConsoleWidget.setObjectName("ConsoleWidget")
        ConsoleWidget.resize(400, 300)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.horizontalLayout = QHBoxLayout(self.dockWidgetContents)
        self.horizontalLayout.setContentsMargins(2, 2, 2, 2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.listwidget_ouput = QListWidget(self.dockWidgetContents)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listwidget_ouput.sizePolicy().hasHeightForWidth())
        self.listwidget_ouput.setSizePolicy(sizePolicy)
        self.listwidget_ouput.setObjectName("listwidget_ouput")
        self.horizontalLayout.addWidget(self.listwidget_ouput)
        self.ok_button = QPushButton(self.dockWidgetContents)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ok_button.sizePolicy().hasHeightForWidth())
        self.ok_button.setSizePolicy(sizePolicy)
        self.ok_button.setObjectName("ok_button")
        self.horizontalLayout.addWidget(self.ok_button)
        ConsoleWidget.setWidget(self.dockWidgetContents)

        self.retranslateUi(ConsoleWidget)
        QMetaObject.connectSlotsByName(ConsoleWidget)

    def retranslateUi(self, ConsoleWidget):
        _translate = QCoreApplication.translate
        ConsoleWidget.setWindowTitle(_translate("ConsoleWidget", "DockWidget"))
        self.ok_button.setText(_translate("ConsoleWidget", "확인"))

#-------------------------------------------------------------------------------    
class ConsoleWidget(QDockWidget, Ui_ConsoleWidget):
    def __init__(self, parent=None):
        super(ConsoleWidget, self).__init__(parent)
        self.setupUi(self)