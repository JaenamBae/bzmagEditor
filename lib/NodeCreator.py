# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NodeCreator.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5.QtGui import (QRegExpValidator)
from PyQt5.QtCore import (QSize, QRect, pyqtSlot, QMetaObject, Qt, 
     QCoreApplication, QRegExp)
from PyQt5.QtWidgets import (QWidget, QDialog, QGridLayout, QGroupBox, 
     QDialogButtonBox, QLineEdit, QLabel, QComboBox, QHBoxLayout, 
     QTreeWidget, QListWidget, QTreeWidgetItem, QListWidgetItem,
     QMessageBox)
import bzmagPy as bzmag

# ------------------------------------------------------------------------------
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(390, 400)
        Dialog.setMinimumSize(QSize(390, 400))
        Dialog.setMaximumSize(QSize(390, 400))
        Dialog.setFocusPolicy(Qt.StrongFocus)
        self.button = QDialogButtonBox(Dialog)
        self.button.setGeometry(QRect(220, 370, 161, 23))
        self.button.setOrientation(Qt.Horizontal)
        self.button.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.button.setObjectName("button")
        self.group_info = QGroupBox(Dialog)
        self.group_info.setGeometry(QRect(10, 10, 370, 110))
        self.group_info.setObjectName("group_info")
        self.grid_layout = QGridLayout(self.group_info)
        self.grid_layout.setObjectName("grid_layout")
        self.edit_name = QLineEdit(self.group_info)
        self.edit_name.setObjectName("edit_name")
        self.grid_layout.addWidget(self.edit_name, 2, 2, 1, 1)
        self.edit_path = QLineEdit(self.group_info)
        self.edit_path.setObjectName("edit_path")
        self.grid_layout.addWidget(self.edit_path, 1, 2, 1, 1)
        self.label_type = QLabel(self.group_info)
        self.label_type.setObjectName("label_type")
        self.grid_layout.addWidget(self.label_type, 3, 0, 1, 1)
        self.label_name = QLabel(self.group_info)
        self.label_name.setObjectName("label_name")
        self.grid_layout.addWidget(self.label_name, 2, 0, 1, 1)
        self.label_path = QLabel(self.group_info)
        self.label_path.setObjectName("label_path")
        self.grid_layout.addWidget(self.label_path, 1, 0, 1, 1)
        self.edit_type = QLineEdit(self.group_info)
        self.edit_type.setObjectName("edit_type")
        self.grid_layout.addWidget(self.edit_type, 3, 2, 1, 1)
        self.group_type = QGroupBox(Dialog)
        self.group_type.setGeometry(QRect(10, 140, 370, 221))
        self.group_type.setObjectName("group_type")
        self.horizontal_layout = QHBoxLayout(self.group_type)
        self.horizontal_layout.setObjectName("horizontal_layout")
        self.tree_module = QTreeWidget(self.group_type)
        self.tree_module.setObjectName("tree_module")
        self.tree_module.headerItem().setText(0, "1")
        self.tree_module.header().setVisible(False)
        self.horizontal_layout.addWidget(self.tree_module)
        self.list_type = QListWidget(self.group_type)
        self.list_type.setObjectName("list_type")
        self.horizontal_layout.addWidget(self.list_type)

        self.retranslateUi(Dialog)
        self.button.accepted.connect(Dialog.accept)
        self.button.rejected.connect(Dialog.reject)
        self.tree_module.currentItemChanged.connect(Dialog.bindNodeList)
        self.list_type.itemClicked.connect(Dialog.setNodeName)
        QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "노드 생성"))
        self.group_info.setTitle(_translate("Dialog", "노드 생성 정보"))
        self.label_type.setText(_translate("Dialog", "노드 타입"))
        self.label_name.setText(_translate("Dialog", "노드 이름"))
        self.label_path.setText(_translate("Dialog", "부모 경로"))
        self.group_type.setTitle(_translate("Dialog", "노드 타입"))


# ------------------------------------------------------------------------------
class NodeCreator(QDialog, Ui_Dialog):   
    def __init__(self, parent=None):
        super(NodeCreator, self).__init__()
        self.node_type_dict = None
        self.setupUi(self)
        self.build_module_list()
        self.build_type_list()
        
        reg_exp = QRegExp('^[a-z + A-Z + _]+\\w+$')
        reg_validator = QRegExpValidator(reg_exp)
        self.edit_name.setValidator(reg_validator)
        
    # public members -----------------------------------------------------------

            
    # private members ----------------------------------------------------------
    def build_module_list(self):
        mlist = bzmag.getModuleList()
        
        self.tree_module.clear()
        root_item = QTreeWidgetItem(self.tree_module)
        root_item.setText(0,'Modules')
        
        for mname in mlist:
            item = QTreeWidgetItem(root_item)
            item.setText(0, mname)
            
        self.tree_module.expandItem(root_item)
        
    def bind_type_list(self, mname):
        self.list_type.clear()
        for tname in bzmag.getTypeList(mname):
            self.list_type.addItem(tname)
            
    def build_type_list(self):       
        self.node_type_dict = {}
        for mname in bzmag.getModuleList():
            for tname in bzmag.getTypeList(mname):
                self.node_type_dict[tname] = 0
                
    def error_msg(self, msg):
        QMessageBox.question(self, 'Error', msg, QMessageBox.Ok, QMessageBox.Ok) 
    
    # Events -------------------------------------------------------------------
    @pyqtSlot(QTreeWidgetItem, QTreeWidgetItem)
    def bindNodeList(self, selected, deselected):
        mname = selected.text(0)
        self.bind_type_list(mname)
        
    @pyqtSlot(QListWidgetItem)
    def setNodeName(self, node):
        self.edit_type.setText(node.text())
        
        
    # slots / signals (?) overload
    def accept(self): 
        node_name = str(self.edit_name.text())
        node_path = str(self.edit_path.text())
        node_type = str(self.edit_type.text())

        if not node_name:
            self.error_msg('Please input into the \'Node Name\' field')
            return
        
        try:
            parent_node = bzmag.get(node_path)
            #print(parent_node)
        except:
            parent_node = None
            self.error_msg('Invalid path for the parent node')
            self.edit_path.setFocus()
            return 
            
        if parent_node.findChild(node_name):
            self.error_msg('Specified node name already exist on parent node')             
            self.edit_name.setFocus()
            return
    
        if not node_type in self.node_type_dict.keys():
            self.error_msg('unable to create specified type')
            self.edit_type.setFocus()
            return
            
        if not bzmag.isNode(node_type):
            self.error_msg('The node type is not kind of bzNodes.\n It could not possible to create a node')
            return
            
        super(NodeCreator, self).accept()
        
    def reject(self):
        print("reject")
        super(NodeCreator, self).reject()
        
