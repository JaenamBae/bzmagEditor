# -*- coding: utf-8 -*-
import sys, os

sys.path.append('QtProperty')
sys.path.append('libqt5')

from PyQt5.QtGui import (
    QColor
    )
    
from PyQt5.QtCore import (
    QCoreApplication, 
    QMetaObject, 
    pyqtSlot, 
    Qt, 
    QRegExp, 
    pyqtSignal,
    QPoint,
    QPointF
    )
    
from PyQt5.QtWidgets import (
    QWidget, 
    QDockWidget, 
    QVBoxLayout, 
    QLineEdit,
    QDoubleSpinBox,
    QSpinBox, 
    QCheckBox
    )

from pyqtcore import (
    QMap, 
    QList
    )
    
from qttreepropertybrowser import (
    QtTreePropertyBrowser
    )

from qtpropertymanager import (
    QtBoolPropertyManager, 
    QtIntPropertyManager, 
    QtDoublePropertyManager,
    QtStringPropertyManager, 
    QtColorPropertyManager,
    QtPointFPropertyManager,
    QtSizePropertyManager, 
    QtRectPropertyManager, 
    QtSizePolicyPropertyManager, 
    QtEnumPropertyManager, 
    QtGroupPropertyManager
    )
    
from qteditorfactory import (
    QtCheckBoxFactory, 
    QtSpinBoxFactory, 
    QtDoubleSpinBoxFactory,
    QtSliderFactory, 
    QtScrollBarFactory, 
    QtLineEditFactory, 
    QtEnumEditorFactory,
    QtCharEditorFactory
    )

import bzmagPy as bzmag


#-------------------------------------------------------------------------------
class Ui_PropertyWidget(object):
    def setupUi(self, PropertyWidget):
        PropertyWidget.setObjectName("PropertyWidget")
        PropertyWidget.resize(400, 300)
        self.widget_property = QWidget()
        self.setFeatures(QDockWidget.AllDockWidgetFeatures | \
            QDockWidget.DockWidgetVerticalTitleBar)
        self.widget_property.setObjectName("widget_property")
        self.verticalLayout = QVBoxLayout(self.widget_property)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.line_path = QLineEdit(self.widget_property)
        self.line_path.setObjectName("line_path")
        self.verticalLayout.addWidget(self.line_path)
        self.prop_browser = QtTreePropertyBrowser()
        self.verticalLayout.addWidget(self.prop_browser)
        PropertyWidget.setWidget(self.widget_property)

        self.retranslateUi(PropertyWidget)
        QMetaObject.connectSlotsByName(PropertyWidget)

    def retranslateUi(self, PropertyWidget):
        _translate = QCoreApplication.translate
        PropertyWidget.setWindowTitle(_translate("PropertyWidget", "DockWidget"))


#-------------------------------------------------------------------------------    
class PropertyWidget(QDockWidget, Ui_PropertyWidget):
    itemUpdated = pyqtSignal(int)
    nameUpdated = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super(PropertyWidget, self).__init__(parent)
        self.setupUi(self)
        
        self.groupManager = QtGroupPropertyManager(self)
        self.boolManager = QtBoolPropertyManager(self)
        self.intManager = QtIntPropertyManager(self)
        self.doubleManager = QtDoublePropertyManager(self)
        self.stringManager = QtStringPropertyManager(self)
        self.colorManager = QtColorPropertyManager(self)
        self.enumManager = QtEnumPropertyManager(self)
        self.pointManager = QtPointFPropertyManager(self)
        
        self.boolManager.valueChangedSignal.connect(self.valueChanged)
        self.intManager.valueChangedSignal.connect(self.valueChanged)
        self.doubleManager.valueChangedSignal.connect(self.valueChanged)
        self.stringManager.valueChangedSignal.connect(self.valueChanged)
        self.colorManager.valueChangedSignal.connect(self.valueChanged)
        self.enumManager.valueChangedSignal.connect(self.valueChanged)
        self.pointManager.valueChangedSignal.connect(self.valueChanged)

        
        checkBoxFactory = QtCheckBoxFactory(self)
        spinBoxFactory = QtSpinBoxFactory(self)
        doubleSpinBoxFactory = QtDoubleSpinBoxFactory(self)
        lineEditFactory = QtLineEditFactory(self)
        comboBoxFactory = QtEnumEditorFactory(self)
        charEditfactory = QtCharEditorFactory(self)


        self.prop_browser.setFactoryForManager(\
            self.boolManager, checkBoxFactory)
        self.prop_browser.setFactoryForManager(\
            self.intManager, spinBoxFactory)
        self.prop_browser.setFactoryForManager(\
            self.doubleManager, doubleSpinBoxFactory)
        self.prop_browser.setFactoryForManager(\
            self.stringManager, lineEditFactory)
        self.prop_browser.setFactoryForManager(\
            self.colorManager.subIntPropertyManager(), spinBoxFactory)
        self.prop_browser.setFactoryForManager(\
            self.enumManager, comboBoxFactory)
            
        self.prop_browser.setFactoryForManager(\
            self.pointManager.subDoublePropertyManager(), doubleSpinBoxFactory)

        self.selectedNodeID_ = -1
        
        # for CS Nodes
        #self.CS_ = {}
        self.EnumToCSID_ = {}
        self.CSIDtoEnum_ = {}
        self.CSNames_ = QList()
        
        # for Material Nodes
        self.EnumToMatID_ = {}
        self.MatIDtoEnum_ = {}
        self.MatNames_ = QList()
        
    # Slot functions -----------------------------------------------------------
    @pyqtSlot(QList)
    def nodesSelected(self, nodeIDs):
        self.prop_browser.clear()
        nodeID = nodeIDs.last()
        if nodeID == None:
            return
        
        node = bzmag.getObject(nodeID)
        if node == None:
            self.selectedNode_ = None
            self.line_path.setText('')
            return
        
        path = node.getAbsolutePath()
        
        #self.CS_.clear()
        self.EnumToCSID_.clear()
        self.CSIDtoEnum_.clear()
        self.CSNames_.clear()
        
        self.EnumToMatID_.clear()
        self.MatIDtoEnum_.clear()
        self.MatNames_.clear()
        
        cs_root = bzmag.get('/coordinate')
        self.build_ObjectItem(cs_root, self.CSIDtoEnum_, self.EnumToCSID_, self.CSNames_, 'Global')
        
        mat_root = bzmag.get('/material')
        self.build_ObjectItem(mat_root, self.MatIDtoEnum_, self.EnumToMatID_, self.MatNames_, 'Vaccum')
        
        if path == '' : path = '/'
        self.line_path.setText(path)
        self.line_path.setReadOnly(True)
        
        #print('Binding node path', path)
        #node = bzmag.get(path)    
        self.selectedNodeID_ = nodeID
        
        
        for type_name in node.getGenerations():
            group = self.groupManager.addProperty(type_name)
            prop_names = node.getPropertyNames(type_name)
            if not prop_names:
                continue
         
            for prop_name in prop_names:
                prop_name, prop_value, prop_type, readonly = \
                    node.getProperty(prop_name)
                #print(prop_name, prop_value, prop_type, readonly)
                self.add_property(group, prop_name, prop_value, prop_type, readonly)
            
            #print('Property Browse : Add Property...', group)
            self.prop_browser.addProperty(group)

    # ------------------------------------------------------------------------     
    def valueChanged(self, property, value):
        if self.selectedNodeID_ == -1:
            return
        
        nodeID = self.selectedNodeID_
        node = bzmag.getObject(nodeID)
        prop_name = property.propertyName()
        prop_name, prop_value, prop_type, readonly = node.getProperty(prop_name)
        #print(prop_name, prop_value, prop_type, readonly)
            
        # when type of the value is 'QColor', make value as '[r, g, b, a]'
        #if type(value).__name__ == 'QColor':
        if prop_type == 'color':
            value = '{}, {}, {}, {}'.format(value.red(), value.green(), \
                value.blue(), value.alpha())
        
        if prop_type == 'node':
            #print('Node value changed : ', value)
            #if value in self.EnumToCSID_.keys():
            #value = self.EnumToCSID_[value]
            
            if prop_name == 'CoordinateSystem':
                value = self.EnumToCSID_[value]
                
            elif prop_name == 'Material':
                value = self.EnumToMatID_[value]

        # when the node is not readonly , update the property value
        if not readonly:
            #print(node, prop_name, str(value))
            node.setProperty(str(prop_name), str(value))
            
        # when the property is 'name' update NOHTree and Node Path
        if prop_name == 'name':
            path = node.getAbsolutePath()
            self.nameUpdated.emit(nodeID)
            self.line_path.setText(path)
            
        self.itemUpdated.emit(nodeID)
        
    # private members -----------------------------------------------------------
    def add_property(self, group, name, value, type, readonly):
        # type is boolean
        if type == 'bool':
            item = self.boolManager.addProperty(name)   
            self.boolManager.setValue(item, (value == 'true'))
            #self.boolManager.setReadOnly(item, readonly)
            group.addSubProperty(item)

        # type is integer types
        elif type == 'int' or type == 'uint' or type == 'int16' or type == 'uint16' or\
             type == 'int32' or type == 'uint32' or type == 'int64' or type == 'uint64':
            item = self.intManager.addProperty(name)
            #self.intManager.setRange(item, 0, 256)
            self.intManager.setValue(item, int(value))
            self.intManager.setReadOnly(item, readonly)
            group.addSubProperty(item)
            
        # type is floating types (float32 or float64)
        elif type == 'float32' or type == 'float64':
            item = self.doubleManager.addProperty(self.tr(name))
            self.doubleManager.setValue(item, float(value))
            #self.doubleManager.setRange(item, 0, 256)
            self.doubleManager.setReadOnly(item, readonly)
            group.addSubProperty(item)
        
        # type is string types (stirng or uri)
        elif type == 'string' or type == 'uri':
            item = self.stringManager.addProperty(name)
            self.stringManager.setValue(item, value)
            self.stringManager.setReadOnly(item, readonly)
            group.addSubProperty(item)
            
            if name == 'name':
                regExp = QRegExp('^[a-z + A-Z + _]+\\w+$')
                self.stringManager.setRegExp(item, regExp)
        
        # type == color
        elif type == 'color':
            item = self.colorManager.addProperty(name)
            rgba = value.split(',')
            color = QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))
            self.colorManager.setValue(item, color)
            group.addSubProperty(item)
        
        # type == vector2
        elif type == 'vector2': 
            item = self.pointManager.addProperty(name)
            xy = value.split(',')
            pt = QPointF(float(xy[0]), float(xy[1]))
            self.pointManager.setValue(item, pt)
            group.addSubProperty(item)
            
        # type == node or object
        elif type == 'object' or type == 'node':
            NodeID = int(value)
            if name == 'CoordinateSystem':
                item = self.enumManager.addProperty(name)
                self.enumManager.setEnumNames(item, self.CSNames_)
                self.enumManager.setValue(item, self.CSIDtoEnum_[NodeID])
                group.addSubProperty(item)
                
            elif name == 'Material':
                item = self.enumManager.addProperty(name)
                self.enumManager.setEnumNames(item, self.MatNames_)
                self.enumManager.setValue(item, self.MatIDtoEnum_[NodeID])
                group.addSubProperty(item)
            
            
    # ------------------------------------------------------------------------        
    def build_ObjectItem(self, parent, IDtoEnum, EnumToID, Name, DefaultName, index = 0):
        if index == 0:
            IDtoEnum[-1] = index
            EnumToID[index] = -1
            Name.append(DefaultName)
            index = index + 1
            
        for node in parent.getChildren():
            IDtoEnum[node.getID()] = index
            EnumToID[index] = node.getID()
            Name.append(node.getName())
            index = index + 1
            
            self.build_ObjectItem(node, IDtoEnum, EnumToID, Name, DefaultName, index)
            
            
 