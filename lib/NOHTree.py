# -*- coding: utf-8 -*-
import sys, os
    
from PyQt5.QtCore import (
    QRect, 
    QSize, 
    Qt, 
    pyqtSignal, 
    pyqtSlot)

from PyQt5.QtGui import (
    QCursor,
    QIcon
    )
    
from PyQt5.QtWidgets import (
    QAbstractItemView, 
    QTreeWidget, 
    QTreeWidgetItem, 
    QMenu,
    QRubberBand, 
    QMessageBox
    )
    
from pyqtcore import (
    QList)
    
from lib.NOHImageProvider import NOHImageProvider
from lib.NodeCreator import NodeCreator
from lib.PropertyWidget import PropertyWidget
import bzmagPy as bzmag

# ----------------------------------------------------------------------------
class NOHTree(QTreeWidget):
    itemAdded = pyqtSignal(int)
    itemsSelected = pyqtSignal(QList)
    
    def __init__(self, parent=None):
        super(NOHTree, self).__init__(parent)
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setObjectName("treewidget_node")
        self.headerItem().setText(0, "Geometry Tree")
        
        self.setSortingEnabled(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # image provider
        self.imageProvider_ = NOHImageProvider('data/NOHTreeImages')

        # rubber band
        self.rubberBand_ = None
        
        # root Node
        self.rootNode_ = None
        
        # GeomNodeId to Item
        self.GeomNodeIdToItem_ = {}
        
        # Root Item of the Tree
        self.rootItem_ = None
        
        # Selected NodeIDs
        self.selectedNodeIDs_ = QList()
        
        # Direct Select Mode (select item on the tree --> true selected by modeler view --> false)
        self.directSelection_ = True
        
        # popup menu ---------------------------------------------------------
        QIcon.setThemeSearchPaths([os.getcwd() + '/icons'])
        QIcon.setThemeName('Faenza')
        
        self.popup_menu = QMenu(self)
        self.menu_data = [(u'Add Node\tAlt-A', self.menu_addnode, u'list-add'),
        None,
        (u'Rename\tF2', self.menu_rename, u'edit-find-replace'),
        (u'Delete\tDel', self.menu_deletenode, u'list-remove')]
        
        for md in self.menu_data:
            if md == None:
                self.popup_menu.addSeparator()
            else:
                mi = self.popup_menu.addAction(md[0])
                mi.triggered.connect(md[1])
                if md[2]:
                    mi.setIcon(QIcon.fromTheme(md[2]))
        
        
    # public members ---------------------------------------------------------
    def build(self, root_path):
        self.rootNode_ = bzmag.get(root_path)
        
        root_item = QTreeWidgetItem(['/'])
        root_item.setExpanded(True)
        self.rootItem_ = root_item
        
        root_item.setData(1, Qt.UserRole, self.rootNode_)
        root_item.setIcon(0, self.imageProvider_.getImage('bzmag'))
            
        self.addTopLevelItem(root_item)
        
        for node in self.rootNode_.getChildren():
            self.build_children(root_item, node)
        
        
    # Events -----------------------------------------------------------------
    def mousePressEvent(self, event):
        super(NOHTree, self).mousePressEvent(event)
        
        # property
        
        # rubber band
        self.origin = event.pos()
        if not self.rubberBand_:
            self.rubberBand_ = QRubberBand(QRubberBand.Rectangle, self)
        self.rubberBand_.setGeometry(QRect(self.origin, QSize()))
        self.rubberBand_.show()

    # ------------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        if self.rubberBand_:
            self.rubberBand_.setGeometry(\
            QRect(self.origin, event.pos()).normalized())
  
        super(NOHTree, self).mouseMoveEvent(event)

    # ------------------------------------------------------------------------
    def mouseReleaseEvent(self, event):
        super(NOHTree, self).mouseReleaseEvent(event)
  
        if self.rubberBand_:
            self.rubberBand_.hide()
  
        self.viewport().update()

    # ------------------------------------------------------------------------
    def contextMenuEvent(self, event):
        self.popup_menu.popup(QCursor.pos())

    # ------------------------------------------------------------------------
    def update(self):
       pass
        
        
    # Slots ------------------------------------------------------------------
    def menu_addnode(self):
        # parent node path (bzmagPy)
        current_item = self.currentItem()
        node = current_item.data(1, Qt.UserRole)
        tpath = node.getAbsolutePath()
        
        nc = NodeCreator(self)
        nc.edit_path.setText(tpath)

        # OK button
        if nc.exec():
            node_name = str(nc.edit_name.text())
            node_path = str(nc.edit_path.text())
            node_type = str(nc.edit_type.text())
            
            # Add bzMag Node
            new_node = None
            if bzmag.isNode(node_type):
                parent_node = bzmag.get(node_path)
                bzmag.pushcwn(parent_node)
                new_node = bzmag.new(node_type, node_name)
                bzmag.popcwn()
                
                # Add QTreeWidgetItem
                if 'GeomHeadNode' == node.getTypeName() and \
                   'geom' == node.getName():
                    child_item = self.add_child_item(current_item, new_node)
                else:
                    child_item = self.add_child_item(current_item.parent(), \
                                 new_node)
                
                self.setCurrentItem(child_item, 0) 
                print('Node Added')
                
    # ------------------------------------------------------------------------
    def menu_rename(self):
        print ("OnRename")

    # ------------------------------------------------------------------------
    def menu_deletenode(self):
        print ("OnDelete")
    
    # ------------------------------------------------------------------------
    def changeSelectedNode(self):
        if not self.directSelection_:
            return
            
        nodeIDs = QList()
        for item in self.selectedItems():
            node = item.data(1, Qt.UserRole)
            nodeID = node.getID()
            nodeIDs.append(nodeID)

        if nodeIDs != self.selectedNodeIDs_:
            #print('NOH Tree, Selected Nodes:', nodeIDs, self.selectedNodeIDs_)
            self.selectedNodeIDs_ = nodeIDs
            self.itemsSelected.emit(nodeIDs)

    # User Slots--------------------------------------------------------------
    @pyqtSlot(int)
    def updateName(self, nodeID):
        item = self.GeomNodeIdToItem_[nodeID]
        
        node = bzmag.getObject(nodeID)
        item.setText(0, node.getName())
    
    # ------------------------------------------------------------------------
    @pyqtSlot(QList)
    def nodesSelected(self, nodeIDs):
        #print('NOH Tree, Selected Nodes1:', nodeIDs, self.selectedNodeIDs_)
        self.directSelection_ = False
        self.unselectItemAllItem()
        if nodeIDs in self.selectedNodeIDs_:
            return

        #print('NOH Tree, Selected Nodes2:', nodeIDs, self.selectedNodeIDs_)
        for nodeID in nodeIDs:
            if nodeID == None:  
                pass
                
            node = bzmag.getObject(nodeID)
            
            item = self.GeomNodeIdToItem_[node.getID()]
            item.setSelected(True)
            while item.parent() != None:
                item.parent().setExpanded(True)
                item = item.parent()
                
        self.directSelection_ = True
        #self.selectedNodeIDs_ = nodeIDs
        self.changeSelectedNode()
        
    # ------------------------------------------------------------------------
    def unselectItemAllItem(self):
        #print('Unselect ALL')
        self.selectionModel().clearSelection()
            
    # private members---------------------------------------------------------
    def add_child_item(self, parent_item, child_node):
        child_item = QTreeWidgetItem(parent_item)
        name = str(child_node.getName())

        child_item.setText(0, name)
        child_item.setData(1, Qt.UserRole, child_node)
        
        if 'GeomBaseNode' in child_node.getGenerations() or\
           'CSNode' == child_node.getTypeName():
            self.itemAdded.emit(child_node.getID())
            
        return child_item
        
    # ------------------------------------------------------------------------
    def build_children(self, parent_item, child_node):
        print('Build Child: ', child_node)
        
        item = self.add_child_item(parent_item, child_node)
        self.GeomNodeIdToItem_[child_node.getID()] = item
        
        for cc_node in child_node.getChildren():
            if 'GeomBaseNode' in child_node.getGenerations():
                if 'GeomHeadNode' == child_node.getTypeName():
                    self.build_children(item, cc_node)
                else:
                    if 'GeomBooleanNode' in child_node.getGenerations() and \
                       'GeomHeadNode' == cc_node.getTypeName():
                        self.build_children(item, cc_node)
                    else:
                        self.build_children(parent_item, cc_node)
                    
            else:
                self.build_children(item, cc_node)
        
        
    # ------------------------------------------------------------------------
    def set_item_icon(self, item, node):
        g = list(node.getGenerations())
        g.reverse()
        for type_name in g:
            try:
                item.setIcon(0, self.imageProvider_.getImage(type_name))
            except:
                pass
        try:
            item.setIcon(0, self.imageProvider_.getImage(node.getName()))
        except:
            pass
            