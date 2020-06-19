# -*- coding: utf-8 -*-
import sys, os
import math

sys.path.append('libqt5')
    
from PyQt5.QtCore import (
    QRectF, 
    QRect, 
    QSizeF, 
    QSize, 
    Qt, 
    pyqtSignal, 
    pyqtSlot
    )
    
from PyQt5.QtGui import (
    QBrush, 
    QColor, 
    QPainter, 
    QPen, 
    QPolygonF, 
    QTransform
    )
    
from PyQt5.QtWidgets import (
    QGraphicsScene, 
    QGraphicsView, 
    QStyle, 
    QTreeWidgetItem
    )
    
from pyqtcore import (
    QMap, 
    QList
    )
        
from lib.CoordinateSystemItem import (CoordinateSystemItem)
from lib.SurfaceItem import (SurfaceItem)
from lib.MeshItem import (MeshItem)
import bzmagPy as bzmag
import numpy as np


        
#-----------------------------------------------------------------------------
class ViewWidget(QGraphicsView):
    itemsSelected = pyqtSignal(QList)
    itemUnSelectAll = pyqtSignal()
    
    def __init__(self, parent):
        super(ViewWidget, self).__init__(parent)
        print("initializeEvent")
        
        self.gridWidth_ = 10
        self.visibleGrid_ = True
        self.isPanning_ = False
        self.mousePressed_ = False
        
        # Selected GeomNodes
        self.selectedNodeIDs_ = QList()
        
        # GeomHeadNodeID to TreeItem
        self.HeadNodeToItem_ = {}
        
        # CSNodeID to TreeItem
        self.CSNodeToItem_ = {}
        
        # GeomNodeID to its csID
        self.GeomNodeToReferedCS_ = {}

        # scene
        scene = QGraphicsScene(self)
        
        #void   setItemIndexMethod(ItemIndexMethod method)
        scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        
        #void   setSceneRect(qreal x, qreal y, qreal w, qreal h)
        scene.setSceneRect(-16000.0, -16000.0, 32000.0, 32000.0)
        
        #void   setScene(QGraphicsScene *scene)
        self.setScene(scene)
        
        # void  setCacheMode(CacheMode mode)
        self.setCacheMode(QGraphicsView.CacheBackground)
        
        #void   setViewportUpdateMode(ViewportUpdateMode mode)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        
        #void   setRenderHints(QPainter::RenderHints hints)
        self.setRenderHint(QPainter.Antialiasing)
        
        #void   setTransformationAnchor(ViewportAnchor anchor)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        
        #void   setResizeAnchor(ViewportAnchor anchor)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        
        #void   setDragMode(DragMode mode)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        
        #void	setVerticalScrollBarPolicy(Qt::ScrollBarPolicy)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        #void	setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        #void   QGraphicsView::scale(qreal sx, qreal sy)
        self.scale(1.0, -1.0)
        
        self.setMinimumSize(400, 400)
        self.setWindowTitle("bzMag Modeler")
        
        self.globalCS_ = CoordinateSystemItem(self)
        self.globalCS_.setViewMode(2)
        self.globalCS_.setZValue(10000000)
        scene.addItem(self.globalCS_)
        
        self.mesh_ = MeshItem(self)
        scene.addItem(self.mesh_)
        #self.mesh_.hide()

        
    # public function --------------------------------------------------------
    def setGridWidth(self, width):
        self.step_ = width

    # ------------------------------------------------------------------------
    #def clear(self):
    #    self.HeadNodeToItem_.clear()
    #    self.GeomNodeToReferedCS_.clear()
    #    self.scene().clear()
    #    self.selectedNodeIDs_ = QList()
        
    # Event functions --------------------------------------------------------
    def mousePressEvent(self, event):
        pos = event.pos()
        pos = self.mapToScene(pos)
        #print(pos)
        if event.button() == Qt.LeftButton:
            self.mousePressed_ = True
            if self.isPanning_:
                self.setCursor(Qt.ClosedHandCursor)
                self._dragPos = event.pos()
                event.accept()
            else:
                super(ViewWidget, self).mousePressEvent(event)
                
                if event.modifiers() & Qt.ControlModifier:
                    item = self.scene().focusItem()
                    if item != None:
                        item.setSelected(not item.isSelected())
                        
                    nodeIDs = QList()
                    for item in self.scene().selectedItems():
                        node = item.getNode()
                        if node != None:
                            nodeIDs.append(node.getID())
                        
                    self.itemsSelected.emit(nodeIDs)
                    
                else:
                    self.itemUnSelectAll.emit()
                    item = self.scene().focusItem()
                    
                    nodeIDs = QList()
                    if item != None: 
                        nodes = QList()
                        node = item.getNode()
                        nodeIDs.append(node.getID())
                        self.itemsSelected.emit(nodeIDs)

        #else:
        #    super(ViewWidget, self).mousePressEvent(event)

        
    # ------------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        if self.mousePressed_ and self.isPanning_:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.horizontalScrollBar().setValue(\
                self.horizontalScrollBar().value() - diff.x())
            self.verticalScrollBar().setValue(\
                self.verticalScrollBar().value() - diff.y())
            self.resetCachedContent()
            
            self.updateBoundingRectOfGlobalCS()
            event.accept()
        else:
            super(ViewWidget, self).mouseMoveEvent(event)
            
            if self.mousePressed_:
                nodeIDs = QList()
                rect = self.rubberBandRect()
                #for item in self.items(rect):
                
                for item in self.scene().selectedItems():
                    if item.getNode() != None:
                        node = item.getNode()
                        nodeIDs.append(node.getID())
                        
                self.itemsSelected.emit(nodeIDs)


    # ------------------------------------------------------------------------
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ShiftModifier:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.isPanning_ = False
                self.setCursor(Qt.ArrowCursor)
            self.mousePressed_ = False
        
        #else:
        #    super(ViewWidget, self).mouseReleaseEvent(event)

    # ------------------------------------------------------------------------
    def mouseDoubleClickEvent(self, event): pass

    # ------------------------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Shift and not self.mousePressed_:
            self.isPanning_ = True
            self.setCursor(Qt.OpenHandCursor)
            
        elif key == Qt.Key_Plus:
            print('Zoom In')
            self.scaleView(1.2)
            
        elif key == Qt.Key_Minus:
            print('Zoom Out')
            self.scaleView(1 / 1.2)
            
        elif key == Qt.Key_F:
            print('Fit to the Window')
            #self.resetTransform()
            scene = self.scene()
            scene.removeItem(self.globalCS_)
            rc = scene.itemsBoundingRect()
            scene.addItem(self.globalCS_)
            self.fitInView(rc, Qt.KeepAspectRatio )
            self.updateBoundingRectOfGlobalCS()
        else:
            super(ViewWidget, self).keyPressEvent(event)

    # ------------------------------------------------------------------------
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift:
            if not self.mousePressed_:
                self.isPanning_ = False
                self.setCursor(Qt.ArrowCursor)
                
        elif event.key() == Qt.Key_Control:
            self.multipleSelection_ = False
            
        else:
            super(ViewWidget, self).keyPressEvent(event)
    
    # ------------------------------------------------------------------------
    def wheelEvent(self, event):
        self.scaleView(math.pow(2.0, event.angleDelta().y() / 240.0))
        self.updateBoundingRectOfGlobalCS()
    
    # ------------------------------------------------------------------------    
    def resizeEvent(self, event):
        super(ViewWidget, self).resizeEvent(event)

        #set GlobalCS_ Rect
        self.updateBoundingRectOfGlobalCS()
        
    # ------------------------------------------------------------------------
    def drawBackground(self, painter, rect):
        super(ViewWidget, self).drawBackground(painter, rect)
        #rect is based on logical value
        #print(rect)
        
        # step is based on logical value (real type ; floating point)
        step = int(self.gridWidth_)
        
        # pen for the background : pen width is set to zero
        pen = QPen(QColor(200, 200, 255, 125))
        pen.setWidth(1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        
        # draw grid
        if self.visibleGrid_:
            # for truncation error when panning (zoom-in)
            rc = rect.adjusted(-step,-step,step,step)
        
            # draw horizontal grid
            start = int(round((rc.top()) / step) * step)
            stop = int(round((rc.bottom()) / step + 1) * step)
            #print('Horizontal: ',start, stop, step)
            for y in range(start, stop, step):
                painter.drawLine(rc.left(), y, rc.right(), y)
                
            # now draw vertical grid
            start = int(round((rc.left()) / step) * step)
            stop = int(round((rc.right()) / step + 1) * step)
            for x in range(start, stop, step):
                painter.drawLine(x, rc.top(), x, rc.bottom())
            

    # User Slots--------------------------------------------------------------
    # Signal Generated from NOHTree
    @pyqtSlot(int)
    def addItem(self, nodeID):
        scene = self.scene()
        node = bzmag.getObject(nodeID)
        
        # make a map for GeomBaseNode to its refered CS ID
        if 'GeomBaseNode' in node.getGenerations():
            cs = node.CoordinateSystem
            csID = -1
            if cs != None: csID = cs.getID()
            self.GeomNodeToReferedCS_[nodeID] = csID
            
        # make a GraphicItem for the HeadNode and
        # make a map for GeomHeadNode ID to GraphicItem
        if 'GeomHeadNode' == node.getTypeName() and node.IsStandAlone:
            print('Head Added')
            item = SurfaceItem()
            scene.addItem(item)
            self.HeadNodeToItem_[nodeID] = item
            item.setNode(node)
        
        # make a map for CSNode ID to GraphicItem
        if 'CSNode' == node.getTypeName():
            print('CS Added')
            item = CoordinateSystemItem(self)
            item.hide()
            scene.addItem(item)
            self.CSNodeToItem_[nodeID] = item
            item.setNode(node)
            
        
            
    # ------------------------------------------------------------------------
    # update Graphic Item with related GeomHead node
    # Signal Generated from PropertyWidget
    @pyqtSlot(int)
    def updateItem(self, nodeID, showCS=True):
        #print('View Widget, Update Node:', nodeID)
        if nodeID == -1:
            return
        
        # get bzmag Node Object by its ID
        node = bzmag.getObject(nodeID)
        
        # Update GeomNode and related GeomNodes
        if 'GeomBaseNode' in node.getGenerations():
            self.updateGeomBaseNodeItem(nodeID, showCS)
            
        # Update CS node and related GeomNodes
        if 'CSNode' == node.getTypeName():
            self.updateCSNodeItem(nodeID, showCS)
        
        self.scene().update()
    
    # View/Hide of CS Item related to the bznode
    # Signal Generated from NOHTree
    @pyqtSlot(QList)
    def nodesSelected(self, nodeIDs):
        # Previous selected items are removed from the scene
        #print('View Widget, Selected Nodes:', nodeIDs, self.selectedNodeIDs_)
        for nodeID in self.selectedNodeIDs_:
            node = bzmag.getObject(nodeID)
            if 'CSNode' == node.getTypeName():
                item = self.CSNodeToItem_[nodeID]
                item.hide()
                
            if 'GeomBaseNode' in node.getGenerations():
                csID = self.GeomNodeToReferedCS_[nodeID]
                if csID != -1:
                    item = self.CSNodeToItem_[csID]
                    item.hide()
            
            if 'GeomHeadNode' == node.getTypeName() and node.IsStandAlone:
                item = self.HeadNodeToItem_[nodeID]
                item.setSelected(False)
            
        # Current selected items are added to the scene
        for nodeID in nodeIDs:
            node = bzmag.getObject(nodeID)
            if 'CSNode' == node.getTypeName():
                item = self.CSNodeToItem_[nodeID]
                item.show()
                
            if 'GeomBaseNode' in node.getGenerations():
                csID = self.GeomNodeToReferedCS_[nodeID]
                if csID != -1:
                    item = self.CSNodeToItem_[csID]
                    item.show()
                    
            if 'GeomHeadNode' == node.getTypeName() and node.IsStandAlone:
                item = self.HeadNodeToItem_[nodeID]
                item.setSelected(True)
        
        self.selectedNodeIDs_ = nodeIDs
    
    # private function -------------------------------------------------------
    def scaleView(self, scaleFactor):
        factor = self.transform().scale(scaleFactor, scaleFactor)\
                 .mapRect(QRectF(0, 0, 1, 1)).width()

        if factor < 0.07 or factor > 100:
            return

        self.scale(scaleFactor, scaleFactor)
        self.resetCachedContent()
    
    # ------------------------------------------------------------------------
    def updateBoundingRectOfGlobalCS(self):
        rect = self.mapToScene(self.viewport().geometry()).boundingRect()
        self.globalCS_.setRect(rect)
        
    # ------------------------------------------------------------------------
    def updateGeomBaseNodeItem(self, nodeID, showCS):
        # Hide CS Item of the previous node from the scene 
        prev_csID = self.GeomNodeToReferedCS_[nodeID]
        if prev_csID != -1 and showCS:
            item = self.CSNodeToItem_[prev_csID]
            item.hide()
        
        node = bzmag.getObject(nodeID)
        
        # Update referred CoordinateSystem of the GeomNode
        csID = -1
        cs = node.CoordinateSystem
        if cs != None: csID = cs.getID()
        self.GeomNodeToReferedCS_[nodeID] = csID
        if csID != -1 and showCS:
            item = self.CSNodeToItem_[csID]
            item.show()
        
        # Find HeadNode
        if 'GeomHeadNode' == node.getTypeName(): hn = node
        else: hn = node.getHeadNode()
        
        if hn == None: return
            
        # When the HeadNode is not standard alone node
        # ex) it is referred by boolean node
        if hn.IsStandAlone == False:
            parent = hn.getParent()
            self.updateItem(parent.getID(), False)

        # Update Top Level HeadNode
        else:
            item = self.HeadNodeToItem_[hn.getID()]
            if item != None: item.setNode(hn)
            
            if hn.IsHide == True: item.hide()
            else: item.show()
                
        # Update CloneFromNode
        while node != None: 
            n = None
            for n in node.getChildren():
                if n.getTypeName() == 'GeomCloneToNode':
                    for o in n.getClonedNodes():
                        parent = o.getParent()
                        self.updateItem(parent.getID(), False)
                    
            node = n  
            
    # ------------------------------------------------------------------------        
    def updateCSNodeItem(self, nodeID, showCS):
        for geomID, csID in self.GeomNodeToReferedCS_.items():
            if csID == nodeID:
                self.updateItem(geomID, showCS)
                
            