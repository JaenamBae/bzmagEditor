import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import (pyqtSlot)
from pyqtcore import (QList)

import numpy as np
import bzmagPy as bzmag

from lib.SurfaceArtist import SurfaceArtist
from lib.CSArtist import CSArtist
from lib.FluxlineArtist import FluxlineArtist
from lib.MeshArtist import MeshArtist

class ViewCanvas(FigureCanvas):
    def __init__(self, parent=None):
        # Figure
        fig = Figure()
        
        # fig를 1행 1칸으로 나누어 1칸안에 넣어줍니다
        ax = fig.add_axes([0,0,1,1])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))
        ax.set_aspect('equal', 'datalim', 'C', True)
        ax.grid()

        # 부모 클래스 초기화
        super(ViewCanvas, self).__init__(fig)
        
        
        # axes 설정
        self.fig_ = fig
        self.axes_ = ax;
        
        # Selected GeomNodes
        self.selectedNodeIDs_ = QList()
        
        # GeomHeadNodeID to ArtistItem
        self.HeadNodeToItem_ = {}
        
        # CSNodeID to ArtistItem
        self.CSNodeToItem_ = {}
        
        # GeomNodeID to its csID
        self.GeomNodeToReferedCS_ = {}
        
        # mesh (ArtistItem)
        self.mesh_ = None
        
        # flux line (ArtistItem)
        self.fluxline_ = None
        
        
    def visible_mesh(self, visible):
        if self.mesh_ == None: return
        if visible == True : 
            self.mesh_.show()
        else :
            self.mesh_.hide()
        self.draw()
            
    def visible_fluxline(self, visible):
        if self.fluxline_ == None: return
        if visible == True : 
            self.fluxline_.show()
        else :
            self.fluxline_.hide()
        self.draw()
            
    def setMeshData(self, mesh):
        if self.mesh_ == None:
            self.mesh_ = MeshArtist(self)
            
        self.mesh_.setMesh(mesh)
        self.draw()
    
    def setPontentials(self, mesh, pot):
        if self.fluxline_ == None:
            self.fluxline_ = FluxlineArtist(self)
        
        self.fluxline_.setPotentials(mesh, pot)
        self.draw()
        
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
        self.draw()
        
    # User Slots--------------------------------------------------------------
    # Signal Generated from NOHTree
    @pyqtSlot(int)
    def addItem(self, nodeID):
        #print("Add Item in the ViewCanvas")
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
            print('Object is Added')
            if node.isCovered():
                item = SurfaceArtist(self)
                self.HeadNodeToItem_[nodeID] = item
                item.setNode(node)
            else:
                item = SurfaceArtist(self)
                self.HeadNodeToItem_[nodeID] = item
                item.setNode(node)
            
        # make a map for CSNode ID to GraphicItem
        if 'CSNode' == node.getTypeName():
            print('CS is Added')
            
            item = CSArtist(self)
            self.CSNodeToItem_[nodeID] = item
            item.setNode(node)
            item.hide()

        self.draw()
        
        
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
        
        self.draw()
    
    # ------------------------------------------------------------------------
    def updateGeomBaseNodeItem(self, nodeID, showCS):
        # Hide CS Artist of the previous node from the canvas 
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
        node = bzmag.getObject(nodeID)
        item = self.CSNodeToItem_[nodeID]
        item.setNode(node)
        
        for geomID, csID in self.GeomNodeToReferedCS_.items():
            if csID == nodeID:
                self.updateItem(geomID, showCS)

        

