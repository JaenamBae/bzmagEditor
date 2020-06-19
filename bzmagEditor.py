# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bzmagEditor.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from lib.NOHTree import *
from lib.PropertyWidget import *
from lib.ConsoleWidget import *
from lib.ViewCanvas import *
from lib.TriangleMesh import *
from lib.MagnetoStaticSolver import *
import json

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# ------------------------------------------------------------------------------
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Size Policy
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        
        #Main Window Layout
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        MainWindow.setSizePolicy(sizePolicy)
        
        # Centeral Widget
        self.centralwidget = QWidget(MainWindow)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        
        # NOHTree
        self.treewidget_node = NOHTree()
        self.treewidget_node.setSizePolicy(sizePolicy)
        
        # Qt5 Canvas
        self.canvas = ViewCanvas()
        self.canvas.setSizePolicy(sizePolicy)
        
        # Spilter (화면을 좌우로 나눌것임, 좌:트리, 우:뷰)
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setEnabled(True)
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.splitter.addWidget(self.treewidget_node)
        self.splitter.addWidget(self.canvas)
        self.splitter.setStretchFactor(0, 1)    # 화면의 1/3 
        self.splitter.setStretchFactor(1, 2)    # 화면의 2/3
        
        # Vertical Layout (하기 Spilter를 담을 컨테이너)
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        
        #tool bar
        toolbar = NavigationToolbar(self.canvas, MainWindow)
        MainWindow.addToolBar(toolbar)
        
        # Status Bar
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        # Output Window (토킹윈도우)
        self.dockwidget_output = ConsoleWidget(self)
        MainWindow.addDockWidget(Qt.DockWidgetArea(8), self.dockwidget_output)
        
        # Property Window (도킹윈도우)
        self.dockwidget_property = PropertyWidget(self)
        MainWindow.addDockWidget(Qt.DockWidgetArea(2), self.dockwidget_property)
        
        # Menu Bar
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        
        self.menu_file = QMenu(self.menubar)
        self.menu_file.setObjectName("menu_file")
        
        self.menu_view = QMenu(self.menubar)
        self.menu_view.setObjectName("menu_view")
        
        self.menu_bzmag = QMenu(self.menubar)
        self.menu_bzmag.setObjectName("menu_bzmag")

        self.menu_help = QMenu(self.menubar)
        self.menu_help.setObjectName("menu_help")
        
        
        # Menu Items
        self.actionOpen_O = QAction(MainWindow)
        self.actionOpen_O.setObjectName("actionOpen_O")
        self.actionOpen_O.triggered.connect(qApp.quit)
        
        self.actionSave_S = QAction(MainWindow)
        self.actionSave_S.setObjectName("actionSave_S")
        self.actionSave_S.triggered.connect(qApp.quit)
        
        self.actionSave_as_A = QAction(MainWindow)
        self.actionSave_as_A.setObjectName("actionSave_as_A")
        self.actionSave_as_A.triggered.connect(qApp.quit)
        
        self.actionExit_E = QAction(MainWindow)
        self.actionExit_E.setObjectName("actionExit_E")
        self.actionExit_E.triggered.connect(qApp.quit)
        
        
        self.actionViewMesh = QAction(MainWindow)
        self.actionViewMesh.setCheckable(True)
        self.actionViewMesh.setChecked(True)
        self.actionViewMesh.setObjectName("actionViewMesh")
        self.actionViewMesh.toggled.connect(self.showMesh)
        
        self.actionViewFluxline = QAction(MainWindow)
        self.actionViewFluxline.setCheckable(True)
        self.actionViewFluxline.setChecked(True)
        self.actionViewFluxline.setObjectName("actionViewFluxline")
        self.actionViewFluxline.toggled.connect(self.showFluxline)
        
        
        self.actionMesh = QAction(MainWindow)
        self.actionMesh.setObjectName("actionMesh")
        self.actionMesh.triggered.connect(self.meshOperation)
        
        self.actionAnalysis = QAction(MainWindow)
        self.actionAnalysis.setObjectName("actionAnalysis")
        self.actionAnalysis.triggered.connect(self.solve)
        
        
        self.menu_file.addAction(self.actionOpen_O)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.actionSave_S)
        self.menu_file.addAction(self.actionSave_as_A)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.actionExit_E)
        
        
        self.menu_view.addAction(self.actionViewMesh)
        self.menu_view.addAction(self.actionViewFluxline)
        
        
        self.menu_bzmag.addAction(self.actionMesh)
        self.menu_bzmag.addAction(self.actionAnalysis)
        
        
        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_view.menuAction())
        self.menubar.addAction(self.menu_bzmag.menuAction())
        self.menubar.addAction(self.menu_help.menuAction())

        self.retranslateUi(MainWindow)
        
        # Signal priority is important!
        self.treewidget_node.itemSelectionChanged.connect(self.treewidget_node.changeSelectedNode)
        
        self.treewidget_node.itemsSelected.connect(lambda nodeIDs : self.dockwidget_property.nodesSelected(nodeIDs))
        self.treewidget_node.itemsSelected.connect(lambda nodeIDs : self.canvas.nodesSelected(nodeIDs))
        self.treewidget_node.itemAdded.connect(lambda nodeID: self.canvas.addItem(nodeID))
        
        self.dockwidget_property.itemUpdated.connect(lambda nodeID: self.canvas.updateItem(nodeID))
        self.dockwidget_property.nameUpdated.connect(lambda nodeID: self.treewidget_node.updateName(nodeID))
        
        #self.graphics_view.itemsSelected.connect(lambda nodeIDs: self.treewidget_node.nodesSelected(nodeIDs))
        #self.graphics_view.itemUnSelectAll.connect(self.treewidget_node.unselectItemAllItem)
        
        QMetaObject.connectSlotsByName(MainWindow)

    # ------------------------------------------------------------------------
    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menu_file.setTitle(_translate("MainWindow", "파일(&F)"))
        self.menu_view.setTitle(_translate("MainWindow", "보기(&V)"))
        self.menu_bzmag.setTitle(_translate("MainWindow", "bzMag(&b)"))
        self.menu_help.setTitle(_translate("MainWindow", "도움말(&H)"))
        self.dockwidget_output.setWindowTitle(_translate("MainWindow", "출력"))
        self.dockwidget_property.setWindowTitle(_translate("MainWindow", "속성"))
        
        self.actionOpen_O.setText(_translate("MainWindow", "열기 (&O)"))
        self.actionOpen_O.setToolTip(_translate("MainWindow", "열기 (O)"))
        self.actionSave_S.setText(_translate("MainWindow", "저장 (&S)"))
        self.actionSave_as_A.setText(_translate("MainWindow", "다른 이름으로 저장... (&A)"))
        
        self.actionViewMesh.setText(_translate("MainWindow", "메시 보기"))
        self.actionViewFluxline.setText(_translate("MainWindow", "플럭스도 보기"))
        
        self.actionMesh.setText(_translate("MainWindow", "요소 나누기"))
        self.actionAnalysis.setText(_translate("MainWindow", "해석하기"))
        
        self.actionExit_E.setText(_translate("MainWindow", "끝내기(&Q)"))


# ------------------------------------------------------------------------------
class bzMagWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(bzMagWindow, self).__init__(parent)
        
        print('Init bzMag Window')
        
        self.geometryRootName_ = 'geom'
        self.coordinateRootName_ = 'coordinate'
        self.materialRootName_ = 'material'
        
        self.setupUi(self)
        self.setWindowTitle('bzmag Editor v0.1 alpha')
        
        #test 
        print('Prepair Test')
        self.test5();
        print('Test Complete')
        
        self.treewidget_node.build('/')
        bzmag.new('Node', '/'+self.coordinateRootName_)
        bzmag.new('Node', '/'+self.materialRootName_)
        
        self.mesh_ = None
        self.pot_ = None

        self.show()

    # ------------------------------------------------------------------------    
    def createCurve(self, start, end, center, name):
        h   = bzmag.new('GeomHeadNode',      '/'+self.geometryRootName_+'/'+name)
        l   = bzmag.new('GeomCurveNode',     '/'+self.geometryRootName_+'/'+name+'/Curve')
        l.setParameters(start, end, center)
        return l
    
    # ------------------------------------------------------------------------
    def createCircle(self, center, radius, name):
        h   = bzmag.new('GeomHeadNode',      '/'+self.geometryRootName_+'/'+name)
        c   = bzmag.new('GeomCircleNode',    '/'+self.geometryRootName_+'/'+name+'/Circle')
        cov = bzmag.new('GeomCoverLineNode', '/'+self.geometryRootName_+'/'+name+'/Circle/Cover')
        c.setParameters(center, radius)
        return cov
    
    # ------------------------------------------------------------------------    
    def createRectangle(self, point, dx, dy, name):
        h   = bzmag.new('GeomHeadNode',      '/'+self.geometryRootName_+'/'+name)
        r   = bzmag.new('GeomRectNode',      '/'+self.geometryRootName_+'/'+name+'/Rectangle')
        cov = bzmag.new('GeomCoverLineNode', '/'+self.geometryRootName_+'/'+name+'/Rectangle/Cover')
        r.setParameters(point, dx, dy)
        return cov
    
    # ------------------------------------------------------------------------        
    def booleanSubtract(self, lhs, rhs):
        path = lhs.getAbsolutePath()
        sub = bzmag.new('GeomSubtractNode', path+'/Subtract')
        tool_head = rhs.getHeadNode()
        sub.attach(tool_head)
        tool_head.detach()
        sub.attach(tool_head)
        return sub
    
    # ------------------------------------------------------------------------
    def clone(self, obj, name):
        path = obj.getAbsolutePath()
        ct = bzmag.new('GeomCloneToNode', path+'/CloneTo')
        
        bzmag.new('GeomHeadNode', '/'+self.geometryRootName_+'/'+name)
        cf = bzmag.new('GeomCloneFromNode', '/'+self.geometryRootName_+'/'+name+'/CloneFrom')
        cf.setCloneToNode(ct)
        return [ct, cf]
        
    # ------------------------------------------------------------------------
    def move(self, obj, dx, dy):
        path = obj.getAbsolutePath()
        m = bzmag.new('GeomMoveNode', path+'/Move')
        m.setParameters(dx, dy)
        return m
        
    # ------------------------------------------------------------------------
    def rotate(self, obj, angle):
        path = obj.getAbsolutePath()
        r = bzmag.new('GeomRotateNode', path+'/Rotate') 
        r.setParameters(angle)
        return r
        
    # ------------------------------------------------------------------------
    def createCS(self, origin, angle, name, parent=None):
        if parent != None:
            if parent.getTypeName() != 'CSNode' : return None
            
            parent_path = parent.getAbsolutePath()
            h = bzmag.new('CSNode', parent_path + '/' + name)
        else:
            h = bzmag.new('CSNode', '/' + self.coordinateRootName_ + '/' + name)
            
        h.setParameters(origin, angle)
        return h
    
    # ------------------------------------------------------------------------
    def createExpression(self, key, expression, descriptioin):
        expr = bzmag.newobj('Expression')
        if not expr.setKey(key):
            pass
    
    # ------------------------------------------------------------------------
    def createMaterial(self, name):
        h = bzmag.new('MaterialNode', '/'+self.materialRootName_+'/'+name)
        return h
            
    # ------------------------------------------------------------------------
    def meshOperation(self):
        print('Prepair to generate the mesh')
        mesh = TriangleMesh()
        mesh.generateMesh(self.geometryRootName_)
        print('Complete to generate the mesh')
        
        self.mesh_ = mesh.output_
        nmesh = len(self.mesh_['triangles'])
        nvert = len(self.mesh_['vertices'])
        print(nmesh,'elements, ', nvert, ' vertices are generated!!')
        #print('mesh...',self.mesh_)
        self.canvas.setMeshData(mesh.output_)
        #self.canvas.visible_mesh(False)
        
    # ------------------------------------------------------------------------
    def solve(self):
        print('Solving...')
        materials = bzmag.get(self.materialRootName_)
        solver = MagnetoStaticSolver()
        
        # 재질정보 셋팅
        solver.setMaterials(materials)
        
        # 메쉬 정보 셋팅 
        solver.setMeshData(self.mesh_)
        
        # 이하는 추후에...
        self.pot_ = solver.solve2()
        
        # 플럭스도 뿌리기
        self.canvas.setPontentials(self.mesh_, self.pot_)
        #self.canvas.visible_fluxline(False)
        print('Complete to solve!!')
    
    # ------------------------------------------------------------------------
    # 콜백함수
    def showMesh(self, state):
        if state:
            self.canvas.visible_mesh(True)
        else:
            self.canvas.visible_mesh(False)
        
    # ------------------------------------------------------------------------
    # 콜백함수
    def showFluxline(self, state):
        if state:
            self.canvas.visible_fluxline(True)
        else:
            self.canvas.visible_fluxline(False)
        
    # ------------------------------------------------------------------------
    def test1(self):
        # bzMag Test
        h1   = bzmag.new('GeomHeadNode',      '/geom/h1')
        r1   = bzmag.new('GeomRectNode',      '/geom/h1/r1')
        c1   = bzmag.new('GeomCoverLineNode', '/geom/h1/r1/cover')
        ct   = bzmag.new('GeomCloneToNode',   '/geom/h1/r1/cover/cloneTo')
        rot1 = bzmag.new('GeomRotateNode',    '/geom/h1/r1/cover/cloneTo/rotate')
        r1.setParameters('-20, -20','40','40')
        rot1.setParameters('_pi/4')

        h2   = bzmag.new('GeomHeadNode',      '/geom/h2')
        r2   = bzmag.new('GeomCircleNode',    '/geom/h2/c1')
        c2   = bzmag.new('GeomCoverLineNode', '/geom/h2/c1/cover')
        sub2 = bzmag.new('GeomSubtractNode',  '/geom/h2/c1/cover/subtract')
        mov2 = bzmag.new('GeomMoveNode',      '/geom/h2/c1/cover/subtract/move')
        r2.setParameters('0,0', '100')
        sub2.attach(h1)
        mov2.setParameters('50', '90')

        h3   = bzmag.new('GeomHeadNode',      '/geom/h3')
        cf   = bzmag.new('GeomCloneFromNode', '/geom/h3/cloneFrom')
        rot3 = bzmag.new('GeomRotateNode',    '/geom/h3/cloneFrom/rotate')
        cf.setCloneToNode(ct)
        rot3.setParameters('-_pi/3')
        
        h4   = bzmag.new('GeomHeadNode',      '/geom/h4')
        r4   = bzmag.new('GeomRectNode',      '/geom/h4/r1')
        rot4 = bzmag.new('GeomRotateNode',    '/geom/h4/r1/rotate')
        mov4 = bzmag.new('GeomMoveNode',      '/geom/h4/r1/rotate/move')
        c4   = bzmag.new('GeomCoverLineNode', '/geom/h4/r1/rotate/move/cover')
        r4.setParameters('30, 0','10','30')
        rot4.setParameters('_pi/12')
        mov4.setParameters('4','10')

        cs1  = bzmag.new('CSNode',            '/coordinate/cs1')
        cs2  = bzmag.new('CSNode',            '/coordinate/cs1/cs2')

        cs1.setParameters('0, 0', '_pi/3')
        cs2.setParameters('50, 0', '0')

        h1.CoordinateSystem = cs1.getID()
        
    # ------------------------------------------------------------------------
    def test2(self):
        c1 = self.createCircle('0, 0', '100', 'Stator')
        c2 = self.createCircle('0, 0', '90', 'Stator_In')
        c3 = self.createCircle('0, 0', '60', 'Shaft')
        
        ca1 = self.createCircle('0, 0', '70', 'Airgap1')
        ca2 = self.createCircle('0, 0', '70.5', 'Airgap2')
        
        [ct, cf] = self.clone(c2, 'StatorBore')
        sub = self.booleanSubtract(c1, ct)
        print('Clone From Node :', cf)
        m1 = self.move(cf, '10', '10')
        cs1 = self.createCS('0, 0', '_pi/3', 'CS1')
        
        
    # ------------------------------------------------------------------------
    def test3(self):
        c1 = self.createCircle('0, 0', '100', 'Stator')
        c2 = self.createCircle('0, 0', '90', 'Stator_2')
        r1 = self.createRectangle('10, 10' ,'10', '10', 'hole')
        
        print('subtract start')
        s1 = self.booleanSubtract(c1, c2)
        
        print('move start')
        m1 = self.move(s1, '10', '30')
        
        print('rotate start')
        rot1 = self.rotate(m1, '_pi/2')
        
        print('move start')
        mov1 = self.move(r1, '3', '10')
        
        cs1 = self.createCS('5, 0', '_pi/4', 'CS1')
        mov1.getParent().CoordinateSystem = cs1.getID()
        
    # ------------------------------------------------------------------------
    # 가장 간단한 자기회로 해석
    def test4(self):
        # left, top, right, bottom
        r1 = self.createRectangle('-10, -10', '20', '20', 'Core')
        r2 = self.createCircle('0, 0', '10.000001', 'CoreInner')
        r3 = self.createRectangle('0, -1', '15', '2', 'Airgap')
        core1 = self.booleanSubtract(r1, r2)
        core2 = self.booleanSubtract(core1, r3)
        
        background = self.createRectangle('-50, -50', '100', '100', 'Background')
        
        l = self.createCurve('-5,0','5,0','0,0', 'TestCurve')
        print('rotate start')
        rot1 = self.rotate(l, '_pi/2')
        
        print('move start')
        mov1 = self.move(rot1, '3', '10')
        
        cs1 = self.createCS('5, 0', '0', 'CS1')
        cs2 = self.createCS('0, 0', '_pi/30', 'CS2', cs1)
        
        
        mat1 = self.createMaterial('vaccum')
        mat2 = self.createMaterial('copper')
        mat3 = self.createMaterial('core')
        
        l.getParent().CoordinateSystem = cs2.getID()
        
    # ------------------------------------------------------------------------
    # 가장 간단한 자기회로 해석
    def test5(self):
        # left, top, right, bottom
        r1 = self.createRectangle('-10, -10', '20', '20', 'Core')
        r2 = self.createRectangle('-5, -5', '10', '10', 'CoreInner')
        core1 = self.booleanSubtract(r1, r2)
        
        
        r3 = self.createRectangle('5, -1', '5', '2', 'Magnet')
        background = self.createRectangle('-50, -50', '100', '100', 'Background')
        
        cs1 = self.createCS('0, 0', '0', 'CS1')
        
        mat1 = self.createMaterial('Magnet')
        mat1.Magnetization = 1.2
        mat1.Mvector = (0,1)
        mat2 = self.createMaterial('Core')
        mat2.Permeability = 3000
        
        
        r1_head = r1.getHeadNode()
        r1_head.Material = mat2.getID()
        r1_head.RequiredNumberOfElements = 1000
        
        r3_head = r3.getHeadNode()
        r3_head.CoordinateSystem = cs1.getID()
        r3_head.Material = mat1.getID()
        r3_head.RequiredNumberOfElements = 200
        
