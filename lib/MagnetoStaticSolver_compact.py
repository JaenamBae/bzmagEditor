# -*- coding: utf-8 -*-
import sys, os, io
import math
import triangle
import csv
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spsolve
import bzmagPy as bzmag
from matplotlib.transforms import (Affine2D)

def fp(inter_bhcurve, x):
    eps = 1e-4
    return (inter_bhcurve(x+eps) - inter_bhcurve(x))/eps
    
#-----------------------------------------------------------------------------
class MagnetoStaticSolver:
    def __init__(self):
        super(MagnetoStaticSolver, self).__init__()
        self.FE_Elements_ = {}
        self.FE_Nodes_    = {}
        self.FE_Source_   = {}
        self.FE_Regions_  = {}
        self.FE_Materials_= {}
        
        # Global Matrix and forcing vector
        self.K_          = []
        self.J_          = []
        
        # Solution (vector potentials)
        self.A_          = []

    #-------------------------------------------------------------------------
    def setMeshData(self, data):
        # 'vertices': array([[x, y],...])
        verts        = data['vertices']
        
        # 'segments': array([[NodeID1, NodeID2], ...], dtype=int32)
        
        # 'vertex_markers': array([[0],...], dtype=int32)
        verts_marker = data['vertex_markers']
        
        #'triangles': array([[NodeID1, NodeID2, NodeID3],...], dtype=int32)
        tris         = data['triangles']
        
        #'triangle_attribute': array([[RegionID],...])
        tris_attr    = data['triangle_attribute']
        
        # 'regions': array([[x, y, RegionID, Area],...])
        regions      = data['regions']
        
        # make FE Nodes data
        for i, vert in enumerate(verts):
            self.FE_Nodes_[i] = {'point': vert, 'marker': int(verts_marker[i])}
        
        # make FE Elements data
        for i, tri in enumerate(tris):
            dim = len(tri)
            self.FE_Elements_[i] = {'dim':dim, 
                                    'nodes': tri, 
                                    'J': 0 , 
                                    'mu_x' : 1, 
                                    'mu_y' : 1, 
                                    'Mx' : 0, 
                                    'My' : 0, 
                                    'regionID' : int(tris_attr[i]),
                                    'SS' : np.zeros((dim,dim)),
                                    'A' : np.zeros(dim),
                                    'Q' : np.zeros(dim)}
        
        # make FE Regions data
        for region in regions:
            regionID = int(region[2])
            head = bzmag.getObject(regionID)
            material = head.Material
            if material == None:
                materialID = -1
            else:
                materialID = material.getID()
                
            #print('make material', materialID)
            if regionID not in self.FE_Regions_:
                self.FE_Regions_[regionID] = {'sourceID': 0, 'materialID': materialID}
            
        #print(self.FE_Regions_)
                
    #-------------------------------------------------------------------------
    def setMaterials(self, matNodes):
        # 아이디 -1은 vaccume 재질이다 (기본 재질)
        self.FE_Materials_[-1] = {'sigma': 0, \
                                  'M': 0, \
                                  'dirM': (0,0), \
                                  'mur': 1, \
                                  'BH': None, \
                                  'B2v': None}
                                  
        # 이제 재질 데이터를 만든다
        for node in matNodes.getChildren():
            if 'MaterialNode' == node.getTypeName():
                nodeID = node.getID()
                BH = None
                B2v = None
                # 이하는 테스트 코드임
                # 임의로 투자율이 1000이 넘으면 비선형 재질로 간주해 봄
                mur = float(node.Permeability)
                if mur > 1000:
                    [BH, B2v] = self.loadBHCurve('35PN210_BH.tab')
                    
                self.FE_Materials_[nodeID] = {'sigma': float(node.Conductivity), \
                                              'M': float(node.Magnetization), \
                                              'dirM': (float(node.Mvector[0]), float(node.Mvector[1])), \
                                              'mur': mur, \
                                              'BH': BH, \
                                              'B2v': B2v}

    #-------------------------------------------------------------------------
    # mu0를 나중에 고려하자...
    def loadBHCurve(self, file):
        mu0 = 4*np.pi*1e-7

        f = open(file, 'r', encoding='utf-8')
        rdr = csv.reader(f, delimiter='\t')

        bh_data = list()
        for line in rdr:
            H = float(line[0])
            B = float(line[1])
            bh_data.append([B, H])
        f.close()
        
        # CubicSpline 보간을 위해 기울기가 mu0 (1로 가정하고, 추후 보상)인 데이터 점을 3개 더 추가한다.
        for i in range(20):
            H_ex = H * (1.5 + i*0.5)
            B_ex = B + mu0*(H_ex - H)
            bh_data.append([B_ex, H_ex])
            H = H_ex
            B = B_ex
        
        BHCurve = np.array(bh_data)
        # B data [Tesla]
        data_x = BHCurve[:,0]
        # H data [A/m]
        data_y = BHCurve[:,1]
        #inter_BHcurve = CubicSpline(data_x, data_y, bc_type='natural', extrapolate=bool)
        inter_BHcurve = interp1d(data_x, data_y, bounds_error=False, fill_value=(data_x[0],data_x[-1]))
        
        # B2-v 데이터 생성 --> 선형보간을 해야 맞나? 스플라인보간을 해야 맞나?
        b2v_data = list()
        for B, H in bh_data:
            if B == 0: v = 0
            else : v = H/B
            
            B2 = B**2
            b2v_data.append([B2, v])
        B2vCurve = np.array(b2v_data)
        # B2 data
        data_x = B2vCurve[:,0]
        # v data
        data_y = B2vCurve[:,1]
        #inter_B2vcurve = CubicSpline(data_x, data_y, bc_type='natural', extrapolate=bool)
        inter_B2vcurve = interp1d(data_x, data_y)
        
        return [inter_BHcurve, inter_B2vcurve]
    

    #-------------------------------------------------------------------------
    # Reference : [1], pp.171~172
    def derivatives_of_shape_functions(self, u, v, x, y, dim):
        # Calculation of shape function and their derivatives
        [N, dNdu, dNdv] = self.shape_function(u, v, dim)
        
        # Calculate of Jacobian
        #   J   = [∂x/∂u, ∂y/∂u ; ∂x/∂v, ∂y/∂v]
        #   J11 = ∑(∂Ni/∂u * xi)
        #   J12 = ∑(∂Ni/∂u * yi)
        #   J21 = ∑(∂Ni/∂v * xi)
        #   J22 = ∑(∂Ni/∂v * yi)
        J = np.zeros((2, 2))
        J[0,0] = np.sum(dNdu*x)
        J[0,1] = np.sum(dNdu*y)
        J[1,0] = np.sum(dNdv*x)
        J[1,1] = np.sum(dNdv*y)
        
        # Calcuation of det(J)
        detJ = np.linalg.det(J)
        
        # Calculation of dNdx, dNdy
        dNdx = np.zeros(dim)
        dNdy = np.zeros(dim)
        [dNdx, dNdy] = np.matmul(np.linalg.inv(J), [dNdu, dNdv])
        
        return [N, dNdu, dNdv, detJ, dNdx, dNdy]
    
    #-------------------------------------------------------------------------
    # Reference : [1], pp.172
    def shape_function(self, u, v, dim):
        # shape function
        N    = np.array([])
        
        # derivative : dN/du
        dNdu = np.array([])
        
        # derivative : dN/dv
        dNdv = np.array([])
        
        # a common value for the below calculations
        t = 1-u-v
        
        # triangular element (linear)
        if dim == 3:
            N =    np.array([t, u, v])
            dNdu = np.array([-1, 1, 0])
            dNdv = np.array([-1, 0 ,1])
        
        # quadrilateral element (bi-linear)
        elif dim == 4:
            N    = np.array([(1-u)*(1-v)/4, (1+u)*(1-v)/4, (1+u)*(1+v)/4, (1-u)*(1+v)/4])
            dNdu = np.array([(-1+v)/4, (1-v)/4, (1+v)/4, (-1-v)/4])
            dNdv = np.array([(-1+u)/4, (-1-u)/4, (1+u)/4, (1-u)/4])
        
        # triangular element (quadratic)
        elif dim == 6:
            N    = np.array([-t*(1-2*t), 4*u*t, -u*(1-2*u), 4*u*v, -v*(1-2*v), 4*v*t])
            dNdu = np.array([1-4*t, 4*(t-u), -1+4*u, 4*v, 0, -4*v])
            dNdv = np.array([1-4*t, -4*u, 0, 4*u, -1+4*v, 4*(t-v)])
        
        # quadrilateral element (quadratic)
        elif dim == 8:
            N    = np.array([-(1-u)*(1-v)*(1+u+v)/4, \
                              (1-u*u)*(1-v)/2, \
                             -(1+u)*(1-v)*(1-u+v)/4, \
                              (1+u)*(1-v*v)/2, \
                             -(1+u)*(1+v)*(1-u-v)/4, \
                              (1-u*u)*(1+v)/2,\
                             -(1-u)*(1+v)*(1+u-v)/4,\
                              (1-u)*(1-v*v)/2])
            dNdu = np.array([ (1-v)*(2*u+v)/4, \
                             -(1-v)*u, \
                              (1-v)*(2*u-v)/4, \
                              (1-v*v)/2, \
                              (1+v)*(2*u+v)/4, \
                             -(1+v)*u, \
                              (1+v)*(2*u-v)/4, \
                             -(1-v*v)/2])
            dNdv = np.array([ (1-u)*(u+2*v)/4, \
                             -(1-u*u)/2, \
                             -(1+u)(u-2*v)/4, \
                             -(1+u)*v, \
                              (1+u)*(u+2*v)/4, \
                              (1-u*u)/2, \
                             -(1-u)(u-2*v)/4, \
                             -(1-u)*v])
                     
        return [N, dNdu, dNdv]
    
    #-------------------------------------------------------------------------
    # Gauss-Legendre integration: Weights and evaluation points for integration on triangle
    # Reference : [1], pp. 173-174
    # where, n is number of integration points
    def get_integration_weight(self, dim):
        # integration points n
        n = 0
        if   dim == 3 : n = 1   # linear triangle
        elif dim == 6 : n = 3   # quadratic triangle
        elif dim == 4 : n = 4   # bi-linear quadrilateral
        elif dim == 8 : n = 4   # quadratic quadrilateral
        
        # 1차 삼각형 요소
        if n == 1:
            ui = [1/3]  # 0 < u < 1
            vi = [1/3]  # 0 < v < 1
            wi = [1/2]
            
        # 2차 삼각형 요소
        elif n == 3:
            ui = [1/6, 2/3, 1/6]  # 0 < u < 1
            vi = [1/6, 1/6, 2/3]  # 0 < v < 1
            wi = [1/6, 1/6, 1/6]
        
        # 사각형요소 (3차 다항식까지 유효함)
        elif n == 4:
            sqrt3 = math.sqrt(3)
            ui = [1/sqrt3, 1/sqrt3, -1/sqrt3, -1/sqrt3] # -1 < u < 1
            vi = [1/sqrt3, -1/sqrt3, 1/sqrt3, -1/sqrt3] # -1 < v < 1
            wi = [1, 1, 1, 1]
            
        return [ui, vi, wi]
    
    
    #-------------------------------------------------------------------------
    # 요소방정식 만들기
    def makeElementEquation(self):
        mu0 = 4 * np.pi * 1e-7
        
        print('------------------------------------')
        print('  Generating the Element Matrices   ')
        print('------------------------------------')
        #print('Number of Elements:', len(self.FE_Elements_))
        for e in self.FE_Elements_.values():
            # 요소의 재질데이터 가져오기
            regionID   = e['regionID']
            headNode = bzmag.getObject(regionID)
            
            region     = self.FE_Regions_[regionID]
            sourceID   = region['sourceID']
            materialID = region['materialID']
            
            # ----------------------------------------------------------------
            # 재질 설정
            # 참조하는 재질이 있으면 셋팅함
            mat = self.FE_Materials_[int(materialID)]
            
            # 투자율 설정
            # 실제로 이방성 재질을 고려하게끔 수식을 유도했으나,
            # 프로그램에서 아직 이방성 재질을 설정할 수 있도록 만들어지지 않아
            # 이방성 재질에 대한 수식을 사용하되 nur_x 와 nur_y 가 같게 설정한다
            # --> 즉 등방성 재질을 사용한 경우의 해석이다
            nu   = 1 / (mat['mur']*mu0)
            
            # BH데이터 설정
            BH  = mat['BH'] # x축 B,   y축 H
            B2v = mat['B2v']# x축 B^2, y축 v
            
            # 자화설정
            Mx = 0
            My = 0
            Magnetization = mat['M']
            if Magnetization > 0:
                vMx, vMy = mat['dirM']
                vM = math.sqrt(vMx*vMx + vMy*vMy)
                uMx = vMx/vM
                uMy = vMy/vM
                Mx = Magnetization * uMx
                My = Magnetization * uMy
                
                # 영역이 참조하는 좌표계를 적용한다
                cs = headNode.CoordinateSystem
                m11, m12, m13, m21, m22, m23 = cs.getTransformation()
                mtx = np.array([[m11, m12, 0],
                                [m21, m22, 0],
                                [0,   0,   1]])
                refcs = Affine2D(matrix=mtx)
                Mx, My = refcs.transform((Mx, My))
            
                
            # ----------------------------------------------------------------
            # 소스 설정
            # 소스 가져오기 --> 추후 구현
            J  = 0
            
            # ----------------------------------------------------------------
            # 요소방정식 생성
            # 요소를 이루는 절점 ID 가져오기
            nodeIDs = e['nodes']
            
            # 추후 어셈블리 시 노드 ID가 필요함
            # 노드 ID를 어셈블리 시 행렬 인덱스와 일치시킬 예정임
            # 참고) 추후 주기경계조건 처리를 해야하는 경우 주기경계 상의 노드 절점은 따로 처리되므로
            #       Master/Slave 상의 노드 중 Slave 노드가 Master 노드로 맵핑됨
            #       결국 Assembly 시 Slave 노드는 존재하지 않는 노드가 되므로 
            #       이들은 행렬 인덱스상 가장 마지막에 위치해야 편리함 (아닐지도 모르고...;;)
            # 2019.08.19   
            # 요소 형태 알아내기(절점의 수로 알아냄)
            dim = len(nodeIDs)
            
            # 반복법에 있어 이전 반복의 A(미지수, 벡터포텐셜) 값을 0으로 초기화
            A_prev = np.zeros(dim)
            
            # 소스항 0으로 초기화
            Q      = np.zeros(dim)
            
            # 요소절점의 좌표 얻기
            x = np.zeros(dim)
            y = np.zeros(dim)
            for j, nodeID in enumerate(nodeIDs):
                # 노드 ID를 얻고
                node = self.FE_Nodes_[nodeID]
                
                # 요소 노드 좌표 (단위 변환을 위해 1e-3 곱해줌; mm->m)
                x[j], y[j] = node['point']*1e-3
                
                # 반복법에 있어, 한번이라도 반복이 발생한 경우에는
                # 이전 연산에서의 벡터포텐셜이 self.A_에 저장되어 있음(즉, 크기가 0보다 큼)
                if len(self.A_) > 0 :  A_prev[j] = self.A_[nodeID]
                else : A_prev[j] = 0
            
            
            # 겔러킨법의 적용 (요소방정식)
            # 미분방정식 : "div(v gradA) + J = 0"
            # 여기서 v는 상수 (v는 텐서일 수 있음 -> 이 경우 수식전개 동일한지 고려해 봐야함)
            #        A는 미지수(자기벡터 포텐셜)
            #        J는 소스 항 (전류밀도)
            # 상기 미분방정식에 겔러킨법 적용함 (가중함수를 이용한 적분이 필요함)
            # 1) 정식화 (예, 첫번째 항)
            #    ∫_global{grad(Nt)·v grad(N)A} dxdy
            #    = ∫_local{grad(Nt)·v grad(N)A det(J)} dudv
            # 2) 수치적분 (가우스 르장드르 적분 적용)
            #    ∫(K(u)) du = ∑wi K(ui)
            
            SS  = np.zeros((dim,dim)) # stiffiness matrix
            SS1 = np.zeros((dim,dim)) # stiffiness matrix sub1 ; 본래 계수행렬
            SS2 = np.zeros((dim,dim)) # stiffiness matrix sub2 ; NR 법을 위한 미분항 계수행렬
            
            # get integration points and corresponding weights
            [ui, vi, wi] = self.get_integration_weight(dim)
                
            # do integration
            for i, w in enumerate(wi): 
                u = ui[i]
                v = vi[i]
                [N, dNdu, dNdv, detJ, dNdx, dNdy] = self.derivatives_of_shape_functions(u, v, x, y, dim)
                
                Nt = np.transpose(N)
                gradN = np.array([dNdx, dNdy])
                gradNt = np.transpose(gradN)
                
                # Flux density 
                # [Bx, By] = Curl(NA)
                Bx =  np.matmul(dNdy, A_prev)
                By = -np.matmul(dNdx, A_prev)
                B2 = Bx**2 + By**2
                absB = np.sqrt(B2)
                
                # BH커브 존재시 투자율의 역수 (nu)을 구한다
                if BH != None:
                    # B가 0인 경우에는 H도 0이기 때문에 H/B로 nu를 구할수 없다
                    if absB == 0: nu = BH(0.00001) / 0.00001
                    else : nu = BH(absB) / absB
                
                # 비선형 해석을 위해서는 dv/dB2 이 필요함
                # 선형 재질일때는 dv/dB2 = 0이다
                dvdB2 = 0
                if B2v != None:
                    dvdB2 = fp(B2v, B2)
                    
                    # v-B2 그래프는 단조증가함수이어야 한다! 해의 수렴을 위하여~!
                    if dvdB2 < 0:
                        print('dvdB2 should be positive value!!, B:', Bx, By, B2)
                        
                SS1 = SS1 + np.matmul(gradNt, nu*gradN) * detJ*w
                Ex = -dNdx*By
                Ey =  dNdy*Bx
                E = Ex + Ey
                ff = np.outer(E, E.T)
                SS2 = SS2 + ((2*dvdB2*ff)* detJ*w)
                
                # source term (Current Density and Magnetization)
                M = np.array([nu*My, -nu*Mx])       # 자화벡터 ; 좌측 y, x 순서 및 부호 주의!
                
                # 소스항 벡터
                Q = Q + (Nt*J - np.dot(gradNt, M)) * detJ*w
            
            # the component of the right-hand side depending on the potentials
            Q = Q - np.matmul(SS1, A_prev)
            
            # 디버깅용
            #if Magnetization > 0:
            if mat['mur'] > 1000:
                if not flagStop :
                    print('B', Bx, By)
                    print('A', A_prev)
                    print('nu', nu)
                    print('dvdB2', dvdB2)
                    print('Q', Q)
                    print('SS1', SS1)
                    print('SS2', SS2)
                    flagStop = True
            
            # 요소 속성 저장
            e['dim'] = dim
            e['J']   = J
            e['nu']  = [nu, nu]
            e['M']   = [Mx, My]
            e['BH']  = BH
            e['B2v'] = B2v
            
            # 요소방정식 계수행렬 및 소스 벡터 저장
            e['SS']  = SS1 + SS2
            e['SS1'] = SS1
            e['SS2'] = SS2
            e['Q']   = Q

        return
        

    #-------------------------------------------------------------------------
    # 요소방정식을 시스템 방정식으로 어셈블리
    # 예) 삼각형요소에는 노드3개 존재(로컬 노드번호:1~3존재) --> Ne1, Ne2, Ne3   
    #     이를 이용한 계수 메트릭스 --> [Ke11, Ke12, Ke13 ; Ke21, Ke22, Ke23; Ke31, Ke32, Ke33]
    #     시스템 방정식은 삼각형 노드의 글로벌 아이디(인덱스)로 구성됨
    #     요소아이디(1~3) Ne1, Ne2, Ne3 글로벌 아이디 N1, N19, M14에 대응된다면
    #     Ke11 --> K[1][1], Ke12 --> K[1][19], Ke13 --> K[1][14] ... 와 같이 대응됨
    #     또한 다른요소에 의해서도 동일한 글로벌 요소행렬 위치를 가질수 있음
    #     즉, K[1][19] 는 위의 예에 보인 요소에 의해 만들어지는 값이기도 하지만 다른 요소에 의해 값이 만들어지기도 함
    #     이를 합치는 과정이 Assembly 과정임
    def matrixAssembly(self):
        print('------------------------------------')
        print('    Assembing the Global Matrix     ')
        print('------------------------------------')
        
        idxRow = []
        idxCol = []
        data   = []
        
        idxSourceRow = []
        idxSourceCol = []
        source       = []
        
        # Step1. 각 요소벌 글로벌 행,렬 인덱스 저장 (idxRow, idxCol)
        #        해당 행,렬에 해당하는 값 저장하기 (data)
        for e in self.FE_Elements_.values():
            dim     = e['dim']
            nodeIDs = e['nodes']
            SS      = e['SS']
            Q       = e['Q']
            for i in range(dim):
                idxSourceRow.append(nodeIDs[i]) 
                idxSourceCol.append(0)
                source.append(Q[i])

                for j in range(dim):
                    idxRow.append(nodeIDs[i])
                    idxCol.append(nodeIDs[j])
                    data.append(SS[i,j])
                    
        # Step.2 희소행렬을 만들기, csr_matrix 라이브러리 활용!
        #        csr_matrix 행렬 생성법을 찾아보면 도움됨
        #        행렬의 같은 위치에 존재하는 항은 모두 더해짐
        self.K_ = csr_matrix((data,(idxRow,idxCol)) ,dtype=float)
        self.J_ = csr_matrix((source,(idxSourceRow,idxSourceCol)), dtype=float)
        
    
    #-------------------------------------------------------------------------
    def applyBoundaryConditions(self):
        # 1)고정경계조건의 경우 (Ax = b)
        #   1) 노드번호(n)에 해당하는 행(A행렬의 n행)은 대각성분을 제외하고 모두 0 처리함
        #      대각성분은 1로 처리
        #   2) 노드번호(n)에 해당하는 소스항 (b벡터의 n항)을 주어진 고정경계값으로 처리함
        # 2) 주기경계조건의 경우
        #   ...??
        # 3) 자연경계조건의 경우 
        pass
        
    #-------------------------------------------------------------------------
    def solve(self):
        mu0 = (4 * np.pi * 1e-7)
        print('------------------------------------')
        print('     Solving the System Matrix      ')
        print('------------------------------------')
        
        iter = 1
        while True:
            self.makeElementEquation()
            self.matrixAssembly()
            dA = spsolve(self.K_, self.J_, 'NATURAL', True)
            
            # 최초의 풀이면
            if len(self.A_) == 0:
                self.A_ = dA
                self.err_base = np.linalg.norm(dA)
            
            # 반복 중이면
            else:
                err = np.linalg.norm(dA) / self.err_base
                print('Iteration:', iter, ', Error:', err)
                self.A_ = self.A_ + dA
                if (err<0.01) or (iter>2):
                    break
                    
                iter = iter + 1
        
    #-------------------------------------------------------------------------
    # 희소행렬을 사용하지 않고 시스템행렬을 꾸밈
    def matrixAssembly2(self):
        print('------------------------------------')
        print('    Assembing the Global Matrix 2   ')
        print('------------------------------------')
        nSize = len(self.FE_Nodes_)
        self.K_ = np.zeros((nSize, nSize))
        self.J_ = np.zeros(nSize)
        
        
        for e in self.FE_Elements_.values():
            dim     = e['dim']
            nodeIDs = e['nodes']
            SS      = e['SS']
            Q       = e['Q']
            for i in range(dim):
                nRow = nodeIDs[i]
                self.J_[nRow] = self.J_[nRow] + Q[i]

                for j in range(dim):
                    nCol = nodeIDs[j]
                    self.K_[nRow,nCol] = self.K_[nRow,nCol] + SS[i,j]
                    
    #-------------------------------------------------------------------------
    # 희소행렬을 사용하지 않고 시스템행렬을 꾸밈
    def solve2(self):
        mu0 = (4 * np.pi * 1e-7)
        print('------------------------------------')
        print('     Solving the System Matrix 2    ')
        print('------------------------------------')
        print('self.K_')
        iter = 1
        while True:
            self.makeElementEquation()
            self.matrixAssembly2()
            dA = np.linalg.solve(self.K_, self.J_)
            
            if len(self.A_) == 0:
                self.A_ = dA
                self.err_base = np.linalg.norm(dA)
            else:
                err = np.linalg.norm(dA) / self.err_base
                print('Iteration:', iter, ', Error:', err)
                self.A_ = self.A_ + dA
                if (err<0.01) or (iter>2):
                    break

                iter = iter + 1
        return self.A_
        
# References
# [1] Joao Pedro A. Bastos, "Electromagnetic Modeling by Finite Element Methods"
# [2] Kay Hameyer, Ronnie Belmans, "Numerical Modelling and Design of Electrical Machines and Devices"
