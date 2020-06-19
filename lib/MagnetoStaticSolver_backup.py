# -*- coding: utf-8 -*-
import sys, os, io
import math
import triangle
import csv
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spsolve
import bzmagPy as bzmag
from matplotlib.transforms import (Affine2D)

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
        self.AA_          = []
        self.bb_          = []

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
                                    'A' : np.zeros((dim,dim)),
                                    'b' : np.zeros(dim),
                                    's' : np.zeros(dim)}
        
        # make FE Regions data
        for region in regions:
            headNodeID = int(region[2])
            head = bzmag.getObject(headNodeID)
            material = head.Material
            if material == None:
                materialID = -1
            else:
                materialID = material.getID()
                
            #print('make material', materialID)
            regionID = region[2]
            if regionID not in self.FE_Regions_:
                self.FE_Regions_[regionID] = {'sourceID': 0, 'materialID': materialID}
            
        #print('FE Region:',self.FE_Regions_)
        [bh, b2v] = self.loadBHCurve('35PN210_BH.tab')
        self.FE_Materials_[2] = [bh, b2v]
    
    #-------------------------------------------------------------------------
    def makeElementEquation(self):
        mu0 = 4 * np.pi * 1e-7
        
        print('------------------------------------')
        print(' Generating of the Element Matrices ')
        print('------------------------------------')
        for e in self.FE_Elements_.values():
            #print('------------------------------------')
            #print('  Gathering Element the Properties  ')
            #print('------------------------------------')
            # 요소의 재질데이터 가져오기
            regionID   = e['regionID']
            headNode = bzmag.getObject(regionID)
            
            region     = self.FE_Regions_[regionID]
            sourceID   = region['sourceID']
            materialID = region['materialID']
            
            # 초기값(materialID가 -1일때)
            mu_x = mu0
            mu_y = mu0
            Mx = 0
            My = 0
            
            bh = None
            b2v = None
            # 참조하는 재질이 있으면 셋팅함
            if materialID != -1:
                # 재질 데이터 가져오기
                matNode = bzmag.getObject(int(materialID))
                
                # 투자율 설정
                mur_x = float(matNode.Permeability)
                mur_y = float(matNode.Permeability)
                mu_x = mu0 * mur_x
                mu_y = mu0 * mur_y
                
                if mur_x > 1000 :
                    [bh, b2v] = self.FE_Materials_[2]
                
                # 자화설정
                Magnetization = float(matNode.Magnetization)
                if Magnetization > 0:
                    sMx, sMy = matNode.Mvector
                    Mx = float(sMx)
                    My = float(sMy)
                    
                    uMx = Mx/math.sqrt(Mx*Mx + My*My)
                    uMy = My/math.sqrt(Mx*Mx + My*My)
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
                    
            #print(mu_x, mu_y, Mx, My)
            # 소스 가져오기 --> 추후 구현
            J  = 0
            
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
            
            # 요소 속성 저장
            e['dim']     = dim
            e['J']       = J
            e['mu']      = [mu_x, mu_y]
            e['M']       = [Mx, My]
            e['bh']      = bh
            e['b2v']     = b2v
                
            
            
            #print('-------------------------------')
            #print('  Calculating Element Maxtrix  ')
            #print('-------------------------------')
            # 요소절점의 좌표 얻기
            x = np.zeros(dim)
            y = np.zeros(dim)
            for j, nodeID in enumerate(nodeIDs):
                # 노드 ID를 얻고
                node = self.FE_Nodes_[nodeID]
                
                # 요소 노드 좌표
                x[j], y[j] = node['point']
            
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
            #s = np.zeros(dim)       # source
            #A = np.zeros((dim,dim)) # stiffiness matrix
            #b = np.zeros(dim)       # unknown
            
            s = np.zeros(dim)       # source
            A = np.zeros((dim,dim)) # stiffiness matrix
            b = np.zeros(dim)       # unknown
            
            
            # get integration points and corresponding weights
            [ui, vi, wi] = self.get_integration_weight(dim)
            
            # do integration
            for i, w in enumerate(wi): 
                u = ui[i]
                v = vi[i]
                [N, dNdu, dNdv, detJ, dNdx, dNdy] = self.derivatives_of_shape_functions(u, v, x, y, dim)
                
                '''
                for j in range(dim):
                    #source term
                    b[j] = b[j] + N[j]*J*detJ*w
                    # stiffiness matrix
                    for k in range(dim):
                        A[j,k] = A[j,k] + (1.0/mu_x*dNdx[j]*dNdx[k] + 1.0/mu_y*dNdy[j]*dNdy[k])*detJ*w
                '''
                Nt = np.transpose(N)
                gradN = np.array([dNdx, dNdy])
                gradNt = np.transpose(gradN)
                mu = np.array([[mu_y, 0], [0, mu_x]])   # 투자율 텐서 ; y, x 순서 주의!
                inv_mu = np.linalg.inv(mu)
                M = np.array([My/mu_y, -Mx/mu_x])       # 자화벡터 ; 좌측 y, x 순서 및 부호 주의!
                
                # stiffiness matrix
                A = A + np.matmul(gradNt, np.matmul(inv_mu, gradN)) * detJ*w
                
                # source term (Current Density and Magnetization)
                b = b + (Nt*J - np.dot(gradNt, M)) * detJ*w
            
            #AA = NR_Jacobian(A, b, 
            # 계산결과 저장
            e['A'] = A
            e['b'] = b
            '''
            print(A)
            print(b)
            #print(c)
            #return
            
            # 하기는 1차 삼각형 요소기반 글로벌 좌표계에서 정식화한 수식으로 Stiffiness 행렬을 구해 본 것
            # 상기 결과와 동일함
            # 요소면적
            delta = (x[0]*y[1] + x[1]*y[2] + x[2]*y[0] - (y[0]*x[1] + y[1]*x[2] + y[2]*x[0])) / 2.0
            
            # 계수 행렬을 위한 계수들 계산
            a = np.zeros(3)
            a[0] = x[1]*y[2] - x[2]*y[1]
            a[1] = x[2]*y[0] - x[0]*y[2]
            a[2] = x[0]*y[1] - x[1]*y[0]
            
            b = np.zeros(3)
            b[0] = y[1] - y[2]
            b[1] = y[2] - y[0]
            b[2] = y[0] - y[1]
            
            c = np.zeros(3)
            c[0] = x[2] - x[1]
            c[1] = x[0] - x[2]
            c[2] = x[1] - x[0]
            
            # 계수행렬 만들기
            K = np.zeros((3,3))
            K[0,0] = ( (1.0/mu_y * b[0] * b[0]) + (1.0/mu_x * c[0] * c[0]) ) / ( 4.0 * delta )
            K[0,1] = ( (1.0/mu_y * b[0] * b[1]) + (1.0/mu_x * c[0] * c[1]) ) / ( 4.0 * delta )
            K[0,2] = ( (1.0/mu_y * b[0] * b[2]) + (1.0/mu_x * c[0] * c[2]) ) / ( 4.0 * delta )
            K[1,0] = ( (1.0/mu_y * b[1] * b[0]) + (1.0/mu_x * c[1] * c[0]) ) / ( 4.0 * delta )
            K[1,1] = ( (1.0/mu_y * b[1] * b[1]) + (1.0/mu_x * c[1] * c[1]) ) / ( 4.0 * delta )
            K[1,2] = ( (1.0/mu_y * b[1] * b[2]) + (1.0/mu_x * c[1] * c[2]) ) / ( 4.0 * delta )
            K[2,0] = ( (1.0/mu_y * b[2] * b[0]) + (1.0/mu_x * c[2] * c[0]) ) / ( 4.0 * delta )
            K[2,1] = ( (1.0/mu_y * b[2] * b[1]) + (1.0/mu_x * c[2] * c[1]) ) / ( 4.0 * delta )
            K[2,2] = ( (1.0/mu_y * b[2] * b[2]) + (1.0/mu_x * c[2] * c[2]) ) / ( 4.0 * delta )
            
            # 구동벡터 만들기
            f = np.zeros(3)
            f[0] = (J * delta / 3.0) + (( (c[0] * Mx/mu_x) - (b[0] * My/mu_y) ) / (2.0)) 
            f[1] = (J * delta / 3.0) + (( (c[1] * Mx/mu_x) - (b[1] * My/mu_y) ) / (2.0)) 
            f[2] = (J * delta / 3.0) + (( (c[2] * Mx/mu_x) - (b[2] * My/mu_y) ) / (2.0)) 
            
            print(K)
            print(f)
            return
            '''

        return
    #-------------------------------------------------------------------------
    def NR_Jacobian(self, A, b, BHcurve):
        pass
        
    #-------------------------------------------------------------------------
    def loadBHCurve(self, file):
        mu0 = 4*np.pi*1e-7

        f = open(file, 'r', encoding='utf-8')
        rdr = csv.reader(f, delimiter='\t')

        bh_data = list()
        for line in rdr:
            H = float(line[0])
            B = float(line[1])
            bh_data.append([H, B])
        f.close()
        
        # CubicSpline 보간을 위해 기울기가 mu0 인 데이터 점을 3개 더 추가한다.
        for i in range(3):
            H_ex = H * (1.5 + i*0.5)
            B_ex = B + mu0*(H_ex - H)
            bh_data.append([H_ex, B_ex])
            H = H_ex
            B = B_ex
        
        BHCurve = np.array(bh_data)
        # H data [A/m]
        data_x = BHCurve[:,0]
        # B data [Tesla]
        data_y = BHCurve[:,1]
        inter_bhcurve = CubicSpline(data_x, data_y, bc_type='natural', extrapolate=bool)
        
        # B2-v 데이터 생성 --> 선형보간을 해야 맞나? 스플라인보간을 해야 맞나?
        b2v_data = list()
        for B, H in bh_data:
            if B > 0:
                v = H/B
                B2 = B**2
                b2v_data.append([B2, v])
        B2vCurve = np.array(b2v_data)
        # B2 data
        data_x = B2vCurve[:,0]
        # v data
        data_y = B2vCurve[:,1]
        inter_b2vcurve = CubicSpline(data_x, data_y, bc_type='natural', extrapolate=bool)
        
        return [inter_bhcurve, inter_b2vcurve]
        
    #-------------------------------------------------------------------------
    def Snm(self, dim, x, y, m, n, A_prev, B2v):
        # get integration points and corresponding weights
        [ui, vi, wi] = self.get_integration_weight(dim)
        
        S = 0
        # do integration
        for i, w in enumerate(wi): 
            u = ui[i]
            v = vi[i]
            [N, dNdu, dNdv, detJ, dNdx, dNdy] = self.derivatives_of_shape_functions(u, v, x, y, dim)
            Nt = np.transpose(N)
            gradN = np.array([dNdx, dNdy])
            gradNt = np.transpose(gradN)
            mu = np.array([[mu_y, 0], [0, mu_x]])   # 투자율 텐서 ; y, x 순서 주의!
            inv_mu = np.linalg.inv(mu)
            
            # stiffiness matrix coefficient S[m,n]
            S = S + np.matmul(gradNt[m, :], np.matmul(inv_mu, gradN[:, n])) * detJ*w
            
        return S
    
    #-------------------------------------------------------------------------
    # NR-Method 를 위한 계수행렬 구하는 sub 함수
    # dim : 요소에 따른 보간 차수
    # x,y : 요소를 이루는 절점의 좌표 (차원은 dim)
    # m,n : 계수행렬의 인덱스 m행, n열
    # A_prev : 이전계산된 벡터포텐셜(미지의 값)
    # B2v : 요소가 참조하는 B2-v데이터
    def fnm(self, dim, x, y, m, n, A_prev, B2v):
        # integration points dim
        # get integration points and corresponding weights
        [ui, vi, wi] = self.get_integration_weight(dim)
        
        # gradN 은 dim x 2 행렬이며,
        # 차원의 dim은 요소형태에 따른 보간차수, 2는 2차원 공간을 뜻함
        # do integration
        for i, w in enumerate(wi): 
            u = ui[i]
            v = vi[i]
            [N, dNdu, dNdv, detJ, dNdx, dNdy] = self.derivatives_of_shape_functions(u, v, x, y, dim)
            
            # gradN
            gradN = np.array([dNdx, dNdy])
            
            # Transpose of gradN
            gradNt = np.transpose(gradN)
        
            # gradNm 은 열벡터 1x2
            gradNm = gradN[m, :]
            # gradNn 또한 열벡터 1x2
            gradNn = gradN[n, :]
        
            # A_prev는 벡터포텐셜이며 열벡터 1 x dim 임
            # B = grad(NA)
            gradNA = np.matmul(gradNt, np.transpose(A_prev))
            B = gradNA[i]
            
            
            dvdB2 = B2v.derivative(1)
            Ekm = np.matmul(gradNm, gradNA)
            Ekn = np.matmul(gradNn, gradNA)
            ff = ff +  (Ekm * Ekn * dvdB2(B) * detJ) * w
        
        ff = ff*2
        return ff

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
        
        # 하기 연산은 for 루프 없이 진행할 수 있을 듯 (구현이 맞는지 체크 - 확인!/2019.08.20)
        '''
        for i in range(6):
            J[0,0] = J[0,0] + dNdu[i]*x[i]
            J[0,1] = J[0,1] + dNdu[i]*y[i]
            J[1,0] = J[1,0] + dNdv[i]*x[i]
            J[1,1] = J[1,1] + dNdv[i]*y[i]
        print('J using for-loop:', J)
        '''
        J[0,0] = np.sum(dNdu*x)
        J[0,1] = np.sum(dNdu*y)
        J[1,0] = np.sum(dNdv*x)
        J[1,1] = np.sum(dNdv*y)
        #print('J using np:', J)
        
        # Calcuation of det(J)
        #detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        detJ = np.linalg.det(J)
        
        # Calculation of dNdx, dNdy
        #   [dNdx; dNdy] = J^-1 [dNdu; dNdv] 
        #   dNdx = 1/detJ * (+J22*dNdu - J12*dNdv)
        #   dNdy = 1/detJ * (-J21*dNdu + J11*dNdv)
        
        
        dNdx = np.zeros(dim)
        dNdy = np.zeros(dim)
        # 하기 연산은 for 루프 없이 진행할 수 있을 듯 (구현이 맞는지 체크 - 확인!/2019.08.20)
        '''
        for i in range(dim):
            dNdx[i] = (+J[1,1]*dNdu[i] - J[0,1]*dNdv[i]) / detJ
            dNdy[i] = (-J[1,0]*dNdu[i] + J[0,0]*dNdv[i]) / detJ
        print('dNdx, dNdy using loop:', dNdx, dNdy)
        '''
        
        [dNdx, dNdy] = np.matmul(np.linalg.inv(J), [dNdu, dNdv])
        #print('dNdx, dNdy using np:', dNdx, dNdy)
        
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
    # 요소방정식을 시스템 방정식으로 어셈블리
    # 예) 삼각형요소에는 노드3개 존재(로컬 노드번호:1~3존재) --> Ne1, Ne2, Ne3   
    #     이를 이용한 계수 메트릭스 --> [Ke11, Ke12, Ke13 ; Ke21, Ke22, Ke23; Ke31, Ke32, Ke33]
    #     시스템 방정식은 삼각형 노드의 글로벌 아이디(인덱스)로 구성됨
    #     요소아이디(1~3) Ne1, Ne2, Ne3 글로벌 아이디 N1, N19, M14에 대응된다면
    #     Ke11 --> K[1][1], Ke12 --> K[1][19], Ke13 --> K[1][14] ... 와 같이 대응됨
    #     또한 다른요소에 의해서도 동일한 글로벌 요소행렬 위치를 가질수 있음
    #     즉, K[1][19] 는 위의 예에 보인 요소에 의해 만들어지는 값이기도 하지만 다른 요소에 의해 값이 만들어지기도 함
    #     이를 합치는 과정이 Assembly 과정임
    def assemblyMatrix(self):
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
            A       = e['A']
            b       = e['b']
            for i in range(dim):
                idxSourceRow.append(nodeIDs[i]) 
                idxSourceCol.append(0)
                source.append(b[i])

                for j in range(dim):
                    idxRow.append(nodeIDs[i])
                    idxCol.append(nodeIDs[j])
                    data.append(A[i,j])
                    
            #print(A)
            #return
        # Step.2 희소행렬을 만들기, csr_matrix 라이브러리 활용!
        #        csr_matrix 행렬 생성법을 찾아보면 도움됨
        #        행렬의 같은 위치에 존재하는 항은 모두 더해짐
        self.AA_ = csr_matrix((data,(idxRow,idxCol)))
        self.bb_ = csr_matrix((source,(idxSourceRow,idxSourceCol)))
        
        
        # 잘만들어 졌는지 확인차...
        #nAA = self.AA_.toarray()
        #nbb = self.bb_.toarray()
        #print(np.shape(self.bb_))
        #print(np.shape(nAA), np.shape(nbb))
        
    #-------------------------------------------------------------------------
    def apply_boundaryconditions(self):
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
        print('------------------------------------')
        print('     Solving the System Matrix      ')
        print('------------------------------------')
        x = spsolve(self.AA_, self.bb_, 'NATURAL', True)
        print(x)
        x = spsolve(self.AA_, self.bb_, 'NATURAL', False)
        print(x)
        return x
        
        
# References
# [1] Joao Pedro A. Bastos, "Electromagnetic Modeling by Finite Element Methods"
# [2] Kay Hameyer, Ronnie Belmans, "Numerical Modelling and Design of Electrical Machines and Devices"
