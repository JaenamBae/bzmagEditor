# -*- coding: utf-8 -*-
import sys, os, io
import math
import triangle
import numpy as np
import bzmagPy as bzmag
import math

#-----------------------------------------------------------------------------
class TriangleMesh:
    def __init__(self):
        super(TriangleMesh, self).__init__()
        
        # triangle input data (*.poly)
        self.input_ = {}
        
        # trinagle output data (mesh results)
        self.output_ = {}
        
        #
        self.g_vertices = np.empty(shape=[0, 2])
        self.g_segments = np.empty(shape=[0, 2])
        self.g_holes = np.empty(shape=[0, 2])
        self.g_regions = np.empty(shape=[0, 4])
        
        
    # ------------------------------------------------------------------------
    def generateMesh(self, root_path):
        tri_node = bzmag.get('/sys/server/triangle')
        tri_node.makePolyStructure(root_path)
        n_polyholes = tri_node.getNumberOfPolyHoles()
        # Step1. Polyhole 의 갯수만큼 루프를 돌며 요소 생성을 위한 기저 절점과 세그먼트 생성
        #        (.Poly 파일 구조 생성이라고 보면 편할듯)
        #        g_verices (절점), g_segments(세그먼트), g_holes(홀), g_regions(영역)에 저장 될 것임
        # Step2. 중복되는 기저절점과 세그먼트제거
        #        mergePolyData() 함수를 이용할 것임
        # Step3. 현재 Polyhole을 지정할 수 있는 임의 좌표 <x> <y>를 만들고
        #        이를 활용하여 해당 영역에 <attribute> 과 <maximum area>를 설정
        # Step4. Mesh Generating
        
        dd = tri_node.getDomainArea()
        print('Domain Area', dd)
        print('There are', n_polyholes, 'region')
        for i in range(0, n_polyholes):
            #if i==0 : continue
            # 현재 영역을 구성하는 절점좌표를 받음
            # 튜플형태(x0, y0, x1, y1, ...)로 넘어오게 되는데,
            # 이를 reshape 해서 2차원 배열로 만들어 편리하게 사용할 것임
            v_data = tri_node.getVertices(i)
            nvertices = len(v_data) / 2
            vertices = np.array(v_data).reshape(int(nvertices), 2)
            #print(vertices)
            
            # 마찬가지 segments 데이터도 튜플형태로 넘어오는데
            # (시작점0_ID, 끝점0_ID, 시작점1_ID, 끝점1_ID, ...) 형태로 넘어옴
            # 마찬가지로 reshape 해서 편리하게 사용함
            s_data = tri_node.getSegments(i)
            nsegments = len(s_data) / 2
            segments = np.array(s_data).reshape(int(nsegments), 2)
            #print(segments)
            
            # hole 데이터도 홀의 위치를 나타내는 절점 좌표가
            # (x0, y0, x1, y1, ...) 형태로 넘어옴
            # reshape 해서 편하게 씀
            h_data = tri_node.getHoles(i)
            nholes = len(h_data) / 2
            holes = np.array(h_data).reshape(int(nholes), 2)
            #print('Holes point')
            #print(holes)
            
            # 중복점 제거
            self.eliminateDuplicate(vertices, segments)
        
            # 각 영역의 vertices, segments, holes 데이터를 합침
            # g_vertices, g_segments, g_holes에 저장됨
            self.mergePolyData(vertices, segments)
            
            # 절점 근사화
            #self.approximateVertices()
            
            # 현재 영역을 대표할 수 있는 좌표 <x> <y>를 찾고 싶음
            # 그냥 현재 기저 절점으로 대충 요소생성후 
            # 임의 요소의 중심 좌표를 현재 영역의 대표 점으로 설정할 것임
            input = dict(vertices=vertices, segments=segments, holes=holes)
            if len(holes)==0: del input['holes']
            
            print('Mesh control area of the Region (ID', tri_node.getPolyHolesID(i),') :', tri_node.getPolyHolesMeshArea(i))
            output = triangle.triangulate(input, 'p')
            tris   = output['triangles']
            verts  = output['vertices']
            
            for tri in tris:
                x0, y0 = verts[tri[0], :]
                x1, y1 = verts[tri[1], :]
                x2, y2 = verts[tri[2], :]
                break
            
            # 영역을 지정하기 위한 대표 절점좌표
            xP = (x0+x1+x2)/3
            yP = (y0+y1+y2)/3
            
            # 이제 영역에 <attribute>과 요소컨트롤을 위한 <maximum area>를 설정할 수 있음
            # <attribute>는 영역의 ID를 설정하기로 하며, 
            # <maximum area>는 사용자로부터 받은 값을 설정하면 좋은데
            # 아직 인터페이스가 마련되어 있지 않으므로 임의값 넣기로 함
            region = np.array([xP, yP, tri_node.getPolyHolesID(i), tri_node.getPolyHolesMeshArea(i)])
            self.g_regions = np.concatenate([self.g_regions, [region]])
            
            
        #print(self.g_regions)
        pthole =  tri_node.getPointsOnHoles()
        self.g_holes = np.array(pthole).reshape(int(len(pthole) / 2), 2)
        input = dict(vertices=self.g_vertices, segments=self.g_segments, holes=self.g_holes, regions=self.g_regions)
        #print(input)
        #return
        
        #n_segments = len(self.g_segments)
        #n_vertices = len(self.g_vertices)
        #print('Number of Segments : ', n_segments)
        #print('Number of Vertices : ', n_vertices)
        
        if len(self.g_holes)==0: del input['holes']
        #self.output_ = triangle.triangulate(input, 'pqAaYY')
        self.output_ = triangle.triangulate(input, 'pqAaY')
        #print(self.output_)
        
        
        #with open('shape.poly','w') as data:
        #    data.write(str(self.output_))
        
    # ------------------------------------------------------------------------
    def generateMeshFromPoly(self, vertices, segments, holes, regions, opt = ''):
        self.input_ = dict(vertices=vertices, segments=segments, holes=holes, regions=regions)
        #if segments == None: self.input_.pop('segments')
        #if holes == None: self.input_.pop('holes')
        #if regions == None: self.input_.pop('regions')
        
        #print(self.input_)
        self.output_ = triangle.triangulate(self.input_, opt)
        
        #vets = self.output_['vertices']
        #xs = vets[:, 0]
        #ys = vets[:, 1]

    # ------------------------------------------------------------------------
    def mergePolyData(self, vertices, segments):
        v_size = len(self.g_vertices)
        pt_map = {}
        for k, vertex in enumerate(vertices):
            mask = (vertex == self.g_vertices)
            mask = mask.T
            result = mask[0] & mask[1]
            indices = [i for i,x in enumerate(result) if x == True]
            if len(indices) > 0:
                #print('Duplicated Vertex:', indices)
                pt_map[k] = indices[0]
            else:
                self.g_vertices = np.concatenate([self.g_vertices, [vertex]])
                pt_map[k] = v_size
                v_size = v_size + 1
                
        for segment in segments:
            ss = segment[0]
            ee = segment[1]
            
            new_segment = np.array([pt_map[ss], pt_map[ee]])
            mask = (new_segment == self.g_segments)
            mask = mask.T
            result = mask[0] & mask[1]
            indices = [i for i,x in enumerate(result) if x == True]
            if len(indices) > 0: 
                #print('Duplicated Segment:', indices)
                pass
            
            new_segment = np.array([pt_map[ee], pt_map[ss]])
            mask = (new_segment == self.g_segments)
            mask = mask.T
            result = mask[0] & mask[1]
            indices = [i for i,x in enumerate(result) if x == True]
            if len(indices) > 0: 
                #print('Duplicated Segment:', indices)
                pass
            
            self.g_segments = np.concatenate([self.g_segments, [new_segment]])
            
            
    # ------------------------------------------------------------------------
    def approximateVertices(self):
        max_v = np.max(abs(self.g_vertices))
        max_value = np.max(abs(max_v))
        print(max_value)
        torr = int(-math.log10(max_value * 1e-6))
        for v in self.g_vertices:
            v[0] = round(v[0], torr)
            v[1] = round(v[1], torr)
            
    # ------------------------------------------------------------------------
    # 중복점 제거를 위한 절점 리넘버링 
    # 1) 우선 중복점에 대해 인덱스 리맵핑을 한다
    #    pt_map[idx] = [idx1, idx2, idx3, ...] 형태로 저장할 것임
    #    여기서 idx는 절점 인덱스, idx1~ 은 idx의 점과 동일한 절점의 인덱스
    # 2) 다음으로 pt_map{} 을 이용해 리넘버링을 한다
    # 예)
    # 인덱스(Key) :   0   1   2   3   4
    # 절점값      : 0,0 0,1 0,1 0,2 0,1
    #               ------------------- 
    # 동일점(Val) :   0   1   2   3   4
    #                 2
    #                 4
    # 리맵핑        -------------------
    #                 0,2,4 원소에 0을 집어 넣음
    #                 1 원소에 1을 집어 넣음
    #                 3 원소에 3을 집어 넣음
    # 결과          --------------------
    # 리맵핑 인덱스   0   1   0   3   0                
    # 리넘버링 인덱스 0   1       2   
    def eliminateDuplicate(self, vertices, segments):
        print('Start renumbering of the nodes, # of nodes :', len(vertices))
        multi_idx = []
        pt_map = {}
        for k, vertex in enumerate(vertices):
            mask = (vertex == vertices)
            mask = mask.T
            result = mask[0] & mask[1]
            indices = [i for i,x in enumerate(result) if x == True]
            
            for index in indices:
                # 중복점이 자신이 아닌경우 진짜 중복이다
                # 단, 이미 검사 완료된 중복노드는 제외한다
                if index >= k :
                    # k 번째 인덱스와 같은 절점노드의 인덱스 저장
                    # (k 보다 큰 인덱스에 한함)
                    if k not in pt_map: pt_map[k] = []
                    pt_map[k].append(index)
                    if len(pt_map[k]) > 1:
                        multi_idx.append(k)
        
        #print(pt_map)
        # 상기 동일점(val) 값을 키로 하여 딕셔너리를 만드는데
        # 해당 키에 대한 값은 새로운 넘버링 값이된다
        pt_remap = {}
        i = 0
        countOK = False;
        for value in pt_map.values():
            for idx in value:
                if idx not in pt_remap: 
                    pt_remap[idx] = i
                    countOK = True
            if countOK:
                i = i+1
                countOK = False
        
        #print(pt_remap)
        # 기저절점 재구성
        new_vertices = []
        for key, val in pt_remap.items():
            if len(new_vertices) == val:
                new_vertices.append(vertices[key])

        vertices = np.array(new_vertices)
        print('Complete renumbering of the nodes, # of nodes :', len(vertices))
        
        # 세그먼트에 대해 리넘버링 된 인덱스로 대체한다
        for segment in segments:
            ss = segment[0] # 시작점 인덱스(원본)
            ee = segment[1] # 끝점 인덱스 (원본)
            segment[0] = pt_remap[ss]    # 시작점인덱스 리넘버링
            segment[1] = pt_remap[ee]    # 시작점인덱스 리넘버링
            
    
    # ------------------------------------------------------------------------
    def getMeshData(self):
        return self.output_