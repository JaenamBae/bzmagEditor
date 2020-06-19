self.treeWidget.setItemWidget

http://dzhelil.info/triangle/examples.html
https://matplotlib.org

Triangle 라이브러리의 __init__.py 에서... 아래 7번째 줄이 없음
추가해 주어야지 Trinagle 의 Attribute를 확인할 수 있음
Attribute는 모델링된 형상의 ID (GeomBaseNode의 ID)로 하여 
추후 재질설정 경계설정 등의 일을 할 수 있게함
    fields = (
        ('pointlist', 'vertices', 'double', 2),
        ('segmentlist', 'segments', 'int32', 2),
        ('holelist', 'holes', 'double', 2),
        ('regionlist', 'regions', 'double', 4),
        ('trianglelist', 'triangles', 'int32', 3),
        ('triangleattributelist', 'triangle_attribute', 'double', 1),
        ('trianglearealist', 'triangle_max_area', 'double', 1),
        ('pointmarkerlist', 'vertex_markers', 'int32', 1),
        ('segmentmarkerlist', 'segment_markers', 'int32', 1),
    )