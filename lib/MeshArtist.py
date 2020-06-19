#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import bzmagPy as bzmag

#-----------------------------------------------------------------------------
class MeshArtist():
    # ------------------------------------------------------------------------
    # based on Data coordinates
    def __init__(self, parent):
        self.parent_ = parent
        self.mesh_ = None
        self.pot_max_ = 0
        
    # ------------------------------------------------------------------------
    def setMesh(self, mesh):
        # 기존 데이터 삭제
        if self.mesh_ != None:
            for tri in self.mesh_:
                tri.remove()

        if not 'vertices' in mesh:
            return
        if not 'triangles' in mesh:
            return
            
        vets = mesh['vertices']
        triangles = mesh['triangles']
        xs = vets[:, 0]
        ys = vets[:, 1]
        axes = self.parent_.axes_
        self.mesh_ = axes.triplot(xs, ys, triangles, 'g-', lw=0.1, antialiased=False)
        
    # ------------------------------------------------------------------------
    def hide(self):
        if self.mesh_ != None: 
            for tri in self.mesh_:
                tri.set_visible(False)
        
    # ------------------------------------------------------------------------
    def show(self):
        if self.mesh_ != None: 
            for tri in self.mesh_:
                tri.set_visible(True)
        
