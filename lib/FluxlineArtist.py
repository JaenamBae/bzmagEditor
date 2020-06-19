#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import bzmagPy as bzmag

#-----------------------------------------------------------------------------
class FluxlineArtist():
    # ------------------------------------------------------------------------
    # based on Data coordinates
    def __init__(self, parent):
        self.parent_ = parent
        self.fluxline_ = None
        self.contour_ = None
        self.nLevels_ = 20
        self.pot_min_ = 0
        self.pot_max_ = 0
        
    # ------------------------------------------------------------------------
    def setPotentials(self, mesh, pot):
        # 기존 데이터 삭제
        if self.fluxline_ != None:
            for flux in self.fluxline_.collections:
                flux.remove()
        if self.contour_ != None:
            for flux in self.contour_.collections:
                flux.remove()
            
        axes = self.parent_.axes_
        
        vets = mesh['vertices']
        triangles = mesh['triangles']
        xs = vets[:, 0]
        ys = vets[:, 1]
        
        self.pot_min_ = min(pot)
        self.pot_max_ = max(pot)
        #pot_del = (pot_max - pot_min) / 20
        #levels = np.arange(pot_min, pot_max, pot_del)
        cmap = cm.get_cmap(name='Blues', lut=None)
        self.fluxline_ = axes.tricontour(xs, ys, triangles, pot, levels=20, linewidths=0.2, colors='k')
        self.contour_ = axes.tricontourf(xs, ys, triangles, pot, levels=20, cmap="RdBu_r")
        
        # Legend
        #cbaxes = inset_axes(self.axes_, width="3%", height="40%", loc='upper left') 
        #self.fig_.colorbar(self.contour_, cax=cbaxes, shrink=0.5)
    
    def setLevels(self, levels):
        self.nLevels_ = levels
        self.fluxline_.set_levels(levels)
        self.contour_.set_levels(levels)
        
    # ------------------------------------------------------------------------
    def hide(self):
        if self.fluxline_ != None: 
            for flux in self.fluxline_.collections:
                flux.set_visible(False)
        if self.contour_ != None: 
            for flux in self.contour_.collections:
                flux.set_visible(False)
        
    # ------------------------------------------------------------------------
    def show(self):
        if self.fluxline_ != None: 
            for flux in self.fluxline_.collections:
                flux.set_visible(True)
        if self.contour_ != None: 
            for flux in self.contour_.collections:
                flux.set_visible(True)
        
