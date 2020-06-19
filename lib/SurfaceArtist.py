import matplotlib.patches as patches
from matplotlib.path import Path
import bzmagPy as bzmag

#-----------------------------------------------------------------------------
class SurfaceArtist():
    # ------------------------------------------------------------------------
    # based on Data coordinates
    def __init__(self, parent):
        self.parent_ = parent
        self.artist_ = None
        
    # ------------------------------------------------------------------------
    def setNode(self, node):
        if self.artist_ != None:
            self.artist_.remove()
            
        path = node.getPath(0)
        x = path[0:len(path):3]
        y = path[1:len(path):3]
        
        verts = list(zip(x, y))
        #print(verts)
        
        codes = path[2:len(path):3]
        color = [x/255 for x in node.Color]
        
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, edgecolor='black', antialiased=False, lw=0.1)
        
        if node.isCovered() == 0 : patch.set_fill(False)
        patch.set_visible(~node.IsHide)
        
        axes = self.parent_.axes_
        self.artist_ = axes.add_patch(patch)
        
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        xc_min, xc_max = axes.get_xlim()
        yc_min, yc_max = axes.get_ylim()
        
        if x_min < xc_min : xc_min = x_min
        if x_max > xc_max : xc_max = x_max
        if y_min < yc_min : yc_min = y_min
        if y_max > yc_max : yc_max = y_max
        
        axes.set_xlim([xc_min, xc_max])
        axes.set_ylim([yc_min, yc_max])
    
    def setSelected(self, selected):
        if selected == True:
            self.artist_.set_linestyle('dashed')
            self.artist_.set_edgecolor('blue')
        else:
            self.artist_.set_linestyle('solid')
            self.artist_.set_edgecolor('black')
    
    # ------------------------------------------------------------------------
    def getArtist(self):
        return self.artist_
        
    # ------------------------------------------------------------------------
    def hide(self):
        self.artist_.set_visible(False)
        
    # ------------------------------------------------------------------------
    def show(self):
        self.artist_.set_visible(True)
        
    # ------------------------------------------------------------------------
    # based on Data coordinates
    def boundingRect(self):
        return [self.minX_, self.maxX_, self.minY_, self.maxY_]
