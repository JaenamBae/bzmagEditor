from matplotlib.transforms import (Affine2D, IdentityTransform, ScaledTranslation)
import bzmagPy as bzmag
import numpy as np

#-----------------------------------------------------------------------------
class CSArtist():
    # ------------------------------------------------------------------------
    # based on Data coordinates
    def __init__(self, parent):
        self.parent_ = parent
        self.artist_ = None
        self.ox_ = 0
        self.oy_ = 0
        
    # ------------------------------------------------------------------------
    def setNode(self, node):
        if self.artist_ != None:
            self.artist_[0].remove()
            self.artist_[1].remove()
            
        if self.parent_ == None: return
        
        [m11, m12, m13, m21, m22, m23] = node.getTransformation()
        #https://matplotlib.org/3.1.0/tutorials/advanced/transforms_tutorial.html
        mtx = np.array([[m11, m12, 0],
                        [m21, m22, 0],
                        [0,   0,   1]])
        refcs = Affine2D(matrix=mtx)
        xx, xy = refcs.transform((0.0, 1.0))
        yx, yy = refcs.transform((1.0, 0.0))
        self.ox_ = m13
        self.oy_ = m23
        
        axes = self.parent_.axes_
        figure = self.parent_.fig_
        trans = (figure.dpi_scale_trans + ScaledTranslation(self.ox_, self.oy_, axes.transData))
        cx = axes.arrow(0, 0, xx, xy, head_width=0.1, head_length=0.2, fc='b', ec='b', transform=trans)
        cy = axes.arrow(0, 0, yx, yy, head_width=0.1, head_length=0.2, fc='r', ec='r', transform=trans)
        self.artist_ = (cx, cy)
        
        
    def setSelected(self, selected):
        if selected == True:
            self.artist_[0].set_linestyle(':')
            self.artist_[1].set_linestyle(':')
        else:
            self.artist_[0].set_linestyle('-')
            self.artist_[1].set_linestyle('-')
        
    # ------------------------------------------------------------------------
    def getArtist(self):
        return self.artist_
        
    # ------------------------------------------------------------------------
    def hide(self):
        if self.artist_ == None: return
        self.artist_[0].set_visible(False)
        self.artist_[1].set_visible(False)
        
    # ------------------------------------------------------------------------
    def show(self):
        if self.artist_ == None: return
        self.artist_[0].set_visible(True)
        self.artist_[1].set_visible(True)
        

