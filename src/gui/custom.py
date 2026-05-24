import types
import pyqtgraph as pg
import astropy.units as u
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QStyleOptionSlider

# ==============================================================================
# CUSTOM SLIDER
# ==============================================================================
class JumpSlider(QSlider):
    """
    A custom QSlider that allows the user to click anywhere on the track 
    to instantly jump the handle to that exact position.
    """
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            val = self.style().sliderValueFromPosition(
                self.minimum(), 
                self.maximum(), 
                event.pos().x(), 
                self.width(), 
                opt.upsideDown
            )
            self.setValue(val)
        super().mousePressEvent(event)

# ==============================================================================
# AXIS FIX: Force Pyqtgraph into raw Scientific Notation
# ==============================================================================
def fix_axis_scaling(ax):
    """
    Monkey-patches a PyQtGraph AxisItem to prevent it from automatically 
    appending SI prefixes (like 'x 0.001') and instead forces it to use 
    standard scientific notation (e.g., '1.00e-03') for small values.
    """
    ax.enableAutoSIPrefix(False)
    orig_tickValues = ax.tickValues
    
    def new_tickValues(self, minVal, maxVal, size):
        ticks = orig_tickValues(minVal, maxVal, size)
        self.scale = 1.0  # Force scale to 1.0 to permanently stop the label hack
        return ticks
        
    def new_tickStrings(self, values, scale, spacing):
        return [f"{v:.2e}" if (abs(v) < 1e-3 and abs(v) > 0) else f"{v:g}" for v in values]
        
    ax.tickValues = types.MethodType(new_tickValues, ax)
    ax.tickStrings = types.MethodType(new_tickStrings, ax)

# ==============================================================================
# CUSTOM WCS AXIS FOR PYQTGRAPH
# ==============================================================================
class WCSAxisItem(pg.AxisItem):
    """
    A custom AxisItem that automatically translates pixel coordinates 
    into Right Ascension and Declination strings using Astropy's WCS.
    """
    def __init__(self, orientation, **kwargs):
        super().__init__(orientation, **kwargs)
        self.wcs = None
        self.is_absolute = False
        self.nx = 1
        self.ny = 1
        self.pix_scale = 1.0

    def update_wcs(self, wcs, nx, ny, pix_scale, is_absolute):
        self.wcs = wcs
        self.nx = nx
        self.ny = ny
        self.pix_scale = pix_scale
        self.is_absolute = is_absolute
        self.picture = None 
        self.update()

    def tickStrings(self, values, scale, spacing):
        # Fallback to standard tick rendering if WCS isn't active
        if not self.is_absolute or self.wcs is None:
            return super().tickStrings(values, scale, spacing)

        strings = []
        for val in values:
            try:
                if self.orientation in ['bottom', 'top']:
                    pix_x = (self.nx / 2.0) - (val / self.pix_scale)
                    pix_y = self.ny / 2.0
                    coord = self.wcs.pixel_to_world(pix_x, pix_y)
                    s = coord.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
                    strings.append(s)
                else: 
                    pix_y = (self.ny / 2.0) + (val / self.pix_scale)
                    pix_x = self.nx / 2.0
                    coord = self.wcs.pixel_to_world(pix_x, pix_y)
                    s = coord.dec.to_string(unit=u.degree, sep=':', precision=1, alwayssign=True, pad=True)
                    strings.append(s)
            except Exception:
                strings.append("")
        return strings

# ==============================================================================
# PA CONVENTION HELPERS
# ==============================================================================
def get_casa_pa(pyqt_roi):
    # Translates PyQt Cartesian angle (in an invertX ViewBox) to CASA IAU PA.
    raw_pa = 90 - pyqt_roi.angle()
    pa = (raw_pa + 90) % 180 - 90
    return 90.0 if pa <= -90 else pa

def get_pyqt_angle(casa_pa):
    # Translates CASA IAU PA to PyQt Cartesian angle.
    return 90 - casa_pa