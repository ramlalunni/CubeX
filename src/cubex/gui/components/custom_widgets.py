"""
Module providing custom PyQt and PyQtGraph UI widgets.

Contains specialized sliders, monkey-patched axis scalers, and WCS-aware 
coordinate axes for the frontend view layer.
"""
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
    Custom QSlider that snaps instantly to the mouse click position.

    Attributes
    ----------
    No additional attributes beyond standard QSlider.
    """
    def mousePressEvent(self, event):
        """
        Handle mouse press events to jump the slider position.

        Parameters
        ----------
        event : PyQt5.QtGui.QMouseEvent
            The mouse event containing click coordinates.

        Returns
        -------
        None
        """
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
    Monkey-patch a PyQtGraph AxisItem for scientific notation.

    Parameters
    ----------
    ax : pyqtgraph.AxisItem
        The axis item to be modified in-place.

    Returns
    -------
    None

    Notes
    -----
    PyQtGraph automatically appends SI prefixes (e.g., 'm', 'u') or adds a 
    scale multiplier suffix (like 'x 0.001'). This function overrides the 
    internal `tickValues` and `tickStrings` methods to enforce standard 
    Python scientific string formatting (e.g., '1.00e-03') for small numbers.
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
    Custom AxisItem that translates pixel coordinates to WCS sky coordinates.

    Attributes
    ----------
    wcs : astropy.wcs.WCS or None
        The Astropy WCS object used for coordinate transformations.
    is_absolute : bool
        If True, displays absolute RA/Dec coordinates. If False, displays relative offsets.
    nx : int
        The number of pixels along the x-axis of the data cube.
    ny : int
        The number of pixels along the y-axis of the data cube.
    pix_scale : float
        The scale factor for pixel-to-arcsecond conversions.

    Notes
    -----
    This widget intercepts the PyQtGraph tick drawing cycle to perform Astropy 
    WCS spatial transformations (`pixel_to_world`). It assumes the origin is 
    at the center of the image.
    """
    def __init__(self, orientation, **kwargs):
        super().__init__(orientation, **kwargs)
        self.wcs = None
        self.is_absolute = False
        self.nx = 1
        self.ny = 1
        self.pix_scale = 1.0

    def update_wcs(self, wcs, nx, ny, pix_scale, is_absolute):
        """
        Update the internal WCS parameters and trigger a redraw.

        Parameters
        ----------
        wcs : astropy.wcs.WCS
            The Astropy WCS object.
        nx : int
            X-axis pixel dimension.
        ny : int
            Y-axis pixel dimension.
        pix_scale : float
            Pixel scale.
        is_absolute : bool
            Whether to display absolute coordinates.

        Returns
        -------
        None
        """
        self.wcs = wcs
        self.nx = nx
        self.ny = ny
        self.pix_scale = pix_scale
        self.is_absolute = is_absolute
        self.picture = None 
        self.update()

    def tickStrings(self, values, scale, spacing):
        """
        Generate formatted tick labels for the axis.

        Parameters
        ----------
        values : list of float
            The tick positions in data coordinates.
        scale : float
            The scale factor applied to the values.
        spacing : float
            The spacing between ticks.

        Returns
        -------
        list of str
            A list of formatted coordinate strings.
        """
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
    """
    Translate a PyQt Cartesian angle to the CASA IAU Position Angle (PA).

    Parameters
    ----------
    pyqt_roi : pyqtgraph.ROI
        The Region of Interest (ROI) object containing the current angle.

    Returns
    -------
    float
        The CASA IAU Position Angle in degrees.

    Notes
    -----
    PyQtGraph measures angles counter-clockwise from the x-axis. Since astronomy 
    images in this tool use an inverted x-axis (East to the left), and IAU PA 
    is measured East of North, this performs the requisite geometric transformation 
    and wraps the result to the [-90, 90] degree domain.
    """
    raw_pa = 90 - pyqt_roi.angle()
    pa = (raw_pa + 90) % 180 - 90
    return 90.0 if pa <= -90 else pa

def get_pyqt_angle(casa_pa):
    """
    Translate a CASA IAU Position Angle to a PyQt Cartesian angle.

    Parameters
    ----------
    casa_pa : float
        The CASA IAU Position Angle in degrees.

    Returns
    -------
    float
        The corresponding PyQt Cartesian angle in degrees.
    """
    return 90 - casa_pa