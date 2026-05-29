"""
Module providing astrometric utility functions.

Wraps `astropy.wcs` functionality to abstract astronomical coordinate conversions 
away from GUI components.
"""
from astropy.coordinates import SkyCoord
import astropy.units as u

def get_ra_dec_str(wcs, px, py, precision=6):
    """
    Convert pixel coordinates to formatted RA/DEC string representations.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object defining the coordinate system.
    px : float
        X pixel coordinate (0-indexed).
    py : float
        Y pixel coordinate (0-indexed).
    precision : int, optional
        Number of decimal places for seconds, by default 6.

    Returns
    -------
    tuple
        A 3-tuple containing:
        - ra_str (str): Right Ascension formatted as HH:MM:SS.ss
        - dec_str (str): Declination formatted as DD:MM:SS.ss
        - sc (astropy.coordinates.SkyCoord): The underlying SkyCoord object.

    Notes
    -----
    Assumes standard FITS pixel convention where pixel centers are at integer 
    coordinates. Relies on Astropy's `SkyCoord` for accurate formatting.
    """
    ra, dec = wcs.pixel_to_world_values(px, py)
    sc = SkyCoord(ra, dec, unit='deg')
    ra_str = sc.ra.to_string(unit=u.hour, sep=':', precision=precision)
    dec_str = sc.dec.to_string(unit=u.deg, sep=':', precision=precision)
    return ra_str, dec_str, sc

def calculate_position_angle(wcs, px1, py1, px2, py2):
    """
    Calculate the astronomical position angle between two pixel locations.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object defining the coordinate system.
    px1 : float
        X pixel coordinate of the starting point.
    py1 : float
        Y pixel coordinate of the starting point.
    px2 : float
        X pixel coordinate of the ending point.
    py2 : float
        Y pixel coordinate of the ending point.

    Returns
    -------
    float
        The position angle in degrees.

    Notes
    -----
    Calculates the angle East of North from point 1 to point 2 on the celestial 
    sphere using great-circle distance math embedded within `SkyCoord`.
    """
    ra1, dec1 = wcs.pixel_to_world_values(px1, py1)
    sc1 = SkyCoord(ra1, dec1, unit='deg')
    
    ra2, dec2 = wcs.pixel_to_world_values(px2, py2)
    sc2 = SkyCoord(ra2, dec2, unit='deg')
    
    return sc1.position_angle(sc2).deg

def world_to_pixel(wcs, ra, dec):
    """
    Convert decimal RA/DEC (in degrees) back to pixel coordinates.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object defining the coordinate system.
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.

    Returns
    -------
    tuple
        A 2-tuple of (px, py) float pixel coordinates.

    Notes
    -----
    Assumes standard FITS projection equations (e.g. SIN, TAN) defined in the WCS header.
    """
    return wcs.world_to_pixel_values(ra, dec)

def parse_coord_string(coord_str, is_ra=True):
    """
    Parse an astronomical sexagesimal coordinate string into decimal degrees.

    Parameters
    ----------
    coord_str : str
        The input string (e.g. '12h34m56s', '12:34:56', or '12.34').
    is_ra : bool, optional
        Flag indicating if the string is Right Ascension (parsed as hours instead 
        of degrees if no explicit unit is given), by default True.

    Returns
    -------
    float
        The parsed coordinate in decimal degrees.

    Notes
    -----
    Relies on Astropy's `Angle` parser. If the string is successfully cast to a 
    pure float, it assumes the input was already in decimal degrees.
    """
    try:
        return float(coord_str)
    except ValueError:
        if is_ra:
            from astropy.coordinates import Angle
            return Angle(coord_str, unit=u.hour).degree
        else:
            from astropy.coordinates import Angle
            return Angle(coord_str, unit=u.deg).degree
