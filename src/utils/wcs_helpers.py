from astropy.coordinates import SkyCoord
import astropy.units as u

def get_ra_dec_str(wcs, px, py, precision=6):
    """
    Convert pixel coordinates to RA/DEC formatted strings.
    """
    ra, dec = wcs.pixel_to_world_values(px, py)
    sc = SkyCoord(ra, dec, unit='deg')
    ra_str = sc.ra.to_string(unit=u.hour, sep=':', precision=precision)
    dec_str = sc.dec.to_string(unit=u.deg, sep=':', precision=precision)
    return ra_str, dec_str, sc

def calculate_position_angle(wcs, px1, py1, px2, py2):
    """
    Calculate the position angle between two pixel coordinates.
    """
    ra1, dec1 = wcs.pixel_to_world_values(px1, py1)
    sc1 = SkyCoord(ra1, dec1, unit='deg')
    
    ra2, dec2 = wcs.pixel_to_world_values(px2, py2)
    sc2 = SkyCoord(ra2, dec2, unit='deg')
    
    return sc1.position_angle(sc2).deg

def world_to_pixel(wcs, ra, dec):
    """
    Convert RA/DEC (in degrees) to pixel coordinates.
    """
    return wcs.world_to_pixel_values(ra, dec)

def parse_coord_string(coord_str, is_ra=True):
    """
    Parse a coordinate string (e.g. '12:34:56' or '12.34') into decimal degrees.
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
