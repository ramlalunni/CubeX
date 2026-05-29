"""
Module for querying and parsing molecular line data from the Splatalogue database.
"""
import re
import astropy.units as u
from PyQt5.QtCore import QThread, pyqtSignal

def format_chemical_formula(formula_str):
    """
    Format raw database chemical formulas and HTML tags into proper Unicode sub/superscripts.

    Parameters
    ----------
    formula_str : str
        The raw chemical formula string retrieved from the database.

    Returns
    -------
    str
        The formatted chemical formula string with Unicode sub/superscripts.

    Raises
    ------
    None

    Notes
    -----
    This function uses regular expressions and string translation maps to 
    convert standard HTML tags (e.g., `<sub>`, `<sup>`) and inline numeric 
    sequences into corresponding Unicode characters for improved GUI rendering.
    """
    if not isinstance(formula_str, str): 
        return formula_str
    
    sub_map = str.maketrans("0123456789+-=()aex", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₓ")
    sup_map = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")

    if '<sub>' in formula_str or '<sup>' in formula_str:
        res = re.sub(r'<sub>(.*?)</sub>', lambda m: m.group(1).translate(sub_map), formula_str)
        res = re.sub(r'<sup>(.*?)</sup>', lambda m: m.group(1).translate(sup_map), res)
        res = re.sub(r'<[^>]+>', '', res) 
        return res

    parts = formula_str.split(' ', 1)
    mol = parts[0]
    state = f" {parts[1]}" if len(parts) > 1 else ""

    mol = re.sub(r'^(\d+)', lambda m: m.group(1).translate(sup_map), mol)
    mol = re.sub(r'(?<=[a-zA-Z])([1-9]\d+)(?=[a-zA-Z])', lambda m: m.group(1).translate(sup_map), mol)
    mol = re.sub(r'(?<=[a-zA-Z])(\d)', lambda m: m.group(1).translate(sub_map), mol)

    return mol + state

class SplatalogueWorker(QThread):
    """
    Background worker thread to query the Splatalogue API without freezing the GUI.

    Attributes
    ----------
    fmin : float
        Minimum frequency for the search in GHz.
    fmax : float
        Maximum frequency for the search in GHz.
    catalogs : tuple or list
        List of catalog names to query.
    v_sys : float
        Systemic velocity in km/s.
    e_max : float
        Maximum upper state energy in Kelvin.
    species : list of str
        List of parsed species patterns to filter by.
    finished : PyQt5.QtCore.pyqtSignal
        Signal emitted when querying is successful. Provides a list of dictionaries.
    error : PyQt5.QtCore.pyqtSignal
        Signal emitted if an error occurs. Provides the error string.

    Notes
    -----
    The Splatalogue query uses `astroquery` and downloads line lists over HTTP.
    This thread performs heavy Pandas dataframe manipulation, filtering by upper
    state energy, and species matching before emitting a deduplicated list.
    """
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, fmin, fmax, catalogs, v_sys, e_max, species):
        """
        Initialize the Splatalogue API worker.

        Parameters
        ----------
        fmin : float
            Minimum frequency in GHz.
        fmax : float
            Maximum frequency in GHz.
        catalogs : list of str
            Catalog names.
        v_sys : float
            Systemic velocity in km/s.
        e_max : float
            Maximum upper state energy in K.
        species : str
            Comma-separated string of species to search for.
        """
        super().__init__()
        self.fmin = fmin
        self.fmax = fmax
        self.catalogs = catalogs
        self.v_sys = v_sys
        self.e_max = e_max
        self.species = [s.strip() for s in species.split(',') if s.strip()]

    def run(self):
        """
        Execute the astroquery Splatalogue search and process results.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Emits an error string via the `error` signal upon encountering an API or parsing issue.
        """
        try:
            # Lazy imports so the app starts faster and errors are caught by the thread
            import pandas as pd
            from astroquery.splatalogue import Splatalogue
            
            res = Splatalogue.query_lines(
                self.fmin * u.GHz, 
                self.fmax * u.GHz, 
                line_lists=self.catalogs,
                show_upper_degeneracy=True,
                export=True 
            )

            if res is None or len(res) == 0:
                self.finished.emit([])
                return

            df = res.to_pandas()

            # Normalize column names depending on what Astroquery returns
            column_mapping = {
                'name': 'formula',
                'chemical_name': 'molecule_name',
                'resolved_QNs': 'QN',
                'orderedfreq': 'restfreq',
                'upper_state_energy_K': 'Eup(K)'
            }
            
            available_cols = [c for c in column_mapping.keys() if c in df.columns]
            output_df = df[available_cols].copy()
            output_df = output_df.rename(columns={k: column_mapping[k] for k in available_cols})

            # Format formulas
            if 'formula' in output_df.columns:
                output_df['formula'] = output_df['formula'].apply(format_chemical_formula)
                
            if 'molecule_name' in output_df.columns:
                output_df['molecule_name'] = output_df['molecule_name'].apply(format_chemical_formula)

            # Clean and convert numeric types
            if 'restfreq' in output_df.columns:
                output_df['restfreq'] = pd.to_numeric(output_df['restfreq'], errors='coerce') / 1000.0
                output_df = output_df.dropna(subset=['restfreq'])
            if 'Eup(K)' in output_df.columns:
                output_df['Eup(K)'] = pd.to_numeric(output_df['Eup(K)'], errors='coerce')

            # Filter by Energy
            if 'Eup(K)' in output_df.columns and self.e_max > 0:
                output_df = output_df[output_df['Eup(K)'] <= self.e_max]

            # Filter by Species string
            if self.species and 'molecule_name' in output_df.columns:
                pattern = '|'.join(self.species)
                output_df = output_df[output_df['molecule_name'].str.contains(pattern, case=False, na=False)]

            # De-duplicate lines that are effectively identical within our resolution
            if 'restfreq' in output_df.columns and 'molecule_name' in output_df.columns:
                output_df['rounded_freq'] = output_df['restfreq'].round(4)
                output_df = output_df.drop_duplicates(subset=['molecule_name', 'rounded_freq'])
                output_df = output_df.drop(columns=['rounded_freq'])
                output_df = output_df.sort_values('restfreq').reset_index(drop=True)

            parsed_results = output_df.to_dict('records')
            self.finished.emit(parsed_results)
                    
        except ImportError as e:
            self.error.emit(f"Missing library: {str(e)}\nTry: pip install pandas astroquery")
        except Exception as e:
            self.error.emit(f"API Error: {str(e)}")