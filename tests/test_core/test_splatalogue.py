from cubex.core.splatalogue import format_chemical_formula

def test_format_chemical_formula_html():
    """Verify that HTML tags <sub> and <sup> are converted to unicode equivalents."""
    assert format_chemical_formula("H<sub>2</sub>O") == "H₂O"
    assert format_chemical_formula("<sup>13</sup>CO") == "¹³CO"
    assert format_chemical_formula("CH<sub>3</sub>OH") == "CH₃OH"

def test_format_chemical_formula_bare_numbers():
    """Verify that bare text formats isotopes to superscript and atoms to subscript."""
    assert format_chemical_formula("13CO") == "¹³CO"
    assert format_chemical_formula("CH3OH") == "CH₃OH"
    assert format_chemical_formula("13CH3OH") == "¹³CH₃OH"

def test_format_chemical_formula_with_state():
    """Verify that the quantum state space delimiter protects trailing numbers."""
    assert format_chemical_formula("CH3OH v=0") == "CH₃OH v=0"
    assert format_chemical_formula("13CO v=1") == "¹³CO v=1"

def test_format_chemical_formula_edge_cases():
    """Verify safe fallback on non-string inputs or unexpected formatting."""
    assert format_chemical_formula(None) is None
    assert format_chemical_formula(123) == 123
    assert format_chemical_formula("") == ""
