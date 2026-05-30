import pytest

def test_primary_imports_succeed():
    """
    Sanity check to ensure the core components of the application
    can be imported without circular dependency errors.
    """
    try:
        import cubex.core.math_kernels
        import cubex.core.exporters
        import cubex.gui.controllers.main_controller
        import cubex.gui.components.explorer_view
    except ImportError as e:
        pytest.fail(f"ImportError encountered during sanity check: {e}")
        
    assert True
