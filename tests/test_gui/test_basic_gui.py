from cubex.gui.components.channel_grid_view import ChannelGridView

def test_channel_grid_view_initialization(qtbot):
    """
    Smoke test to ensure that a core GUI component can be cleanly 
    instantiated and its layout configured without crashing.
    """
    # Instantiate an isolated view component
    view = ChannelGridView()
    
    # Register the widget with qtbot so it gets properly cleaned up after the test
    qtbot.addWidget(view)
    
    # Basic sanity assertions
    assert view is not None
    assert view.windowTitle() == "Channel Grid"
    
    # Verify that inner UI elements were constructed
    assert view.btn_export is not None
    assert view.btn_export.text() == "Export to PDF"
    assert view.combo_cmap.count() > 0  # Should be populated with colormap options
