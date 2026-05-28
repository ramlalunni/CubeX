from PyQt5.QtCore import Qt
import pyqtgraph as pg

def make_roi_rotatable_with_ctrl(roi):
    original_move_point = roi.movePoint
    def custom_move_point(handle, pos, modifiers=Qt.NoModifier, finish=True, coords='parent'):
        h_dict = next((h for h in roi.handles if h['item'] == handle), None)
        if h_dict:
            if 'orig_center' not in h_dict:
                h_dict['orig_center'] = h_dict['center']
                
            if modifiers & Qt.ControlModifier:
                h_dict['type'] = 'r'
                h_dict['center'] = pg.Point(0.5, 0.5)
                handle.setCursor(Qt.ClosedHandCursor)
            else:
                h_dict['type'] = 's'
                h_dict['center'] = h_dict['orig_center']
                handle.setCursor(Qt.CrossCursor)
        original_move_point(handle, pos, modifiers, finish, coords)
    roi.movePoint = custom_move_point

class ChannelMapViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.drag_start = None
        self.current_roi = None
        self.parent_tab = None

    def mouseDragEvent(self, ev, axis=None):
        if ((ev.modifiers() == Qt.ControlModifier and ev.isStart()) or self.current_roi is not None) and self.parent_tab:
            mode = self.parent_tab.combo_panel_mode.currentText()
            if mode == "Spatial Analysis":
                tool = self.parent_tab.combo_spatial_tool.currentText()
                if tool == "Point":
                    ev.ignore()
                    return
                if ev.isStart():
                    self.drag_start = self.mapSceneToView(ev.buttonDownScenePos())
                    if tool == "Line":
                        self.current_roi = pg.LineSegmentROI([[self.drag_start.x(), self.drag_start.y()], [self.drag_start.x() + 0.1, self.drag_start.y() + 0.1]], pen=pg.mkPen('c', width=2))
                    elif tool == "Rectangle":
                        self.current_roi = pg.RectROI([self.drag_start.x(), self.drag_start.y()], [1e-5, 1e-5], pen=pg.mkPen('c', width=2))
                        self.current_roi.addScaleHandle([0, 0], [1, 1])
                        self.current_roi.addScaleHandle([1, 1], [0, 0])
                        self.current_roi.addScaleHandle([0, 1], [1, 0])
                        self.current_roi.addScaleHandle([1, 0], [0, 1])
                        self.current_roi.addScaleHandle([0.5, 0], [0.5, 1])
                        self.current_roi.addScaleHandle([0.5, 1], [0.5, 0])
                        self.current_roi.addScaleHandle([0, 0.5], [1, 0.5])
                        self.current_roi.addScaleHandle([1, 0.5], [0, 0.5])
                        make_roi_rotatable_with_ctrl(self.current_roi)
                    elif tool == "Ellipse":
                        self.current_roi = pg.EllipseROI([self.drag_start.x(), self.drag_start.y()], [1e-5, 1e-5], pen=pg.mkPen('c', width=2))
                        self.current_roi.addScaleHandle([0, 0], [1, 1])
                        self.current_roi.addScaleHandle([1, 1], [0, 0])
                        self.current_roi.addScaleHandle([0, 1], [1, 0])
                        self.current_roi.addScaleHandle([1, 0], [0, 1])
                        self.current_roi.addScaleHandle([0.5, 0], [0.5, 1])
                        self.current_roi.addScaleHandle([0.5, 1], [0.5, 0])
                        self.current_roi.addScaleHandle([0, 0.5], [1, 0.5])
                        self.current_roi.addScaleHandle([1, 0.5], [0, 0.5])
                        make_roi_rotatable_with_ctrl(self.current_roi)
                    
                    if self.current_roi:
                        self.addItem(self.current_roi)
                        ev.accept()
                elif ev.isFinish():
                    if self.current_roi:
                        self.parent_tab.add_spatial_region(self.current_roi, tool)
                        self.current_roi = None
                    ev.accept()
                else:
                    if self.current_roi:
                        current_pos = self.mapSceneToView(ev.scenePos())
                        if tool == "Line":
                            handles = self.current_roi.getHandles()
                            if len(handles) > 1:
                                self.current_roi.movePoint(handles[1], current_pos)
                        else:
                            w = current_pos.x() - self.drag_start.x()
                            h = current_pos.y() - self.drag_start.y()
                            self.current_roi.setSize([w, h])
                    ev.accept()
            elif self.parent_tab.is_pv_drawing_mode():
                if ev.isStart():
                    self.drag_start = self.mapSceneToView(ev.buttonDownScenePos())
                    self.current_roi = pg.LineSegmentROI(
                        [
                            [self.drag_start.x(), self.drag_start.y()],
                            [self.drag_start.x() + 0.1, self.drag_start.y() + 0.1],
                        ],
                        pen=pg.mkPen('m', width=2),
                    )
                    self.addItem(self.current_roi)
                    ev.accept()
                elif ev.isFinish():
                    if self.current_roi:
                        self.parent_tab.add_pv_cut(self.current_roi)
                        self.current_roi = None
                    ev.accept()
                else:
                    if self.current_roi:
                        current_pos = self.mapSceneToView(ev.scenePos())
                        handles = self.current_roi.getHandles()
                        if len(handles) > 1:
                            self.current_roi.movePoint(handles[1], current_pos)
                    ev.accept()
            elif mode == "Spectrum":
                tool = self.parent_tab.combo_roi.currentText()
                if tool in ["Whole Map", "Point (Beam)", "Custom Polygon"]:
                    ev.ignore()
                    return
                
                if ev.isStart():
                    self.drag_start = self.mapSceneToView(ev.buttonDownScenePos())
                    if tool == "Rectangle":
                        self.current_roi = pg.RectROI([self.drag_start.x(), self.drag_start.y()], [1e-5, 1e-5], pen=pg.mkPen('#f1c40f'))
                        self.current_roi.addScaleHandle([0, 0], [1, 1])
                        self.current_roi.addScaleHandle([1, 1], [0, 0])
                        self.current_roi.addScaleHandle([0, 1], [1, 0])
                        self.current_roi.addScaleHandle([1, 0], [0, 1])
                        self.current_roi.addScaleHandle([0.5, 0], [0.5, 1])
                        self.current_roi.addScaleHandle([0.5, 1], [0.5, 0])
                        self.current_roi.addScaleHandle([0, 0.5], [1, 0.5])
                        self.current_roi.addScaleHandle([1, 0.5], [0, 0.5])
                        make_roi_rotatable_with_ctrl(self.current_roi)
                    elif tool == "Ellipse":
                        self.current_roi = pg.EllipseROI([self.drag_start.x(), self.drag_start.y()], [1e-5, 1e-5], pen=pg.mkPen('#f1c40f'))
                        self.current_roi.addScaleHandle([0, 0], [1, 1])
                        self.current_roi.addScaleHandle([1, 1], [0, 0])
                        self.current_roi.addScaleHandle([0, 1], [1, 0])
                        self.current_roi.addScaleHandle([1, 0], [0, 1])
                        self.current_roi.addScaleHandle([0.5, 0], [0.5, 1])
                        self.current_roi.addScaleHandle([0.5, 1], [0.5, 0])
                        self.current_roi.addScaleHandle([0, 0.5], [1, 0.5])
                        self.current_roi.addScaleHandle([1, 0.5], [0, 0.5])
                        make_roi_rotatable_with_ctrl(self.current_roi)
                    
                    if self.current_roi:
                        self.addItem(self.current_roi)
                        ev.accept()
                elif ev.isFinish():
                    if self.current_roi:
                        self.parent_tab._finish_roi_addition(self.current_roi, tool)
                        self.current_roi = None
                    ev.accept()
                else:
                    if self.current_roi:
                        current_pos = self.mapSceneToView(ev.scenePos())
                        w = current_pos.x() - self.drag_start.x()
                        h = current_pos.y() - self.drag_start.y()
                        self.current_roi.setSize([w, h])
                    ev.accept()
            else:
                super().mouseDragEvent(ev, axis)
        else:
            super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):
        if ev.modifiers() == Qt.ControlModifier and self.parent_tab:
            mode = self.parent_tab.combo_panel_mode.currentText()
            if mode == "Spatial Analysis":
                tool = self.parent_tab.combo_spatial_tool.currentText()
                pos = self.mapSceneToView(ev.scenePos())
                
                hit = False
                for item in self.parent_tab.spatial_rois:
                    roi = item["roi"]
                    if hasattr(roi, 'shape'):
                        if roi.shape().contains(roi.mapFromScene(ev.scenePos())):
                            self.parent_tab.select_spatial_region(roi)
                            hit = True
                            
                if not hit:
                    for item in self.parent_tab.spatial_rois:
                        roi = item["roi"]
                        if isinstance(roi, pg.LineSegmentROI) and self.parent_tab.line_roi_hit_test(roi, ev.scenePos()):
                            self.parent_tab.select_spatial_region(roi)
                            hit = True
                            break
                if hit:
                    ev.accept()
                    return
                
                if tool == "Point":
                    # Use a small ROI for point since PointROI doesn't exist
                    sz = self.parent_tab.pix_scale_arcsec * 0.1 if hasattr(self.parent_tab, 'pix_scale_arcsec') else 0.1
                    roi = pg.ROI([pos.x() - sz/2, pos.y() - sz/2], [sz, sz], pen=pg.mkPen('c', width=2))
                    self.addItem(roi)
                    self.parent_tab.add_spatial_region(roi, "Point")
                    ev.accept()
                    return
            elif self.parent_tab.is_pv_drawing_mode():
                for item in self.parent_tab.pv_cuts:
                    if self.parent_tab.line_roi_hit_test(item["roi"], ev.scenePos()):
                        self.parent_tab.select_pv_cut(item["roi"])
                        ev.accept()
                        return
            elif mode == "Spectrum":
                tool = self.parent_tab.combo_roi.currentText()
                if tool == "Point (Beam)":
                    pos = self.mapSceneToView(ev.scenePos())
                    self.parent_tab.change_roi(tool, cx=pos.x(), cy=pos.y())
                    ev.accept()
                    return
                
        super().mouseClickEvent(ev)

class SpectrumViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.drag_start = None
        self.current_roi = None
        self.parent_tab = None
        self.dragging_roi = None
        self.dragging_roi_initial_x = None

    def mouseDragEvent(self, ev, axis=None):
        if (ev.modifiers() == Qt.ControlModifier and ev.isStart()) or self.current_roi is not None or getattr(self, 'dragging_roi', None) is not None:
            if ev.isStart():
                mp = self.mapSceneToView(ev.buttonDownScenePos())
                self.drag_start = mp
                
                clicked_roi = None
                if self.parent_tab:
                    active_rois = self.parent_tab.get_active_spectrum_rois()
                    for item in active_rois:
                        roi = item["roi"]
                        if hasattr(roi, 'getData'):
                            x_data, _ = roi.getData()
                            if x_data is not None and len(x_data) >= 2:
                                min_x, max_x = min(x_data), max(x_data)
                                if min_x <= mp.x() <= max_x:
                                    clicked_roi = roi
                                    break
                
                if clicked_roi:
                    self.dragging_roi = clicked_roi
                    x_data, _ = clicked_roi.getData()
                    self.dragging_roi_initial_x = list(x_data)
                else:
                    self.current_roi = pg.PlotDataItem(pen=pg.mkPen(color='c', width=4))
                    self.addItem(self.current_roi)
                ev.accept()
            elif ev.isFinish():
                if getattr(self, 'dragging_roi', None) is not None:
                    if self.parent_tab:
                        self.parent_tab.update_spectrum_region_calc()
                        for item in self.parent_tab.get_active_spectrum_rois():
                            if item["roi"] == self.dragging_roi and "update_text_pos" in item:
                                item["update_text_pos"]()
                    self.dragging_roi = None
                elif self.parent_tab and self.current_roi:
                    self.parent_tab.add_spectrum_region(self.current_roi)
                    self.current_roi = None
                ev.accept()
            else:
                current_pos = self.mapSceneToView(ev.scenePos())
                dx = current_pos.x() - self.drag_start.x()
                
                if getattr(self, 'dragging_roi', None) is not None:
                    new_x_data = [x + dx for x in self.dragging_roi_initial_x]
                    self.dragging_roi.setData(new_x_data, [0, 0])
                    if self.parent_tab:
                        for item in self.parent_tab.get_active_spectrum_rois():
                            if item["roi"] == self.dragging_roi and "update_text_pos" in item:
                                item["update_text_pos"]()
                elif self.current_roi:
                    self.current_roi.setData([self.drag_start.x(), current_pos.x()], [0, 0])
                ev.accept()
        else:
            super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):
        super().mouseClickEvent(ev)
