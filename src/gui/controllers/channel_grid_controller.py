import os
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QCheckBox

class ChannelGridController:
    """
    Controller for the Channel Grid window.
    Handles generating the grid of images, updating histograms, and exporting to PDF.
    """
    def __init__(self, view, explorer_tab):
        self.view = view
        self.tab = explorer_tab
        
        self.images = []
        self.view_boxes = []
        self.pos_tup = None
        self.scale_tup = None
        
        # Connect to view signals
        self.view.cmap_changed.connect(self.change_cmap)
        self.view.reset_zoom_clicked.connect(self.reset_zoom)
        self.view.export_pdf_clicked.connect(self.export_to_pdf)
        
        # Connect UI element signals
        self.view.grid_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.view.hist.sigLevelsChanged.connect(self.on_hist_levels_changed)
        self.view.hist.sigLookupTableChanged.connect(self.on_hist_lut_changed)
        
        self.update_grid()
        
    def update_grid(self):
        # Clear existing grid
        self.view.grid_widget.clear()
        self.images.clear()
        self.view_boxes.clear()
        
        cube, v_axis, minX, maxX = self.tab.controller.get_velocity_subset(use_full_range=False) if hasattr(self.tab, 'controller') else self.tab.get_velocity_subset(use_full_range=False)
        if cube is None or len(v_axis) == 0:
            self.view.grid_widget.addLabel("No channels in selected range.", col=0, row=0)
            return
            
        n_channels = len(v_axis)
        cols = int(np.ceil(np.sqrt(n_channels)))
        rows = int(np.ceil(n_channels / cols))
        
        # Set a fixed size. Without axes, grid size matches data aspect ratio exactly. Add 2px for margin.
        base_w = 200
        base_h = int(200 * (self.tab.ny / self.tab.nx)) if self.tab.nx > 0 else 200
        self.view.grid_widget.setFixedSize(int(cols * base_w) + 2, int(rows * base_h) + 2)
        
        # Get current histogram state from main channel map
        main_hist = self.tab.view_channel.ui.histogram
        levels = main_hist.getLevels()
        
        # Update dialog's histogram gradient to match main one
        self.view.hist.gradient.restoreState(main_hist.gradient.saveState())
        lut = self.view.hist.gradient.getLookupTable(256)
        
        # Adapt histogram range to be reasonable (0 to max positive data)
        cube_max = float(np.nanmax(cube)) if np.nanmax(cube) > 0 else 1.0
        new_levels = [0, cube_max]
        
        pos_tup = ((self.tab.nx / 2) * self.tab.pix_scale_arcsec, -(self.tab.ny / 2) * self.tab.pix_scale_arcsec)
        scale_tup = (-self.tab.pix_scale_arcsec, self.tab.pix_scale_arcsec)
        
        self.pos_tup = pos_tup
        self.scale_tup = scale_tup
        
        self.view.grid_widget.ci.layout.setSpacing(0)
        self.view.grid_widget.ci.layout.setContentsMargins(1, 1, 1, 1)
        
        first_plot = None
        for idx in range(n_channels):
            r, c = divmod(idx, cols)
            
            p = self.view.grid_widget.addPlot(row=r, col=c)
            p.setAspectLocked(True)
            p.invertY(False)
            p.invertX(True)
            p.hideButtons()
            p.layout.setContentsMargins(0, 0, 0, 0)
            
            vb = p.getViewBox()
            vb.setDefaultPadding(0.0)
            vb.setBorder(pg.mkPen(color='w', width=1))
            
            if first_plot is None:
                first_plot = p
            else:
                p.setXLink(first_plot)
                p.setYLink(first_plot)
                
            # Hide all axes
            p.hideAxis('left')
            p.hideAxis('bottom')
            p.hideAxis('right')
            p.hideAxis('top')
            
            img = pg.ImageItem()
            p.addItem(img)
            
            # Set data
            img.setImage(cube[idx, :, :], scale=scale_tup, pos=pos_tup)
            img.setLevels(new_levels)
            img.setLookupTable(lut)
            
            # Add velocity text at top left corner of the ViewBox (screen coordinates)
            vel_text = pg.TextItem(f"{v_axis[idx]:.2f} km/s", color='w', anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 150))
            vel_text.setParentItem(p.getViewBox())
            vel_text.setPos(5, 5)
            vel_text.setZValue(100)
            
            self.images.append(img)
            self.view_boxes.append({'vb': vb, 'img': img, 'vel': v_axis[idx], 'data': cube[idx, :, :]})
            
        # Link the histogram to the first image to make it active
        if self.images:
            self.view.hist.setImageItem(self.images[0])
            # Set the levels and visible bounds again because setImageItem overrides them
            self.view.hist.setLevels(new_levels[0], new_levels[1])
            self.view.hist.setHistogramRange(0, cube_max)
            
            # Sync the colormap with the dropdown selection (which includes fallbacks for non-native ones like cubehelix)
            self.change_cmap(self.view.combo_cmap.currentText())
            
            # Delay the autoRange reset to ensure layout has fully updated after adding/removing tiles
            pg.QtCore.QTimer.singleShot(100, self.reset_zoom)
                
    def change_cmap(self, cmap_name):
        cmap_name = cmap_name.lower()
        try:
            self.view.hist.gradient.loadPreset(cmap_name)
        except KeyError:
            cmap = plt.get_cmap(cmap_name)
            pos = np.linspace(0.0, 1.0, 64)
            colors = cmap(pos) * 255
            self.view.hist.gradient.setColorMap(pg.ColorMap(pos, colors.astype(np.ubyte)))
            
        self.on_hist_lut_changed()

    def reset_zoom(self):
        if self.view_boxes:
            vb = self.view_boxes[0]['vb']
            vb.autoRange(padding=0)
            
    def on_mouse_moved(self, pos):
        if not self.view_boxes:
            return
            
        for item in self.view_boxes:
            vb = item['vb']
            if vb.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)

                img = item['img']
                local_pos = img.mapFromView(mouse_point)
                px = int(local_pos.x())
                py = int(local_pos.y())
                
                data = item['data']
                if 0 <= px < data.shape[0] and 0 <= py < data.shape[1]:
                    val = data[px, py]
                    # Calculate offsets manually from the local image pixel coordinate to guarantee
                    # perfectly zeroed centers regardless of ViewBox projection linking
                    ra_offset = (data.shape[0] / 2.0 - local_pos.x()) * self.tab.pix_scale_arcsec
                    dec_offset = (local_pos.y() - data.shape[1] / 2.0) * self.tab.pix_scale_arcsec
                    
                    self.view.lbl_hover.setText(f"{item['vel']:.2f} km/s  |  ({px}, {py})  |  RA: {ra_offset:.2f}\"  |  DEC: {dec_offset:.2f}\"  |  {val:.4e} {self.tab.display_unit}")
                    self.view.lbl_hover.setStyleSheet("font-family: monospace; font-size: 11.5px; color: #3498db; font-weight: bold; padding: 5px;")
                else:
                    self.view.lbl_hover.setText(f"{item['vel']:.2f} km/s  |  RA: --  |  DEC: --  |  --")
                    self.view.lbl_hover.setStyleSheet("font-family: monospace; font-size: 11.5px; color: #aaa; padding: 5px;")
                return
                
        # If we reach here, we are not hovering over any valid tile
        self.view.lbl_hover.setText("Hover over a tile to see details")
        self.view.lbl_hover.setStyleSheet("font-family: monospace; font-size: 11.5px; color: #aaa; padding: 5px;")
                
    def on_hist_levels_changed(self):
        levels = self.view.hist.getLevels()
        for img in self.images:
            img.setLevels(levels)
            
    def on_hist_lut_changed(self):
        lut = self.view.hist.gradient.getLookupTable(256)
        for img in self.images:
            img.setLookupTable(lut)
            
    def update_from_main_hist(self):
        main_hist = self.tab.view_channel.ui.histogram
        levels = main_hist.getLevels()
        self.view.hist.setLevels(levels[0], levels[1])
            
    def update_from_main_lut(self):
        main_hist = self.tab.view_channel.ui.histogram
        self.view.hist.gradient.restoreState(main_hist.gradient.saveState())
        
    def export_to_pdf(self):
        parent_filename = "cube"
        if getattr(self.tab, 'current_file_name', None):
            parent_filename = os.path.basename(self.tab.current_file_name)
        base_filename = os.path.splitext(parent_filename)[0]
        default_filename = f"{base_filename}_channel_map_grid.pdf"
        
        dialog = QFileDialog(self.view, "Save PDF", default_filename, "PDF Files (*.pdf)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("pdf")
        
        layout = dialog.layout()
        chk_title = QCheckBox("Include Plot Title in Export")
        chk_title.setChecked(False)
        if layout:
            try:
                layout.addWidget(chk_title, layout.rowCount(), 0, 1, layout.columnCount())
            except Exception:
                layout.addWidget(chk_title)
                
        if dialog.exec_() != QFileDialog.Accepted:
            return
            
        files = dialog.selectedFiles()
        if not files:
            return
            
        filename = files[0]
        include_title = chk_title.isChecked()
        
        if filename:
            try:
                cube, v_axis, minX, maxX = self.tab.controller.get_velocity_subset(use_full_range=False) if hasattr(self.tab, 'controller') else self.tab.get_velocity_subset(use_full_range=False)
                if cube is None or len(v_axis) == 0:
                    return
                    
                n_channels = len(v_axis)
                cols = int(np.ceil(np.sqrt(n_channels)))
                rows = int(np.ceil(n_channels / cols))
                
                fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), squeeze=False)
                
                if include_title:
                    fig.suptitle(f"{base_filename}_channel_map_grid", fontsize=16)
                    
                levels = self.view.hist.getLevels()
                # Use absolute WCS or not doesn't affect cmap name here, but we can get it from app
                cmap_name = self.tab.parent_window.current_cmap if hasattr(self.tab, 'parent_window') else 'turbo'
                
                extent = [self.tab.nx/2 * self.tab.pix_scale_arcsec, -self.tab.nx/2 * self.tab.pix_scale_arcsec, 
                          -self.tab.ny/2 * self.tab.pix_scale_arcsec, self.tab.ny/2 * self.tab.pix_scale_arcsec]
                          
                for idx in range(rows * cols):
                    r = idx // cols
                    c = idx % cols
                    ax = axes[r, c]
                    
                    if idx < n_channels:
                        plot_data = cube[idx, :, :].T
                        
                        ax.imshow(plot_data, origin='lower', cmap=cmap_name, 
                                       vmin=levels[0], vmax=levels[1], extent=extent)
                                       
                        ax.text(0.05, 0.95, f"{v_axis[idx]:.2f} km/s", transform=ax.transAxes,
                                color='white', verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, pad=1))
                                
                        if c == 0:
                            ax.set_ylabel('DEC offset (arcsec)')
                        else:
                            ax.set_yticklabels([])
                            
                        if r == rows - 1 or idx + cols >= n_channels:
                            ax.set_xlabel('RA offset (arcsec)')
                        else:
                            ax.set_xticklabels([])
                    else:
                        ax.axis('off')
                        
                plt.tight_layout()
                if include_title:
                    plt.subplots_adjust(top=0.95)
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception:
                pass
