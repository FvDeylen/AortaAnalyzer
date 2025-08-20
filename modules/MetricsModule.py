
import os
import re

import numpy as np 
import pandas as pd 
import vtk 
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator,QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget
)

from defaults import *

class MetricsModule(QWidget):
    metrics_changed = pyqtSignal()
    new_metrics = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # state
        self.patient_dict = None 
        self.centerline = None  # preprocessed centerline
        self.max_diameter_id = None 
        self.current_diameter_id = [0,20]
        self.current_branch_id = 0
        self.patient_height = None
        self.ahi_score = None
        self.wheel_active = False
        self.current_line = None
        self.current_filter = None
        self.current_marker_id = None
        self.move_marker = []  # workaround for mousmove observer ?!
        self.new_volume = True
        self.current_volume = None
        self.current_surface = None

        # vtk/Qt objects for visualization
        self.landmark_actors = {}
        self.landmark_ids = {}
        self.empty_table_spots = []
        self.volume_marker_line = []
        self.volume_marker_filter = []
        self.volume_marker_actors = []
        self.volume_marker_ids = []

        # QT UI  
        self.button_max_diameter = QPushButton("Compute Maximum Diameter")
        self.button_max_diameter.setEnabled(False)
        self.button_max_diameter.setCheckable(True)
        self.max_diameter_label = QLabel("Maximum Diameter: ")
        self.max_diameter_label.setStyleSheet("background-color:white;")
        
        self.button_show_diameter = QPushButton("Show Diameter Along Surface")
        self.button_show_diameter.setEnabled(False)
        self.button_show_diameter.setCheckable(True)
    
        self.button_set_landmarks = QPushButton("Set Landmark: ")
        self.button_set_landmarks.setEnabled(False)
        self.dropdown_landmarks = QComboBox()
        self.dropdown_landmarks.addItems(["Sinotubular junction", "Mid-ascending aorta", "Distal ascending aorta","Aortic arch", "Proximal descending aorta", "Mid-descending aorta", "Other ..."])    
        self.dropdown_landmarks.setEnabled(False)
        self.__setupLandmarkTable()
        
        self.button_remove_landmarks = QPushButton("Remove Landmarks")
        self.button_remove_landmarks.setEnabled(False)
        self.button_remove_landmarks.setCheckable(True)

        self.button_volume_computation = QPushButton("Measure Volume && Surface")
        self.button_volume_computation.setEnabled(False)
        self.button_volume_computation.setCheckable(True)
        self.button_confirm_bounds = QPushButton("Confirm Bounds")
        self.button_confirm_bounds.setEnabled(False)
        self.button_add_markers = QPushButton("Add Cut Marker")
        self.button_add_markers.setVisible(False)
        self.button_add_markers.setCheckable(True)
        
        self.label_opacity = {}
        self.current_diameter_label = QLabel("Current Diameter: ")
        self.current_diameter_label.setStyleSheet("background-color:white")
        opacity = QGraphicsOpacityEffect()
        opacity.setOpacity(0.6)
        self.label_opacity["diameter"] = opacity
        self.current_diameter_label.setGraphicsEffect(opacity)
        self.volume_label = QLabel("Volume: ")  
        self.volume_label.setStyleSheet("background-color:white")
        opacity = QGraphicsOpacityEffect()
        opacity.setOpacity(0.6)
        self.label_opacity["volume"] = opacity
        self.volume_label.setGraphicsEffect(opacity)
        self.surface_label = QLabel("Surface: ")
        opacity = QGraphicsOpacityEffect()
        opacity.setOpacity(0.6)
        self.label_opacity["surface"] = opacity
        self.surface_label.setGraphicsEffect(opacity)
        self.surface_label.setStyleSheet("background-color:white")
        self.height_label = QLabel("Patient Height (m)")
        self.line_edit_height = QLineEdit()
        self.line_edit_height.setValidator(QDoubleValidator(0.5,2.2,2))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.line_edit_height.setSizePolicy(sizePolicy)
        self.line_edit_height.setEnabled(False)
        self.ahi_label = QLabel("AHI (diameter/height):        ")
        self.ahi_label.setStyleSheet("background-color:white")
        self.ahi_label.setToolTip("Aortic Height Index: aortic diameter/patient height")
        
        # connect signals/slots
        self.button_max_diameter.clicked.connect(self.showMaxDiameterMarker)
        self.button_show_diameter.clicked.connect(self.diameterMode)
        self.button_set_landmarks.clicked.connect(self.setLandmark)
        self.button_remove_landmarks.clicked.connect(self.removeLandmarkMode)
        self.button_volume_computation.clicked.connect(self.volumeMode)
        self.button_confirm_bounds.clicked.connect(self.confirmBoundsVolume)
        self.button_add_markers.clicked.connect(self.addMarkerMode)
        self.landmark_table.itemClicked.connect(self.clickTableLandmark)
        self.landmark_table.itemDoubleClicked.connect(self.deleteCustomLandmark)
        self.line_edit_height.editingFinished.connect(self.setHeight)
        
        # VTK UI
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.surface_view = QVTKRenderWindowInteractor(self)
        self.surface_view.SetInteractorStyle(self.interactor_style)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1,1,1)
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, 0, -100)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, -1, 0)
        self.surface_view.GetRenderWindow().AddRenderer(self.renderer)
        self.text_patient = vtk.vtkTextActor()
        self.text_patient.SetInput("No lumen or centerlines file found.")
        self.text_patient.SetDisplayPosition(10, 10)
        self.text_patient.GetTextProperty().SetColor(0, 0, 0)
        self.text_patient.GetTextProperty().SetFontSize(20)
        self.renderer.AddActor(self.text_patient)

        # add everything to a layout
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.button_max_diameter)
        self.button_layout.addWidget(QLabel("|"))
        self.button_layout.addWidget(self.button_show_diameter)
        self.button_layout.addWidget(self.button_set_landmarks)
        self.button_layout.addWidget(self.dropdown_landmarks)
        self.button_layout.addWidget(self.button_remove_landmarks)
        self.button_layout.addWidget(QLabel("|"))
        self.button_layout.addWidget(self.button_volume_computation)
        self.button_layout.addWidget(self.button_confirm_bounds)
        self.button_layout.addWidget(self.button_add_markers)
        self.button_layout.addStretch()
        self.ahi_layout = QHBoxLayout()
        self.ahi_layout.addWidget(self.height_label)
        self.ahi_layout.addWidget(self.line_edit_height)
        self.volume_surface_layout = QHBoxLayout()
        self.volume_surface_layout.addWidget(self.volume_label)
        self.volume_surface_layout.addWidget(self.surface_label)
        self.diameter_layout = QVBoxLayout()
        self.diameter_layout.addWidget(self.landmark_table) 
        self.diameter_layout.addWidget(self.current_diameter_label)
        self.diameter_layout.addWidget(self.max_diameter_label)
        self.diameter_layout.addLayout(self.ahi_layout)
        self.diameter_layout.addWidget(self.ahi_label)
        self.diameter_layout.addLayout(self.volume_surface_layout)
        self.view_layout = QHBoxLayout()
        self.view_layout.addWidget(self.surface_view) 
        self.view_layout.addLayout(self.diameter_layout)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addLayout(self.button_layout) 
        self.main_layout.addLayout(self.view_layout)
        
        # lumen vtk pipeline
        self.reader_lumen = vtk.vtkSTLReader()
        mapper_lumen = vtk.vtkPolyDataMapper()
        mapper_lumen.SetInputConnection(self.reader_lumen.GetOutputPort())
        self.actor_lumen = vtk.vtkActor()
        self.actor_lumen.SetMapper(mapper_lumen)
        self.actor_lumen.GetProperty().SetColor(0.9,0.9,0.9)
        # actor for volume measurement 
        self.actor_clip_volume = None  
        
        # centerline vtk pipeline
        self.reader_centerline = vtk.vtkXMLPolyDataReader()
        self.max_tube_marker,self.max_tube_filter,self.max_tube_actor = self.__setupMarker()
        self.diameter_tube_marker,self.diameter_tube_filter,self.diameter_tube_actor = self.__setupMarker(opacity=0.4)
        self.picker = vtk.vtkPropPicker()
        
        # start render window
        self.surface_view.Initialize()
        self.surface_view.Start()
    

    def __setupLandmarkTable(self):
        self.landmark_colors = LANDMARK_COLORS
        # set structure
        self.landmark_table = QTableWidget() 
        self.landmark_table.setRowCount(10)
        self.landmark_table.setColumnCount(3)
        self.landmark_table.setHorizontalHeaderLabels(["Landmark","Diameter",""])
        self.landmarks = ["Sinotubular junction", "Mid-ascending aorta", "Distal ascending aorta","Aortic arch", "Proximal descending aorta", "Mid-descending aorta"]
        # fill table with colors and predefined landmarks
        for i in range(len(self.landmark_colors)):
            if i < len(self.landmarks):
                self.landmark_table.setItem(i,0,QTableWidgetItem(self.landmarks[i]))
            color_item = QTableWidgetItem("")
            color = QColor(int(self.landmark_colors[i][0]*255), int(self.landmark_colors[i][1]*255), int(self.landmark_colors[i][2]*255))
            color_item.setBackground(color)
            self.landmark_table.setItem(i,2,color_item)
        # format table 
        self.landmark_table.resizeColumnsToContents()
        header = self.landmark_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.landmark_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.landmark_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.landmark_table.setSizePolicy(sizePolicy)
        self.landmark_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.landmark_table.setEnabled(False)

    
    def __setupMarker(self, opacity=None,color=None):
        # marker for max diameter and diameter of main branch 
        marker = vtk.vtkLineSource()
        filter = vtk.vtkTubeFilter()
        filter.SetNumberOfSides(25)
        filter.CappingOn()
        filter.SetInputConnection(marker.GetOutputPort())
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if color:
            actor.GetProperty().SetColor(color)
        else:
            actor.GetProperty().SetColor(0,0,0)
        actor.GetProperty().SetInterpolationToFlat()
        actor.GetProperty().LightingOff()
        
        if opacity:
            actor.GetProperty().SetOpacity(opacity) 

        return marker, filter, actor

    def __fullSetupVolumeBound(self,idx):
        id0,id1 = idx[0],idx[1]
        marker, filter, actor = self.__setupMarker(opacity=0.4)  # marker, filter, actor

        marker.SetPoint1(self.centerline.c_pos_lists[id0][id1-1])
        marker.SetPoint2(self.centerline.c_pos_lists[id0][id1+1])
        filter.SetRadius(1.5*self.centerline.c_radii_lists[id0][id1])
        self.volume_marker_line.append(marker)
        self.volume_marker_filter.append(filter)
        self.volume_marker_actors.append(actor)
        self.volume_marker_ids.append(idx)

            
    def loadPatient(self,patient_dict,centerline):
        # reset scene 
        self.patient_dict = patient_dict
        self.centerline = centerline
        lumen_file, centerline_file, metrics_file = patient_dict["model"], patient_dict["centerlines"], patient_dict["metrics"]
        self.renderer.RemoveActor(self.max_tube_actor)
        self.renderer.RemoveActor(self.diameter_tube_actor)
        if self.volume_marker_actors:
            for actor in self.volume_marker_actors:
                self.renderer.RemoveActor(actor)
            self.volume_marker_actors = []
            self.volume_marker_ids = []
            self.volume_marker_filter = []
            self.volume_marker_line = []
        if self.actor_clip_volume:
            self.renderer.RemoveActor(self.actor_clip_volume)
            self.actor_clip_volume = None
            self.current_volume = None
            self.current_surface = None
        self.max_diameter_id = None
        self.current_diameter_id = [0,20]   
        self.current_branch_id = 0
        self.patient_height = None
        self.line_edit_height.clear()
        self.ahi_score = None
        self.removeAllLandmarkActors()
        self.wheel_active = False
        self.current_diameter_label.setText("Current Diameter:")
        self.label_opacity["diameter"].setOpacity(0.6)
        self.max_diameter_label.setText("Maximum Diameter: ")
        self.ahi_label.setText("AHI (diameter/height):        ")
        self.volume_label.setText("Volume: ")
        self.label_opacity["volume"].setOpacity(0.6)
        self.surface_label.setText("Surface: ")
        self.label_opacity["surface"].setOpacity(0.6)
        if self.button_remove_landmarks.isChecked():
            self.button_remove_landmarks.setChecked(False)
        
        if lumen_file and centerline_file:
            # load lumen
            self.reader_lumen.SetFileName("") # forces a reload
            self.reader_lumen.SetFileName(lumen_file)
            self.reader_lumen.Update()
            self.renderer.AddActor(self.actor_lumen)
            self.text_patient.SetInput(os.path.basename(lumen_file)[:-4])
            
            # load scene if saved
            self.button_max_diameter.setEnabled(True)
            self.button_max_diameter.setChecked(False)
            self.button_show_diameter.setEnabled(True)
            self.button_volume_computation.setEnabled(True)
            self.line_edit_height.setEnabled(True)
            if self.button_show_diameter.isChecked():
                self.button_show_diameter.setChecked(False)
                self.diameterMode()
            elif self.button_volume_computation.isChecked():
                self.button_volume_computation.setChecked(False)
                self.volumeMode()
            
            # read in metrics if available
            if metrics_file:
                # remove custom landmarks 
                for row in range(7,10):
                    item = self.landmark_table.item(row,0)
                    if item:  
                        self.deleteCustomLandmark(item,update_table=False)
                df = pd.read_csv(metrics_file)
                # fill table and set markers
                for index,row in df.iterrows():
                    # set max diameter if in file
                    if row["Landmark"] == "Maximum diameter":
                        self.computeMaxDiameter()
                        self.button_max_diameter.click()
                    # for landmarks in file -> add actor
                    else:
                        landmark = row["Landmark"]
                        # add custom landmark to landmark table and dropdown
                        if not landmark in self.landmarks:
                            self.updateLandmarkTable(landmark)
                        centerline_index = row["Centerline Index"]
                        centerline_index = centerline_index.split(", ")
                        id0,id1 = int(centerline_index[0]),int(centerline_index[1])
                        self.landmark_ids[landmark] = [id0,id1]
                        color = self.landmark_colors[self.landmarks.index(landmark)]
                        landmark_actor = self.defineLandmarkActor(id0,id1,True,color)
                        self.landmark_actors[landmark] = landmark_actor
                        self.renderer.AddActor(landmark_actor)
                        #add value to table
                        self.landmark_table.setItem(self.landmarks.index(landmark),1,QTableWidgetItem(str(round(self.centerline.c_radii_lists[id0][id1]*2,2))))
                # check if patient hight saved
                if "Patient Height" in df.columns:
                    height = df["Patient Height"].loc[0]
                    self.line_edit_height.setText(str(height))
                    self.setHeight()  # automatically sets AHI if max_diameter set
                # check for volume computation
                if "Bounds volume" in df.columns:
                    string_bounds = str(df["Bounds volume"])
                    matches = re.findall(r'\[(\d+),\s*(\d+)\]', string_bounds)
                    marker_ids = [[int(a), int(b)] for a, b in matches]

                    # display bounds and compute values
                    for bound in marker_ids:
                        self.__fullSetupVolumeBound(bound)
                        #print(bound)
                    self.determineLeftBound()
                    self.clipVolumeRegion(load_from_csv=True)
                    
                    
        else: 
            self.renderer.RemoveActor(self.actor_lumen)
            self.text_patient.SetInput("No lumen or centerlines file found.")
            self.button_max_diameter.setEnabled(False)
            self.button_show_diameter.setEnabled(False)
            self.button_set_landmarks.setEnabled(False)
            self.dropdown_landmarks.setEnabled(False)
            self.landmark_table.setEnabled(False)
            self.button_remove_landmarks.setEnabled(False)
            self.button_volume_computation.setEnabled(False)
            self.button_confirm_bounds.setEnabled(False)
            self.line_edit_height.setEnabled(False)

        # reset scene and render
        self.renderer.ResetCamera()
        self.surface_view.GetRenderWindow().Render()
    
    
    def computeMaxDiameter(self):
        max_tot = 0.0
        max_tot_id = None
        count_inner_list = 0
        # iterate over branches, fetch max diamter and idx
        for i in self.centerline.c_radii_lists:
            max_branch = np.max(i)
            if max_branch > max_tot:
                max_tot = max_branch
                branch_id = np.argmax(i)
                max_tot_id = (count_inner_list,branch_id)
            count_inner_list += 1
    
        # save max diamter
        self.max_diameter_id = max_tot_id
        id0, id1 = max_tot_id[0], max_tot_id[1] 
        
        # display marker and value
        self.max_tube_marker.SetPoint1(self.centerline.c_pos_lists[id0][id1-1])
        self.max_tube_marker.SetPoint2(self.centerline.c_pos_lists[id0][id1+1])
        self.max_tube_filter.SetInputConnection(self.max_tube_marker.GetOutputPort())
        self.max_tube_filter.SetRadius(1.5*self.centerline.c_radii_lists[id0][id1])
        max_diam = round(max_tot*2,2)
        self.max_diameter_label.setText("Maximum Diameter: " + str(max_diam)+" mm")
        
        # automatically determine AHI if height available
        if self.patient_height:
            self.setAHI()
    
    
    def showMaxDiameterMarker(self):
        # compute max diamteter at first click for patient 
        if not self.max_diameter_id:
            self.computeMaxDiameter()
            self.metrics_changed.emit()
        
        # show/hide diamter marker and text with max diameter depending on button 
        if self.button_max_diameter.isChecked():
            self.renderer.AddActor(self.max_tube_actor)
        else: 
            self.renderer.RemoveActor(self.max_tube_actor)
        
        self.surface_view.GetRenderWindow().Render()
    
    
    def setAHI(self):
        id0,id1 = self.max_diameter_id[0],self.max_diameter_id[1]
        diameter = self.centerline.c_radii_lists[id0][id1]*2
        self.ahi_score = diameter/self.patient_height
        self.ahi_label.setText("AHI (diameter/height):    " + str(round(self.ahi_score,4)))
    
    def setHeight(self):
        # user input: patient height
        height = self.line_edit_height.text()
        self.patient_height = float(height)
        # automatically comput AHI if height and max diameter available 
        if self.patient_height and self.max_diameter_id:
            self.setAHI()
        self.metrics_changed.emit()
    
      
    def diameterMode(self):  
        # show/hide movable actor for diameter 
        if self.button_show_diameter.isChecked(): 
            self.click_diameter = self.interactor_style.AddObserver("RightButtonPressEvent", self.rightClickDiameter)
           
            self.setDiameterMarker()
            self.renderer.AddActor(self.diameter_tube_actor)
            
            #enable to set landmarks
            self.button_set_landmarks.setEnabled(True)
            self.dropdown_landmarks.setEnabled(True)
            self.landmark_table.setEnabled(True)
            self.button_remove_landmarks.setEnabled(True)
            label_text = "Current Diameter:     "+str(round(self.current_diameter,2))+" mm"
            opacity = 1.0
            # ensure that either volume or diamter mode activated
            if self.button_volume_computation.isChecked():
                self.button_volume_computation.setChecked(False)
                self.volumeMode()
        else: 
            self.interactor_style.RemoveObserver(self.click_diameter)
            self.renderer.RemoveActor(self.diameter_tube_actor)
            
            #disable landmarks
            self.button_set_landmarks.setEnabled(False)
            self.dropdown_landmarks.setEnabled(False)
            self.landmark_table.setEnabled(False)
            self.button_remove_landmarks.setEnabled(False)
            label_text = "Current Diameter: "
            opacity = 0.6
        self.current_diameter_label.setText(label_text)
        self.label_opacity["diameter"].setOpacity(opacity)
        self.surface_view.GetRenderWindow().Render()  
    
    
    def setDiameterMarker(self):
        id0,id1 = self.current_diameter_id[0],self.current_diameter_id[1]
        if id1 == len(self.centerline.c_pos_lists[id0])-1 or (id1 == 0):  # catch ends and bifurcations
            return 
        self.current_diameter = self.centerline.c_radii_lists[id0][id1]*2
        self.diameter_tube_marker.SetPoint1(self.centerline.c_pos_lists[id0][id1-1])
        self.diameter_tube_marker.SetPoint2(self.centerline.c_pos_lists[id0][id1+1])
        self.diameter_tube_filter.SetRadius(1.5*self.centerline.c_radii_lists[id0][id1])
        
        self.current_diameter_label.setText("Current Diameter:     "+str(round(self.current_diameter,2))+" mm")
        self.surface_view.GetRenderWindow().Render()
     
    def mouseWheelBackward(self,obj,event):  
        # move diameter marker down along centerline
        id0,id1 = self.current_diameter_id[0],self.current_diameter_id[1]
        
        # catch upper end of vessel
        if id1 == len(self.centerline.c_pos_lists[id0])-1:
            return 
        
        # catch bifurcation -> lead back to last click 
        if tuple(self.current_diameter_id) in self.centerline.c_parent_indices and (self.centerline.c_parent_indices[self.current_branch_id][0] !=0 or self.centerline.c_parent_indices.index(tuple(self.current_diameter_id)) ==self.current_branch_id):  # self.current_branch_id !=0:
            branch = self.centerline.c_parent_indices.index(tuple(self.current_diameter_id))
            self.current_diameter_id = [branch,1]
        else:           
            self.current_diameter_id[1] += 1 
        self.setDiameterMarker()
    
    def mouseWheelForward(self,obj,event):  
        # move diameter marker down along centerline
        id0,id1 = self.current_diameter_id[0],self.current_diameter_id[1]
        
        # catch lower end of vessel
        if id1 <= 1 and id0 == 0:
            return
        
        #catch bifurcation
        elif id1 == 1 and id0 != 0:
            self.current_diameter_id = list(self.centerline.c_parent_indices[id0])
        else:
            self.current_diameter_id[1] -= 1 
        self.setDiameterMarker()  
     
    def rightClickDiameter(self,obj,event):
        self.positionDiameter(obj,event)
        self.move_diameter = self.interactor_style.AddObserver("MouseMoveEvent",self.positionDiameter)
        self.release_click = self.interactor_style.AddObserver("RightButtonReleaseEvent", self.endPositionDiameter)
        
    def positionDiameter(self,obj,event): 
        # if click on surface: set diameter marker at this position
        x,y = self.surface_view.GetEventPosition()
        clostest_id = self.centerline.getClosestCenterlinePoint(x,y,self.renderer)
        if clostest_id != None:
            # only allow scroll if not on surface
            if self.wheel_active:
                self.interactor_style.RemoveObserver(self.mouse_forward)
                self.interactor_style.RemoveObserver(self.mouse_backward)
                self.wheel_active = False
            self.current_diameter_id = clostest_id
            self.current_branch_id = clostest_id[0]
            self.setDiameterMarker()
        # if click not on surface: add observers to scroll diameter marker
        elif not self.wheel_active:
            self.mouse_forward = self.interactor_style.AddObserver("MouseWheelForwardEvent", self.mouseWheelForward)
            self.mouse_backward = self.interactor_style.AddObserver("MouseWheelBackwardEvent", self.mouseWheelBackward)
            self.wheel_active = True
            
    def endPositionDiameter(self,obj,event):
        self.interactor_style.RemoveObserver(self.move_diameter)
        if self.wheel_active:
            self.interactor_style.RemoveObserver(self.mouse_forward)
            self.interactor_style.RemoveObserver(self.mouse_backward)
        self.interactor_style.RemoveObserver(self.release_click)
        self.wheel_active = False
     
     
    def updateLandmarkTable(self,new_landmark):
        # add new landmark
        placement = len(self.landmarks)
        self.dropdown_landmarks.insertItem(placement,new_landmark)
        self.landmarks.append(new_landmark)
        self.landmark_table.setItem(placement,0,QTableWidgetItem(new_landmark))
        return placement
    
    def setLandmark(self):
        landmark = self.dropdown_landmarks.currentText()
        # get userinput for new landmark
        if landmark == "Other ...":
            new_landmark, ok = QInputDialog.getText(self, "Set New Landmark", "Enter name of new landmark:")
            if new_landmark and ok and new_landmark not in self.landmarks:
                landmark = new_landmark
                row = self.updateLandmarkTable(landmark)
                if row == 9:
                    self.dropdown_landmarks.removeItem(10)
            else:
                return
        
        # show marker at landmark position
        self.landmark_ids[landmark] = self.current_diameter_id.copy()
        color = self.landmark_colors[self.landmarks.index(landmark)]
        landmark_actor = self.defineLandmarkActor(color=color)
        
        # make sure that every landmark is only defined once for patient
        if landmark in self.landmark_actors:
            self.renderer.RemoveActor(self.landmark_actors[landmark])
        self.landmark_actors[landmark] = landmark_actor
        self.renderer.AddActor(self.landmark_actors[landmark])
        self.surface_view.GetRenderWindow().Render()
        
        #add value to table
        self.landmark_table.setItem(self.landmarks.index(landmark),1,QTableWidgetItem(str(round(self.current_diameter,2))))
        self.metrics_changed.emit()

    def defineLandmarkActor(self,id0=None,id1=None,loadPatient=False,color=None):
        # create actor to show marker 
        if not loadPatient:
            id0,id1 = self.current_diameter_id[0],self.current_diameter_id[1]
            
        marker, filter, actor = self.__setupMarker(color=color)
        marker.SetPoint1(self.centerline.c_pos_lists[id0][id1-1])
        marker.SetPoint2(self.centerline.c_pos_lists[id0][id1+1])
        filter.SetRadius(1.5*self.centerline.c_radii_lists[id0][id1])
        return actor

    
    def deleteCustomLandmark(self,item,update_table=True):
        landmark = item.text()
        landmark_id = self.landmarks.index(landmark)  
        # ensure that landmark checked (not diameter) and custom landmark (6 predefined)
        if not item.column()==0 or not landmark_id>5:
            return
        
        # remove from table and update dropdown  
        self.landmark_table.takeItem(landmark_id,0) 
        self.landmark_table.takeItem(landmark_id,1) 
        self.dropdown_landmarks.removeItem(landmark_id)   
        if landmark in self.landmark_actors:
            self.removeSelectedLandmark(landmark)
        if landmark_id==9:
            self.dropdown_landmarks.addItem("Other ...")
        self.landmarks.remove(landmark) 
        
        if update_table:
            for row in range(landmark_id+1,self.landmark_table.rowCount()):
                if self.landmark_table.item(row,0):
                    landmark_item = self.landmark_table.item(row,0)
                    self.landmark_table.takeItem(row,0)
                    self.landmark_table.setItem(row-1,0,landmark_item)
                    diameter_item = self.landmark_table.item(row,1)
                    self.landmark_table.takeItem(row,1)
                    self.landmark_table.setItem(row-1,1,diameter_item)
                    
                    new_color = self.landmark_colors[row-1]
                    self.landmark_actors[landmark_item.text()].GetProperty().SetColor(new_color)
                    self.surface_view.GetRenderWindow().Render()

        
    def removeLandmarkMode(self):
        if self.button_remove_landmarks.isChecked():
            # disable setting landmarks and diameter
            self.button_show_diameter.setEnabled(False)
            self.wheel_active = False
            self.button_set_landmarks.setEnabled(False)
            self.dropdown_landmarks.setEnabled(False)
            
            self.interactor_style.RemoveObserver(self.click_diameter)
            
            # add observers for removal
            self.click_remove_landmark = self.interactor_style.AddObserver("RightButtonPressEvent",self.rightClickLandmark)
            
        else:
            # reactivate setting of landmarks
            self.button_show_diameter.setEnabled(True)
            self.button_set_landmarks.setEnabled(True)
            self.dropdown_landmarks.setEnabled(True)
            
            # add back observers
            self.click_diameter = self.interactor_style.AddObserver("RightButtonPressEvent", self.rightClickDiameter)
            
            # remove landmark observers
            self.interactor_style.RemoveObserver(self.click_remove_landmark)
    
    def removeSelectedLandmark(self,landmark):
        if not landmark in self.landmark_actors:
            return
        # remove vtk object and saved position
        actor = self.landmark_actors[landmark]
        self.renderer.RemoveActor(actor)
        del self.landmark_actors[landmark]
        del self.landmark_ids[landmark]
        
        # remove diameter value from table
        row = self.landmarks.index(landmark)
        self.landmark_table.takeItem(row,1)
        self.metrics_changed.emit()
        self.surface_view.GetRenderWindow().Render()
    
    def rightClickLandmark(self,obj,event):
        x,y = self.surface_view.GetEventPosition()
        clostest_id = self.centerline.getClosestCenterlinePoint(x,y,self.renderer)
        # catch click not on surface
        if clostest_id == None:
            return
        # check if landmark near position -> remove
        delete_landmark = None
        d_min = float('inf')
        for landmark in self.landmark_ids:
            id0,id1 = self.landmark_ids[landmark][0],self.landmark_ids[landmark][1]
            landmark_pos = self.centerline.c_pos_lists[id0][id1]
            d = ((np.array(self.centerline.c_pos_lists[clostest_id[0]][clostest_id[1]])-np.array(landmark_pos))**2).sum()
            if d <=10 and d<d_min:
                d_min = d
                delete_landmark = landmark
        if delete_landmark:
            self.removeSelectedLandmark(delete_landmark)
            
    
    def clickTableLandmark(self,item):
        landmark = item.text()
        # if remove landmark activated remove selected landmark 
        if self.button_remove_landmarks.isChecked():
            self.removeSelectedLandmark(landmark)
        # if landmark in row set as selected landmark
        elif landmark in self.landmarks:
            self.dropdown_landmarks.setCurrentText(landmark)
            

    def removeAllLandmarkActors(self):
        for landmark in self.landmark_actors:
            self.renderer.RemoveActor(self.landmark_actors[landmark])
            row = self.landmarks.index(landmark)
            self.landmark_table.takeItem(row,1)

        self.landmark_actors = {}
        self.landmark_ids = {}
    

    def volumeMode(self):
        label_text_v = "Volume: "
        label_text_s = "Surface: "
        opacity = 0.6
        if self.button_volume_computation.isChecked():
            self.button_confirm_bounds.setEnabled(True)
            # ensure that either volume or diamter mode activated
            if self.button_show_diameter.isChecked():
                self.button_show_diameter.setChecked(False)
                self.diameterMode()
            if not self.volume_marker_actors:
                # setup left and right bound
                self.__fullSetupVolumeBound([0,50])
                self.__fullSetupVolumeBound([0,100])
                self.surface_view.GetRenderWindow().Render()

            # display measurement if available 
            for marker in self.volume_marker_actors: 
                self.renderer.AddActor(marker)
            self.right_click_move = self.interactor_style.AddObserver("RightButtonPressEvent", self.moveClosestMarker)
            self.renderer.AddActor(self.actor_clip_volume)
            if self.current_volume:
                label_text_v = "Volume:     "+str(round(self.current_volume/1000,2))+" cm<sup>3</sup>"
                label_text_s = "Surface:    "+str(round(self.current_surface/100,2))+" cm<sup>2</sup>"
                opacity = 1.0

        else: 
            self.button_confirm_bounds.setEnabled(False)
            for marker in self.volume_marker_actors: 
                self.renderer.RemoveActor(marker)  
            self.interactor_style.RemoveObserver(self.right_click_move)
            self.renderer.RemoveActor(self.actor_clip_volume)
        self.volume_label.setText(label_text_v)
        self.label_opacity["volume"].setOpacity(opacity)
        self.surface_label.setText(label_text_s)
        self.label_opacity["surface"].setOpacity(opacity)
        self.surface_view.GetRenderWindow().Render()


    def getClosestMarkerId(self):
        x,y = self.surface_view.GetEventPosition()
        clostest_point_id = self.centerline.getClosestCenterlinePoint(x,y,self.renderer)
        if clostest_point_id: 
            d_min = float('inf')
            closest_marker_id = None
            for i in range(len(self.volume_marker_ids)):
                id0,id1 = self.volume_marker_ids[i][0],self.volume_marker_ids[i][1]
                marker_pos = self.centerline.c_pos_lists[id0][id1]
                d = ((np.array(self.centerline.c_pos_lists[clostest_point_id[0]][clostest_point_id[1]])-np.array(marker_pos))**2).sum()
                if d <=30 and d<d_min:  
                    d_min = d
                    closest_marker_id = i
            return closest_marker_id
    
    def moveClosestMarker(self,obj,event):
        # determine if marker near clicked position -> if found add interaction 
        marker_id = self.getClosestMarkerId()
        if marker_id != None:
            self.current_marker_id = [marker_id,0,0]
            self.current_line = self.volume_marker_line[marker_id]
            self.current_filter = self.volume_marker_filter[marker_id]
            self.positionMarker(obj,event)
            self.move_marker.append(self.interactor_style.AddObserver("MouseMoveEvent",self.positionMarker))
            self.move_marker.append(self.interactor_style.AddObserver("RightButtonReleaseEvent", self.endMoveMarker))
    
    def positionMarker(self,obj,event):
        x,y = self.surface_view.GetEventPosition()
        clostest_point_id = self.centerline.getClosestCenterlinePoint(x,y,self.renderer)
        if clostest_point_id:
            id0,id1 = clostest_point_id[0],clostest_point_id[1]
            if id1 >= len(self.centerline.c_pos_lists[id0])-4 or (id1 <= 4):  # catch ends and bifurcations - 4 necessary due to computation of normal
                return
            self.current_marker_id[1],self.current_marker_id[2] = id0,id1
            self.current_line.SetPoint1(self.centerline.c_pos_lists[id0][id1-1])
            self.current_line.SetPoint2(self.centerline.c_pos_lists[id0][id1+1])
            self.current_filter.SetRadius(1.5*self.centerline.c_radii_lists[id0][id1])
            self.surface_view.GetRenderWindow().Render()
    
    def endMoveMarker(self,obj,event):
        if self.current_marker_id != None: 
            marker_id = self.current_marker_id[0]
            del self.current_marker_id[0]
            # check if near bifurcation -> move to parent branch  
            parent_branch, parent_position = self.centerline.c_parent_indices[self.current_marker_id[0]]
            if self.current_marker_id[1] <= 30:  # and self.current_marker_id[0]!=parent_branch:
                self.current_marker_id = [parent_branch,parent_position]
                self.current_line.SetPoint1(self.centerline.c_pos_lists[parent_branch][parent_position-1])
                self.current_line.SetPoint2(self.centerline.c_pos_lists[parent_branch][parent_position+1])
                self.current_filter.SetRadius(1.5*self.centerline.c_radii_lists[parent_branch][parent_position])
                self.surface_view.GetRenderWindow().Render()
            self.volume_marker_ids[marker_id] = self.current_marker_id
            
        self.current_line = None
        self.current_filter = None
        self.current_marker_id = None
        for observer in self.move_marker:
            self.interactor_style.RemoveObserver(observer)
        self.move_marker = []
    

    def includeBranchesPopUp(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Branching vessels")
        msg_box.setText("Do you want to include the branching vessels into the volume computation?")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes |
                                QMessageBox.StandardButton.No |
                                QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
        msg_box.setIcon(QMessageBox.Icon.Question)
        # Hide the Cancel button visually but still detect if user closes the dialog
        msg_box.button(QMessageBox.StandardButton.Cancel).setVisible(False)

        include = msg_box.exec()

        if include == QMessageBox.StandardButton.No:
            self.button_add_markers.setVisible(True)
            self.button_add_markers.setChecked(True)
            self.addMarkerMode()
            self.new_volume = False

        elif include == QMessageBox.StandardButton.Yes:
            self.clipVolumeRegion()
    
    def addMarkerMode(self):
        if self.button_add_markers.isChecked(): 
            self.interactor_style.RemoveObserver(self.right_click_move)
            self.right_click_add = self.interactor_style.AddObserver("RightButtonPressEvent",self.addMarker)
            self.move_marker = []
        else:
            self.interactor_style.RemoveObserver(self.right_click_add)
            self.right_click_move = self.interactor_style.AddObserver("RightButtonPressEvent", self.moveClosestMarker)
    
    def addMarker(self,obj,event):
        # add Marker at clicked position
        x,y = self.surface_view.GetEventPosition()
        clostest_point_id = self.centerline.getClosestCenterlinePoint(x,y,self.renderer)        
        if clostest_point_id: 
            marker,filter,actor = self.__setupMarker(opacity=0.4)
            id0,id1 = clostest_point_id
            marker.SetPoint1(self.centerline.c_pos_lists[id0][id1-1])
            marker.SetPoint2(self.centerline.c_pos_lists[id0][id1+1])
            filter.SetRadius(1.5*self.centerline.c_radii_lists[id0][id1])
            self.renderer.AddActor(actor)

            self.volume_marker_line.append(marker)
            self.volume_marker_filter.append(filter)
            self.volume_marker_actors.append(actor)
            self.volume_marker_ids.append([id0,id1])
            self.surface_view.GetRenderWindow().Render()
        
    def determineLeftBound(self):
        # determine which bound left, which right for definition of clip function
            distance_to_origin = [0,0]
            bound_no = 0
            volume_bound_ids = self.volume_marker_ids[:2]
            for bound in volume_bound_ids:
                #parent = bound[0]
                while bound[0] != 0:
                    distance_to_origin[bound_no] += bound[1]
                    bound = self.centerline.c_parent_indices[bound[0]]
                    #parent = self.centerline.c_parent_indices[parent][0]
                # add distance on branch 0 
                distance_to_origin[bound_no] += bound[1]
                bound_no += 1
            self.left_marker_id = distance_to_origin.index(min(distance_to_origin))
         
    def confirmBoundsVolume(self):
        # confirm either original preexisting bounds (left,right) or set bounds of bifurcating vessels for volume capping 
        if self.new_volume:
            # remove previous volume if present
            if self.actor_clip_volume:
                self.renderer.RemoveActor(self.actor_clip_volume)
                self.actor_clip_volume = None 
                self.volume_label.setText("Volume: ")
                self.surface_label.setText("Surface: ")
            self.determineLeftBound()
            left_bound_id, right_bound_id = self.volume_marker_ids[self.left_marker_id], self.volume_marker_ids[1-self.left_marker_id]
            # check if there are branching vessels between two preexisting markers 
            bifurcation = left_bound_id[0] != right_bound_id[0]
            for branch in self.centerline.c_child_branches[left_bound_id[0]]:
                branching_id1 = self.centerline.c_parent_indices[branch][1]
                if left_bound_id[1] <= branching_id1 < right_bound_id[1]:
                    bifurcation = True
                    break
            if bifurcation: 
                # enable user to decide how to handle branching vessels
                self.includeBranchesPopUp()
            else:
                # clip directly if not branching vessels
                self.clipVolumeRegion()
        else: 
            self.clipVolumeRegion()
            self.new_volume = True
            self.button_add_markers.setChecked(False)
            self.addMarkerMode()
        
    def get_local_cut(self,flip,id):
        # define local cut function at given bound marker 
        id0,id1 = id
        local_function = vtk.vtkImplicitBoolean()
        local_function.SetOperationTypeToIntersection()
        sphere = vtk.vtkSphere()
        sphere.SetCenter(self.centerline.c_pos_lists[id0][id1])
        sphere.SetRadius(1.6*self.centerline.c_radii_lists[id0][id1])
        local_function.AddFunction(sphere)

        plane = vtk.vtkPlane()
        plane.SetOrigin(self.centerline.c_pos_lists[id0][id1])
        #calculate normal by using points around set centerline point
        pb = np.sum(self.centerline.c_pos_lists[id0][id1-3:id1-1],axis=0)/3  # points before
        pa = np.sum(self.centerline.c_pos_lists[id0][id1+1:id1+3],axis=0)/3  # points after
        normal = pa - pb 
        normal *= flip  # flip normal to get plane in right direction
        normal =  normal / np.linalg.norm(normal)
        plane.SetNormal(normal)
        local_function.AddFunction(plane)
        return local_function

    def close_holes(self,data_open_cut,connection_open_cut):
        # fill holes at cut ends 
        fill_holes_filter = vtk.vtkFillHolesFilter()
        fill_holes_filter.SetInputConnection(connection_open_cut)
        fill_holes_filter.SetHoleSize(1000.0)
        
        # make triangle winding order consistent 
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(fill_holes_filter.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()
        # restore original normals 
        normals.GetOutput().GetPointData().SetNormals(data_open_cut.GetPointData().GetNormals())
        
        # how many cells (original and after filter)
        num_original_cells = data_open_cut.GetNumberOfCells()
        num_new_cells = normals.GetOutput().GetNumberOfCells()
        iterator = normals.GetOutput().NewCellIterator()
        iterator.InitTraversal()
        # iterate over original cells -> remaining cells are newly added Cells at holes
        for c in range(num_original_cells):
            iterator.GoToNextCell()
            
        holeData = vtk.vtkPolyData()
        holeData.Allocate(normals.GetOutput(), num_new_cells-num_original_cells)
        holeData.SetPoints(normals.GetOutput().GetPoints())
        
        cell = vtk.vtkGenericCell()
        # get new cells
        for c in range(num_new_cells-num_original_cells):
            iterator.GetCell(cell)
            holeData.InsertNextCell(iterator.GetCellType(),cell.GetPointIds())
            iterator.GoToNextCell()
            
        # create regions at cut from new cells
        connectivity = vtk.vtkConnectivityFilter()
        connectivity.SetInputData(holeData)
        connectivity.SetExtractionModeToAllRegions()
        connectivity.ColorRegionsOn()
        connectivity.Update()
        capped_holes = []
        merged_cut = vtk.vtkAppendPolyData()
        merged_cut.AddInputData(data_open_cut)
        # save regions as polygons
        for regionId in range(connectivity.GetNumberOfExtractedRegions()):
            region = vtk.vtkConnectivityFilter()
            region.SetInputData(holeData)
            region.SetExtractionModeToSpecifiedRegions()
            region.AddSpecifiedRegion(regionId)
            region.Update()
            
            region_polyData = vtk.vtkPolyData()
            region_polyData.DeepCopy(region.GetOutput())
            capped_holes.append(region_polyData)
            merged_cut.AddInputData(region_polyData)
            
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(merged_cut.GetOutputPort())
        cleaner.Update()

        # determine area of original bound caps (preexisting bounds)
        ref_points = [self.centerline.c_pos_lists[self.volume_marker_ids[0][0]][self.volume_marker_ids[0][1]], 
                      self.centerline.c_pos_lists[self.volume_marker_ids[1][0]][self.volume_marker_ids[1][1]]]
        bound_caps = []
        for ref_pt in ref_points:
            min_dist = float("inf")
            closest_region = None
            for i, region_polydata in enumerate(capped_holes):
                implicit_distance = vtk.vtkImplicitPolyDataDistance()
                implicit_distance.SetInput(region_polydata)

                dist = abs(implicit_distance.EvaluateFunction(ref_pt))
                if dist < min_dist:
                    min_dist = dist
                    closest_region = i
            bound_caps.append(closest_region)
        # compute surface area of caps
        combined_area = 0
        for i in bound_caps:
            triangle_filter = vtk.vtkTriangleFilter()
            triangle_filter.SetInputData(capped_holes[i])
            triangle_filter.Update()

            mass = vtk.vtkMassProperties()
            mass.SetInputData(triangle_filter.GetOutput())
            mass.Update()

            area = mass.GetSurfaceArea()
            combined_area += area
        return cleaner.GetOutput(), combined_area

    def clipVolumeRegion(self,load_from_csv=False):
        # define cut function using set markers
        clip_function = vtk.vtkImplicitBoolean()
        clip_function.SetOperationTypeToUnion()
        i = 0
        for bound in self.volume_marker_ids:
            if i == self.left_marker_id:
                flip = 1
                reference_point = self.centerline.c_pos_lists[bound[0]][bound[1]]
            else: 
                flip = -1
            clip_function.AddFunction(self.get_local_cut(flip, bound))
            i += 1
            
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.reader_lumen.GetOutput())
        clipper.SetClipFunction(clip_function)
        clipper.Update()

        # extract correct area from cut surface
        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputData(clipper.GetOutput())
        connectivity_filter.SetExtractionModeToClosestPointRegion()
        connectivity_filter.SetClosestPoint(reference_point)
        connectivity_filter.Update()

        # display area of volume measurement on top of original surface
        normals_cliped_volume = vtk.vtkTriangleMeshPointNormals()
        normals_cliped_volume.SetInputConnection(connectivity_filter.GetOutputPort())
        bloated_volume = vtk.vtkWarpVector()
        bloated_volume.SetInputConnection(normals_cliped_volume.GetOutputPort())
        bloated_volume.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, vtk.vtkDataSetAttributes.NORMALS)
        bloated_volume.SetScaleFactor(0.03)
        clipMapper = vtk.vtkDataSetMapper()
        clipMapper.SetInputConnection(bloated_volume.GetOutputPort())
        self.actor_clip_volume = vtk.vtkActor()  
        self.actor_clip_volume.SetMapper(clipMapper)
        self.actor_clip_volume.GetProperty().SetColor(0.5,0.5,0.5)
        
        # determine volume and surface area
        closed_cut, bound_surface_area = self.close_holes(connectivity_filter.GetOutput(),connectivity_filter.GetOutputPort())  
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(closed_cut)
        mass_properties.Update()
        self.current_volume = mass_properties.GetVolume()
        self.current_surface = mass_properties.GetSurfaceArea()  # problem here: patches on side of bounds 
        self.current_surface -=  bound_surface_area

        # remove branch cut markers
        if len(self.volume_marker_actors) > 2:
            for branch_bound_actor in self.volume_marker_actors[2:]:
                self.renderer.RemoveActor(branch_bound_actor)
            self.volume_marker_actors = self.volume_marker_actors[:2]
            self.volume_marker_ids = self.volume_marker_ids[:2]
            self.volume_marker_filter = self.volume_marker_filter[:2]
            self.volume_marker_line = self.volume_marker_line[:2]
                
        if not load_from_csv:
            self.renderer.AddActor(self.actor_clip_volume)
            self.volume_label.setText("Volume:     "+str(round(self.current_volume/1000,2))+" cm<sup>3</sup>")
            self.label_opacity["volume"].setOpacity(1.0)
            self.surface_label.setText("Surface:   "+str(round(self.current_surface/100,2))+" cm<sup>2</sup>")
            self.label_opacity["surface"].setOpacity(1.0)
            self.button_add_markers.setVisible(False)
            self.metrics_changed.emit()
        self.surface_view.GetRenderWindow().Render()



    def save(self):
        patient_ID = self.patient_dict['patient_ID']
        base_path  = self.patient_dict['base_path']
        export_path = os.path.join(base_path, patient_ID + "_metrics.csv")
        
        landmarks = []
        ids = []
        positions = []
        diameters = []
        # collect all set landmarks
        for l in self.landmarks:
            if l in self.landmark_ids:
                landmarks.append(l)
                idx = self.landmark_ids[l]
                idx = str(idx[0]) + ", " + str(idx[1])
                ids.append(idx)
                id0,id1 = self.landmark_ids[l][0],self.landmark_ids[l][1]
                positions.append(self.centerline.c_pos_lists[id0][id1])
                diameters.append(self.centerline.c_radii_lists[id0][id1]*2)
        # append max diameter 
        if self.max_diameter_id:
            landmarks.append("Maximum diameter")
            ids.append(self.max_diameter_id)
            id0,id1 = self.max_diameter_id[0],self.max_diameter_id[1]
            positions.append(self.centerline.c_pos_lists[id0][id1])
            diameters.append(self.centerline.c_radii_lists[id0][id1]*2)  
        # save data in file
        data = {
            "Landmark":landmarks,
            "Position":positions,
            "Centerline Index": ids,
            "Diameter (mm)":diameters
            }
        if self.patient_height:
            data[""] = ""
            patient_height = [""]*len(landmarks)
            patient_height[0] = self.patient_height
            data["Patient Height"] = patient_height
            if self.ahi_score:
                ahi = [""]*len(landmarks)
                ahi[0] = self.ahi_score
                data["AHI (diameter/height)"] = ahi
        if self.current_volume:
            data[""] = ""
            volume_measure, surface_measure, bounds = [""]*len(landmarks),[""]*len(landmarks),[""]*len(landmarks)
            volume_measure[0],surface_measure[0], bounds[0] = self.current_volume, self.current_surface, self.volume_marker_ids
            data["Volume measurement (mm^3)"] = volume_measure
            data["Surface measurement (mm^2)"] = surface_measure
            data["Bounds volume"] = bounds

        export_df = pd.DataFrame(data)
        export_df.to_csv(export_path)
        self.new_metrics.emit()
        
    def discard(self):
        self.loadPatient(self.patient_dict, self.centerline)
        
    def closeEvent(self,event):
        self.surface_view.Finalize()
    
    
    