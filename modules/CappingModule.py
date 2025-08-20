import os 
from math import inf, isinf

import numpy as np 
import vtk 
from vtk.util.numpy_support import numpy_to_vtk  # vtk_to_numpy
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget
)

from defaults import *

class CappingModule(QWidget):
    data_modified = pyqtSignal()
    new_capping = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # state
        self.patient_dict = None 
        self.centerline = None          # preprocessed centerline
        self.cut_points_centerline = None  # changable objects for cutting 
        self.cut_radii_centerline = None 
        self.inletCap =  inf  # TODO
        self.capped_lumen = None               
        self.capped_centerline = None
        self.capped_holes = []
        self.cap_suggestion = -20  # distance from end of centerline 
        self.cap_suggestion_inlet = 15  # distance from source of centerline
        self.current_line = None
        self.current_filter = None
        self.current_marker_id = None
        self.move_marker = []  # workarounnd for mousmove observer
        self.move = False  # handle observer to move marker
        self.bifurcation_message = True
        
        # vtk objects for visualization
        self.actors_holes = []
        self.marker_ids = []
        self.marker_line = []
        self.marker_filter = []
        self.marker_actors = []
        
        # QT UI  
        self.button_suggest_markers = QPushButton("New Markers")
        self.button_suggest_markers.setEnabled(False)
        self.button_suggest_markers.setCheckable(True)
        self.button_add_marker = QPushButton("Add Outlet Marker")
        self.button_add_marker.setEnabled(False)
        self.button_add_marker.setCheckable(True)
        self.button_remove_markers = QPushButton("Remove Outlet Marker") 
        self.button_remove_markers.setEnabled(False)
        self.button_remove_markers.setCheckable(True)
        self.button_remove_all = QPushButton("Remove all Outlet Markers")
        self.button_remove_all.setEnabled(False)
        self.button_cap_ends = QPushButton("Cap Ends At Markers")
        self.button_cap_ends.setEnabled(False)
        
        # connect signals/slots
        self.button_suggest_markers.clicked.connect(self.newCappingMode)
        self.button_add_marker.clicked.connect(self.addMarkerMode)
        self.button_remove_markers.clicked.connect(self.removeMarkerMode)
        self.button_remove_all.clicked.connect(self.removeAllMarkers)
        self.button_cap_ends.clicked.connect(self.discardCapping)   
        self.button_cap_ends.clicked.connect(self.capEnds)
        
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
        
        # add everything to layout 
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.button_suggest_markers)
        self.button_layout.addWidget(self.button_add_marker)
        self.button_layout.addWidget(self.button_remove_markers)
        self.button_layout.addWidget(QLabel("|"))
        self.button_layout.addWidget(self.button_remove_all)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.button_cap_ends)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addWidget(self.surface_view)
        
        # lumen vtk pipeline
        self.reader_lumen = vtk.vtkSTLReader()
        mapper_lumen = vtk.vtkPolyDataMapper()
        mapper_lumen.SetInputConnection(self.reader_lumen.GetOutputPort())
        self.actor_lumen = vtk.vtkActor()
        self.actor_lumen.SetMapper(mapper_lumen)
        self.actor_lumen.GetProperty().SetColor(0.9,0.9,0.9)
        
        #self.reader_capped_lumen = vtk.vtkSTLReader()
        self.mapper_capped_lumen = vtk.vtkPolyDataMapper()
        self.actor_capped_lumen = vtk.vtkActor()
        self.actor_capped_lumen.SetMapper(self.mapper_capped_lumen)
        self.actor_capped_lumen.GetProperty().SetColor(0.9,0.9,0.9)
        self.actor_capped_lumen.GetProperty().SetOpacity(0.7)

        # centerline vtk pipeline
        self.reader_centerline = vtk.vtkXMLPolyDataReader()
        #self.reader_capped_centerline = vtk.vtkXMLPolyDataReader()
        self.mapper_centerline = vtk.vtkPolyDataMapper()
        self.actor_centerline = vtk.vtkActor()
        self.actor_centerline.SetMapper(self.mapper_centerline)
        self.actor_centerline.GetProperty().SetColor(0,0,0)
        self.actor_centerline.GetProperty().SetLineWidth(2)
        self.actor_centerline.GetProperty().RenderLinesAsTubesOn()
        
        # picker for mouse position
        self.picker = vtk.vtkPropPicker()
        
        # start render window
        self.surface_view.Initialize()
        self.surface_view.Start()
    
    
    def __setupMarker(self, idx, color=None):
        # marker pipeline for capping position
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
            
        # set marker position
        id0,id1 = idx[0],idx[1]
        marker.SetPoint1(self.centerline.c_pos_lists[id0][id1-1])
        marker.SetPoint2(self.centerline.c_pos_lists[id0][id1+1])
        filter.SetRadius(1.5*self.centerline.c_radii_lists[id0][id1])
        
        # save components for later access 
        self.marker_line.append(marker)
        self.marker_filter.append(filter)
        self.marker_actors.append(actor)
        self.marker_ids.append([id0,id1])
        self.renderer.AddActor(actor)  

        #return marker, filter, actor
    
    
    def __suggestCapping(self):
        # inlet
        idx = (0,self.cap_suggestion_inlet)
        self.__setupMarker(idx,color=(0.2, 1, 0.2))
        
        # outlets
        for c in range(len(self.centerline.c_pos_lists)):
            idx = (c,len(self.centerline.c_pos_lists[c])+self.cap_suggestion)
            self.__setupMarker(idx,color=(1, 0.2, 0.2))
            
        self.surface_view.GetRenderWindow().Render()
    
    
    def loadPatient(self,patient_dict, centerline):
        self.patient_dict = patient_dict
        self.centerline = centerline
        self.cut_points_centerline = self.centerline.c_pos_lists.copy()
        self.cut_radii_centerline = self.centerline.c_radii_lists.copy()
        lumen_file, centerline_file, capping_file = patient_dict["model"], patient_dict["centerlines"], patient_dict["capping"]
        self.removeAllMarkers(new_patient=True)
        self.discardCapping()
       
        if lumen_file and centerline_file:
            #self.renderer.RemoveActor(self.actor_capped_lumen)
            #self.renderer.RemoveActor(self.actor_centerline)
            # load lumen
            self.reader_lumen.SetFileName("") # forces a reload
            self.reader_lumen.SetFileName(lumen_file)
            self.reader_lumen.Update()
            self.renderer.AddActor(self.actor_lumen)
            # load centerline for capping 
            self.text_patient.SetInput(os.path.basename(lumen_file)[:-4])
            self.reader_centerline.SetFileName("")
            self.reader_centerline.SetFileName(centerline_file)
            self.reader_centerline.Update()
            
            if capping_file:
                self.button_suggest_markers.setEnabled(True)
                self.button_suggest_markers.setChecked(False)  
                self.newCappingMode(new_patient=True)
                # read capped lumen 
                format = "."+capping_file.split(".")[1]
                if "stl" in capping_file:
                    #format = ".stl"
                    reader_capped_lumen = vtk.vtkSTLReader()
                else:
                    #format = ".obj"
                    reader_capped_lumen = vtk.vtkOBJReader()
                reader_capped_lumen.SetFileName("")# forces a reload
                reader_capped_lumen.SetFileName(capping_file)
                reader_capped_lumen.Update()
                self.mapper_capped_lumen.SetInputConnection(reader_capped_lumen.GetOutputPort())
                self.renderer.AddActor(self.actor_capped_lumen)
                self.actor_lumen.GetProperty().SetOpacity(0.2)
                base_path  = self.patient_dict['base_path']
                capping_path = os.path.join(base_path, "models","capping")
                # read hole polygons
                for hole_file in os.listdir(capping_path):
                    if "capped" in hole_file:  # skip clipped lumen and centerline
                        continue
                    path = os.path.join(capping_path,hole_file)
                    if format == ".stl":
                        reader_caps = vtk.vtkSTLReader()
                    else: 
                        reader_caps = vtk.vtkOBJReader()
                    reader_caps.SetFileName(path)
                    reader_caps.Update()
                    self.capped_holes.append(reader_caps.GetOutput())
                    
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(reader_caps.GetOutputPort())
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(0.9,0.9,0.9)
                    self.actors_holes.append(actor)
                    self.renderer.AddActor(actor)
                # read capped centerline 
                file_capped_centerline = os.path.join(capping_path,self.patient_dict['patient_ID'] + "_centerline_capped.vtp")
                if os.path.exists(file_capped_centerline):
                    reader_capped_centerline = vtk.vtkXMLPolyDataReader()
                else:
                    reader_capped_centerline = vtk.vtkOBJReader()
                    file_capped_centerline = os.path.join(capping_path,self.patient_dict['patient_ID'] + "_centerline_capped.obj")
                reader_capped_centerline.SetFileName("") # forces reload
                reader_capped_centerline.SetFileName(file_capped_centerline)
                reader_capped_centerline.Update()
                self.mapper_centerline.SetInputConnection(reader_capped_centerline.GetOutputPort())
                self.renderer.AddActor(self.actor_centerline)
                
            else:
                # instead of capped lumen+centerline load suggestion for capping
                self.button_suggest_markers.setEnabled(False)
                self.button_suggest_markers.setChecked(True)
                self.newCappingMode(new_patient=True)
                self.discardCapping()
            
        else:
            self.discardCapping()
            self.renderer.RemoveActor(self.actor_lumen)
            self.renderer.RemoveActor(self.actor_capped_lumen)
            self.renderer.RemoveActor(self.actor_centerline)
            self.text_patient.SetInput("No lumen or centerlines file found.")
            
            self.button_suggest_markers.setEnabled(False)
            self.button_add_marker.setEnabled(False)
            self.button_remove_all.setEnabled(False)
            self.button_remove_markers.setEnabled(False)
            self.button_cap_ends.setEnabled(False)
            
        
         # reset scene and render
        self.renderer.ResetCamera()
        self.surface_view.GetRenderWindow().Render()
        
    
    def newCappingMode(self,new_patient=False):
        if self.button_suggest_markers.isChecked():
            # suggest cap positions and enable interaction with markers
            self.renderer.RemoveActor(self.actor_capped_lumen)
            self.renderer.RemoveActor(self.actor_centerline)
            for actor in self.actors_holes:
                self.renderer.RemoveActor(actor)
            self.right_click_move = self.interactor_style.AddObserver("RightButtonPressEvent", self.moveClosestMarker)
            self.actor_lumen.GetProperty().SetOpacity(1)
            self.__suggestCapping()
            self.button_add_marker.setEnabled(True)
            self.button_remove_all.setEnabled(True)
            self.button_remove_markers.setEnabled(True)
            self.button_cap_ends.setEnabled(True)  
            self.move = True
        else:
            # hide markers, show capped geometries and old lumen for comparison 
            self.removeAllMarkers(new_patient=True)
            self.button_add_marker.setEnabled(False)
            self.button_remove_all.setEnabled(False)
            self.button_remove_markers.setEnabled(False)
            self.button_cap_ends.setEnabled(False)
            self.renderer.AddActor(self.actor_capped_lumen)
            self.renderer.AddActor(self.actor_centerline)
            for actor in self.actors_holes:
                self.renderer.AddActor(actor)
            self.actor_lumen.GetProperty().SetOpacity(0.2)
            if self.move:
                self.interactor_style.RemoveObserver(self.right_click_move)
        if not new_patient:
            self.surface_view.GetRenderWindow().Render()
    
    
    def getClosestMarkerId(self):
        x,y = self.surface_view.GetEventPosition()
        clostest_point_id = self.centerline.getClosestCenterlinePoint(x,y,self.renderer)
        if clostest_point_id: 
            d_min = float('inf')
            closest_marker_id = None
            for i in range(len(self.marker_ids)):
                id0,id1 = self.marker_ids[i][0],self.marker_ids[i][1]
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
            self.current_line = self.marker_line[marker_id]
            self.current_filter = self.marker_filter[marker_id]
            self.positionMarker(obj,event)
            self.move_marker.append(self.interactor_style.AddObserver("MouseMoveEvent",self.positionMarker))
            #self.release_click = self.interactor_style.AddObserver("RightButtonReleaseEvent", self.endMoveMarker)
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
    
    def infoPupUp(self):
        message_box = QMessageBox(self)
        message_box.setWindowTitle("Repositioning of Marker")
        message_box.setText("The marker has been placed near a bifurcation. To ensure a clean cut, the marker will be moved on the parent branch.")
        message_box.setStandardButtons(QMessageBox.StandardButton.Ok)

        checkbox = QCheckBox("Don't show this message again.")
        message_box.setCheckBox(checkbox)
        button = message_box.exec()
        if button == QMessageBox.StandardButton.Ok and checkbox.isChecked():
            print("Do not show")
            self.bifurcation_message = False


    def endMoveMarker(self,obj,event):
        if self.current_marker_id != None: 
            marker_id = self.current_marker_id[0]
            del self.current_marker_id[0]
            # check if near bifurcation -> move to parent branch  
            # use unprocessed centerline? 
            parent_branch, parent_position = self.centerline.c_parent_indices[self.current_marker_id[0]]
            if self.current_marker_id[1] <= 30 and marker_id > 0:  # and self.current_marker_id[0]!=parent_branch:  # 30: set sensitivity around bifurcation, marker_id>0: ignore inlet
                self.current_marker_id = [parent_branch,parent_position]
                self.current_line.SetPoint1(self.centerline.c_pos_lists[parent_branch][parent_position-1])
                self.current_line.SetPoint2(self.centerline.c_pos_lists[parent_branch][parent_position+1])
                self.current_filter.SetRadius(1.5*self.centerline.c_radii_lists[parent_branch][parent_position])
                # inform user why marker has to be used
                if self.bifurcation_message:
                    self.infoPupUp()
                self.surface_view.GetRenderWindow().Render()
            self.marker_ids[marker_id] = self.current_marker_id
            
        self.current_line = None
        self.current_filter = None
        self.current_marker_id = None
        for observer in self.move_marker:
            self.interactor_style.RemoveObserver(observer)
        self.move_marker = []
        #self.interactor_style.RemoveObserver(self.release_click)
        
        
    def addMarkerMode(self):
        if self.button_add_marker.isChecked(): 
            self.right_click_add = self.interactor_style.AddObserver("RightButtonPressEvent",self.addMarker)
            self.move_marker = []
            # ensure that either add or remove button is checked
            if self.button_remove_markers.isChecked():
                self.button_remove_markers.setChecked(False)
                self.removeMarkerMode()
        else:
            self.interactor_style.RemoveObserver(self.right_click_add)
    
    def addMarker(self,obj,event):
        # add Marker at clicked position
        x,y = self.surface_view.GetEventPosition()
        clostest_point_id = self.centerline.getClosestCenterlinePoint(x,y,self.renderer)        
        if clostest_point_id: 
            self.__setupMarker(clostest_point_id,color=(1, 0.2, 0.2))
            self.surface_view.GetRenderWindow().Render()

            
    def removeMarkerMode(self):
        if self.button_remove_markers.isChecked(): 
            
            self.interactor_style.RemoveObserver(self.right_click_move)
            self.right_click_remove = self.interactor_style.AddObserver("RightButtonPressEvent",self.removeMarker)
            # ensure that either add or remove button is checked
            if self.button_add_marker.isChecked():
                self.button_add_marker.setChecked(False)
                self.addMarkerMode()
        else:
            self.interactor_style.RemoveObserver(self.right_click_remove)
            self.right_click_move = self.interactor_style.AddObserver("RightButtonPressEvent", self.moveClosestMarker)
            
    def removeMarker(self,obj,event):
        delete_marker_id = self.getClosestMarkerId()
        # disable removing inlet marker
        if delete_marker_id == 0:
            return
        if delete_marker_id:
            actor = self.marker_actors[delete_marker_id]
            self.renderer.RemoveActor(actor)
            del self.marker_line[delete_marker_id]
            del self.marker_filter[delete_marker_id]
            del self.marker_actors[delete_marker_id]
            del self.marker_ids[delete_marker_id]
            self.surface_view.GetRenderWindow().Render()
    
    
    def removeAllMarkers(self,new_patient=False):
        # determine if inlet should be deleted too
        if new_patient:
            idx = 0
        else: 
            idx = 1
        for i in range(len(self.marker_actors)):
            if i == 0 and not new_patient:
                continue
            self.renderer.RemoveActor(self.marker_actors[i])
        del self.marker_line[idx:]
        del self.marker_filter[idx:]
        del self.marker_actors[idx:]
        del self.marker_ids[idx:]
        
        if not new_patient:
            self.surface_view.GetRenderWindow().Render()
    

    def defineSpheresBranch(self,id,inlet=False,following_branch=False):
        # define sphere cut function along the given branch -> cover whole branch 
        id0,id1 = id[0],id[1]
        spheres = vtk.vtkImplicitBoolean()
        spheres.SetOperationTypeToUnion()
        # define region (on centerline) and size dependent of case
        if inlet:
            start = 1
            stop = id1
            multiplicator = 3
            end_multiplicator = multiplicator
        elif following_branch:
            start = 1
            stop = len(self.centerline.c_pos_lists[id0])
            multiplicator = 1.5
            end_multiplicator = 3
        elif id0 == 0:
            start = id1+1
            stop = len(self.centerline.c_pos_lists[id0])
            multiplicator = 1.5
            end_multiplicator = 3
        else:
            start = id1+1
            stop = len(self.centerline.c_pos_lists[id0])
            multiplicator = 1.5
            end_multiplicator = 3
            
        for id in range(start,stop,10):
            sphere = vtk.vtkSphere()
            sphere.SetCenter(self.centerline.c_pos_lists[id0][id])
            sphere.SetRadius(multiplicator*self.centerline.c_radii_lists[id0][id])
            spheres.AddFunction(sphere)
        # ensure that end is also clipped
        sphere = vtk.vtkSphere()
        sphere.SetCenter(self.centerline.c_pos_lists[id0][stop-2])  # diameter at end of centerline not suitable -> get diameter further up
        sphere.SetRadius(end_multiplicator*self.centerline.c_radii_lists[id0][id1])
        spheres.AddFunction(sphere)
        return spheres
    
    def createCenterlinePolydata(self,positions,radii,omit_branches):
        # add back overlaps 
        offset = [0]*len(self.centerline.c_pos_lists)
        offset[0] = self.marker_ids[0][1]
        for parent in self.centerline.c_child_branches:
            for child in self.centerline.c_child_branches[parent]:
                if child in omit_branches or self.centerline.c_parent_indices[child][0]==0: # skip completely cut branches
                    continue
                branch_id = self.centerline.c_parent_indices[child][1]+offset[parent]
                offset[child] = branch_id  # compensate for added points at bifurcations
                positions[child] = np.concatenate((positions[parent][:branch_id].copy(),positions[child].copy()))  
    	# delete completely cappped branches in end to avoid index error
        omit_branches = sorted(omit_branches,reverse=True)  # remove from list end 
        for child in omit_branches:
            del positions[child]
            del radii[child]

        centerline_polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        radii_flat = vtk.vtkFloatArray()
        radii_flat.SetName('MaximumInscribedSphereRadius')
        point_offset = 0
        
        # for line in positions:
        for l in range(len(positions)):
            line = positions[l]
            branch_points = vtk.vtkPoints()
            for i in range(line.shape[0]):
                # insert points from numpy array to vtk points 
                branch_points.InsertNextPoint(line[i,0], line[i,1], line[i,2])
                points.InsertNextPoint(branch_points.GetPoint(i))

            # adapt vtk line object to number of points 
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(branch_points.GetNumberOfPoints())
            
            for i in range(branch_points.GetNumberOfPoints()):
                # assign ids in line to general point id 
                polyline.GetPointIds().SetId(i,point_offset + i)
            
            lines.InsertNextCell(polyline)
            point_offset += branch_points.GetNumberOfPoints()
            
            # add radii to list 
            for r in radii[l]:
                radii_flat.InsertNextValue(r)
            
        # save data in vtk objects
        centerline_polydata.SetPoints(points)
        centerline_polydata.SetLines(lines)
        centerline_polydata.GetPointData().AddArray(radii_flat)
        return centerline_polydata
    
    def interpolate_color(self):
        color_start = [1.0, 0.0, 0.0]  
        color_end = [0.0, 0.0, 1.0]    

        # create vtk array to store color values
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        points = self.capped_centerline.GetPoints()
        lines = self.capped_centerline.GetLines()

        # iterate through lines 
        lines.InitTraversal()
        for i in range(lines.GetNumberOfCells()):
            #get points Ids for this line
            point_ids = vtk.vtkIdList()
            lines.GetNextCell(point_ids)
            num_points = point_ids.GetNumberOfIds()

            # interpolate color on line points
            for j in range(num_points):
                t = j / (num_points - 1)  # interpolation factor
                color =  [int(255*((1 - t) * color_start[i] + t * color_end[i])) for i in range(3)] # colors should be in [0,255]
                colors.InsertNextTuple(color)

        # assign colors to polydata 
        self.capped_centerline.GetPointData().SetScalars(colors)
    
    def getClipFunction(self):
        """
        definition of clip function by iterating over clip position - in this process the centerline is clipped and prepared to be saved into a vtk object
        """
        # get copy of processed centerline 
        cap_points_centerline = self.centerline.c_pos_lists.copy()
        cap_radii_centerline = self.centerline.c_radii_lists.copy()
        
        clip_function = vtk.vtkImplicitBoolean()
        clip_function.SetOperationTypeToUnion()
        children = []
        # iterate over markers and define local clip functions
        for i in range(len(self.marker_ids)):
            offset = 0 
            marker = self.marker_ids[i]
            id0,id1 = marker[0],marker[1]
            if i == 0:  
                flip = 1
                clip_function_spheres = self.defineSpheresBranch(marker,inlet=True)
                cap_points_centerline[0] = cap_points_centerline[0][self.marker_ids[0][1]:]  
                cap_radii_centerline[0] = cap_radii_centerline[0][self.marker_ids[0][1]:]
            else: 
                flip = -1
                clip_function_spheres = self.defineSpheresBranch(marker)
                
                #cut centerline 
                if id0 == 0:
                    offset = self.marker_ids[0][1]
                cap_points_centerline[id0] = cap_points_centerline[id0][:id1-offset]  
                cap_radii_centerline[id0] = cap_radii_centerline[id0][:id1-offset]
                # check for child branches/bifurcations
                if id0 in self.centerline.c_child_branches:
                    # check if bifurcation is after marker
                    check_child = []
                    for child in self.centerline.c_child_branches[id0]:
                        if self.centerline.c_parent_indices[child][1] >= id1:
                            check_child.append(child)
                    j = 0
                    while j < len(check_child) and j<100:  # TODO: while == problem??
                        current_child = check_child[j]
                        # check for subbranches 
                        if current_child in self.centerline.c_child_branches:
                            check_child.extend(self.centerline.c_child_branches[current_child])
                        children.append(current_child)
                        j += 1     

            # get clean cut at marker position
            plane = vtk.vtkPlane()
            plane.SetOrigin(self.centerline.c_pos_lists[id0][id1])
            #calculate normal by using points around set centerline point
            pb = np.sum(self.centerline.c_pos_lists[id0][id1-3:id1-1],axis=0)/3  # points before
            pa = np.sum(self.centerline.c_pos_lists[id0][id1+1:id1+3],axis=0)/3  # points after
            normal = pa - pb 
            normal *= flip  # flip normal to get plane in right direction
            normal =  normal / np.linalg.norm(normal)
            plane.SetNormal(normal)
            
            clip_function_local = vtk.vtkImplicitBoolean()
            clip_function_local.SetOperationTypeToIntersection()
            clip_function_local.AddFunction(clip_function_spheres)
            clip_function_local.AddFunction(plane)
            
            clip_function.AddFunction(clip_function_local)
        self.capped_centerline = self.createCenterlinePolydata(cap_points_centerline,cap_radii_centerline,children)
        self.mapper_centerline.SetInputData(self.capped_centerline)
        
        # test if connectivity contained
        #self.interpolate_color()
        return clip_function

    def computeCappedHoles(self,connection_lumen):
        # fill holes at cut ends 
        fill_holes_filter = vtk.vtkFillHolesFilter()
        fill_holes_filter.SetInputConnection(connection_lumen)
        fill_holes_filter.SetHoleSize(1000.0)
        
        # make triangle winding order consistent 
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(fill_holes_filter.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()
        # restore original normals 
        normals.GetOutput().GetPointData().SetNormals(self.capped_lumen.GetPointData().GetNormals())
        
        # how many cells (original and after filter)
        num_original_cells = self.capped_lumen.GetNumberOfCells()
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
        
        # save regions as polygons
        for regionId in range(connectivity.GetNumberOfExtractedRegions()):
            region = vtk.vtkConnectivityFilter()
            region.SetInputData(holeData)
            region.SetExtractionModeToSpecifiedRegions()
            region.AddSpecifiedRegion(regionId)
            region.Update()
            
            region_polyData = vtk.vtkPolyData()
            region_polyData.DeepCopy(region.GetOutput())
            self.capped_holes.append(region_polyData)
        # vtk pipeline for holes holes
        for hole in self.capped_holes:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(hole)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.9,0.9,0.9)
            self.actors_holes.append(actor)
    
    def capEnds(self): 
        clip_function = self.getClipFunction()  # also deals with capping centerline 
        clipper_lumen = vtk.vtkClipPolyData()
        clipper_lumen.SetInputData(self.reader_lumen.GetOutput())
        clipper_lumen.SetClipFunction(clip_function)
        clipper_lumen.Update()
    
        # remove remaining branch parts/artifacts of surface
        connectivity_filter = vtk.vtkConnectivityFilter()
        connectivity_filter.SetInputConnection(clipper_lumen.GetOutputPort())
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.ColorRegionsOff()
        self.capped_lumen = connectivity_filter.GetOutput()
        self.mapper_capped_lumen.SetInputConnection(connectivity_filter.GetOutputPort())  
        self.mapper_capped_lumen.ScalarVisibilityOff()
        
        self.computeCappedHoles(connectivity_filter.GetOutputPort()) 

        # update the scene (remove markers, show capping)
        self.button_suggest_markers.setEnabled(True)
        self.button_suggest_markers.setChecked(False)
        self.newCappingMode()
        
        self.surface_view.GetRenderWindow().Render()
        self.data_modified.emit()


    def discardCapping(self):  
        self.capped_holes = []
        for actor in self.actors_holes:
            self.renderer.RemoveActor(actor)
        self.actors_holes = []
        self.capped_lumen = None
        self.capped_centerline = None
        
        
    def writeFile(self,format,path,data,centerline=False):
        if centerline: 
            if format == ".vtp":
                writer = vtk.vtkXMLPolyDataWriter()
            else: 
                writer = vtk.vtkOBJWriter()
        else:
            if format == ".stl":
                writer = vtk.vtkSTLWriter() 
            else:
                writer = vtk.vtkOBJWriter()
        writer.SetFileName(path)
        writer.SetInputData(data)
        writer.Write()    

    
    def fileDialog(self):
        dlg = QMessageBox()
        dlg.setWindowTitle("Fileformat for capped objects")
        dlg.setText("Choose a file format to save the capped objects.")
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        
        widget = QWidget()
        # define buttons 
        label_lumen = QLabel("Lumen")
        option_stl_lumen = QRadioButton(".stl")
        option_stl_lumen.setChecked(True)   # default
        option_obj_lumen = QRadioButton(".obj")
        label_centerline = QLabel("Centerline")   
        option_vtp_centerline = QRadioButton(".vtp")
        option_vtp_centerline.setChecked(True)  # default
        option_obj_centerline = QRadioButton(".obj")
        
        # buttons should be exclusiv
        button_box_lumen = QButtonGroup()
        button_box_lumen.setExclusive(True)
        button_box_lumen.addButton(option_stl_lumen)
        button_box_lumen.addButton(option_obj_lumen)
        button_box_centerline = QButtonGroup()
        button_box_centerline.setExclusive(True)
        button_box_centerline.addButton(option_vtp_centerline)
        button_box_centerline.addButton(option_obj_centerline)
        
        # add to layout 
        layout_lumen = QHBoxLayout()
        layout_lumen.addWidget(label_lumen)
        layout_lumen.addWidget(option_stl_lumen)
        layout_lumen.addWidget(option_obj_lumen)
        layout_centerline = QHBoxLayout()
        layout_centerline.addWidget(label_centerline)
        layout_centerline.addWidget(option_vtp_centerline)
        layout_centerline.addWidget(option_obj_centerline)
        dlg_layout = QVBoxLayout()
        dlg_layout.addLayout(layout_lumen)
        dlg_layout.addLayout(layout_centerline)
        widget.setLayout(dlg_layout)
        
        # position buttons in window 
        dlg.layout().addWidget(widget, 1, 0, 1, dlg.layout().columnCount())
        button = dlg.exec()
        
        # fetch choice
        if button == QMessageBox.StandardButton.Ok:
            if option_stl_lumen.isChecked():
                format_lumen = ".stl"
            else: 
                format_lumen = ".obj"

            if option_vtp_centerline.isChecked():
                format_centerline = ".vtp"
            else: 
                format_centerline = ".obj"
                
            return format_lumen, format_centerline
            
        else:  
            self.data_modified.emit() # user should still be able to save data 
            return None, None 
        
        
    
    
    def save(self):
        patient_ID = self.patient_dict['patient_ID']
        base_path  = self.patient_dict['base_path']
        format_lumen, format_centerline = self.fileDialog()
        
        # export canceled 
        if format_lumen == None:
            return True
        path_lumen = os.path.join(base_path, "models","capping", patient_ID + "_lumen_capped" + format_lumen) 
        if self.capped_lumen.GetNumberOfPoints() > 0:
            if not os.path.exists(os.path.join(base_path,"models","capping")):
                os.makedirs(os.path.join(base_path,"models","capping"))
            else: 
                # clear files from old capping
                for f in os.listdir(os.path.join(base_path,"models","capping")):
                    os.remove(os.path.join(base_path,"models","capping",f))
            
            # save capped lumen
            self.writeFile(format_lumen,path_lumen,self.capped_lumen)
            
            # save holes
            for i in range(len(self.capped_holes)):
                path = os.path.join(base_path, "models","capping", patient_ID +"_cap" + str(i) + format_lumen)
                self.writeFile(format_lumen,path,self.capped_holes[i])
            
            # save capped centerline
            if self.capped_centerline.GetNumberOfPoints() > 0:
                path = os.path.join(base_path, "models","capping", patient_ID + "_centerline_capped" + format_centerline)
                self.writeFile(format_centerline,path,self.capped_centerline,True)
                
            self.new_capping.emit(format_lumen)
        
    
    def discard(self):
        self.loadPatient(self.patient_dict, self.centerline)
        
    def closeEvent(self,event):
        self.surface_view.Finalize()
        
