import os 
import shutil 
import sys
from collections import OrderedDict

import numpy as np
import nrrd
import pydicom
import nibabel as nib
from PyQt6.QtGui import QColor
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTreeWidgetItem
    )

# internal imports 
from defaults import *
from mainwindow_ui import Ui_MainWindow
from modules.SegmentationModule import SegmentationModule
from modules.CenterlineModule import CenterlineModule
from modules.MetricsModule import MetricsModule
from modules.CappingModule import CappingModule
from modules.Preprocessors import CenterlinePreprocessor

# TODO: icons?, other load formats, size tree widget
class AortaFramework(QMainWindow,Ui_MainWindow):
    # setup of UI and pipeline, Centerline Module, Segmentation Module and interactors based on the CarotisAnalyzer (code adjusted): https://git.rz.uni-jena.de/hi52cek/carotidanalyzer/-/tree/main
    def __init__(self):
        super().__init__() 
        self.setupUI(self)
        self.tree_widget_data.setExpandsOnDoubleClick(False)
        
        # state 
        self.unsaved_changes = False
        self.compute_threads_active = 0   
        self.working_dir = ""
        self.patient_data = []
        self.active_patient_dict = {'patient_ID':None}
        self.active_patient_tree_widget_item = None
        self.data = []
        self.processed_centerline = CenterlinePreprocessor()
        self.locations = []
        self.DICOM_source_dir = ""
        self.load_patient_ID = ""
        
        locale = QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates)
        QtCore.QLocale.setDefault(locale)

        # instantiate modules and add to stack
        self.segmentation_module = SegmentationModule(self)
        self.segmentation_module.ui_statusbar = self.statusbar  # pass reference to statusbar to Segemntation Module in order to acess it
        self.centerline_module = CenterlineModule(self)
        self.metrics_module = MetricsModule(self)
        self.capping_module = CappingModule(self)
        self.module_stack.addWidget(self.segmentation_module)
        self.module_stack.addWidget(self.centerline_module)
        self.module_stack.addWidget(self.metrics_module)
        self.module_stack.addWidget(self.capping_module)
        
        # assure that only one module can be active 
        self.processing_modules = [
            self.action_segmentation_module, self.action_centerline_module
        ]
        self.vis_modules = [
            self.action_metrics_module, self.action_capping_module
        ]

        
        # connect signals and slots 
        self.action_load_new_DICOM.triggered.connect(self.openDICOMDirDialog)
        self.action_load_new_nifti.triggered.connect(self.loadNewNifti)
        self.action_set_working_directory.triggered.connect(self.openWorkingDirDialog)
        self.action_delete_selected_patient.triggered.connect(self.deleteSelectedPatient)
        self.action_data_inspector.triggered[bool].connect(self.viewDataInspector)
        self.action_segmentation_module.triggered[bool].connect(self.viewSegmentationModule)
        self.action_centerline_module.triggered[bool].connect(self.viewCenterlineModule)
        self.action_metrics_module.triggered[bool].connect(self.viewMetricsModule)
        self.action_capping_module.triggered[bool].connect(self.viewCappingModule)
        self.action_discard_changes.triggered.connect(self.discardChanges)
        self.action_save_and_propagate.triggered.connect(self.saveAndPropagate)
        self.action_quit.triggered.connect(self.close)
        self.button_load_file.clicked.connect(self.loadSelectedPatient)
        self.tree_widget_data.itemDoubleClicked.connect(self.loadSelectedPatient) 
        
        ## modified, new data from modules 
        self.segmentation_module.data_modified.connect(self.changesMade)
        self.segmentation_module.new_segmentation.connect(self.newSegmentation)
        self.segmentation_module.new_models.connect(self.newModels)
        self.centerline_module.data_modified.connect(self.changesMade)
        self.centerline_module.new_centerlines.connect(self.newCenterlines)
        self.metrics_module.metrics_changed.connect(self.changesMade)
        self.metrics_module.new_metrics.connect(self.newMetrics)
        self.capping_module.data_modified.connect(self.changesMade)
        self.capping_module.new_capping.connect(self.newCapping)

        
        # restore state properties 
        settings = QtCore.QSettings()
        dir = settings.value("LastWorkingDir")
        if dir != None and os.path.exists(dir):
            self.setWorkingDir(dir)

    #################### set and load data
    def updateTree(self):
        # update tree widget, load patient
        self.setWorkingDir(self.working_dir)   
        for patient in self.patient_data:
            if patient['patient_ID'] == self.load_patient_ID:
                self.active_patient_dict = patient
                if patient["centerlines"]:
                    self.processed_centerline.reader_centerline.SetFileName("") # forces a reload
                    self.processed_centerline.reader_centerline.SetFileName(self.active_patient_dict["centerlines"])
                    self.processed_centerline.reader_centerline.Update()
                    self.processed_centerline.preprocess()       
                self.segmentation_module.loadPatient(patient)
                self.centerline_module.loadPatient(patient)
                self.metrics_module.loadPatient(patient, self.processed_centerline)
                self.capping_module.loadPatient(patient, self.processed_centerline)
                break
        
        # set as activated widget
        for i in range(self.tree_widget_data.topLevelItemCount()):
                if self.tree_widget_data.topLevelItem(i).text(0) == self.load_patient_ID:
                    self.active_patient_tree_widget_item = self.tree_widget_data.topLevelItem(i)
                    break
        self.setPatientTreeItemColor(self.active_patient_tree_widget_item, COLOR_SELECTED)
        
    
    def report_DICOM_Progress(self, progress_val, progress_msg):
        self.pbar.setValue(progress_val)
        self.pbar.setFormat(progress_msg + " (%p%)")
        

    def openDICOMDirDialog(self): 
        # set path for dcm file
        source_dir = QFileDialog.getExistingDirectory(self, "Set source directory of DICOM files")
        
        # userinput for target filename if dcm path set 
        if source_dir:
            dir_name, ok = QInputDialog.getText(self, "Set Patient Directory", "Enter name of directory for patient data:")
            # check if directory exists, if yes -> open new dialog and check again 
            if dir_name and ok:
                while (os.path.exists(os.path.join(self.working_dir, dir_name)) or
                       os.path.exists(os.path.join(self.working_dir,("case_" + dir_name)))):
                    dir_name, ok = QInputDialog.getText(self, "Set patient Directory", "Directory/Case allready exists! Please choose another name:")
                    # break if dialog canceled by user 
                    if not ok: 
                        break

                if dir_name and ok: 
                    # create directory 
                    filename = dir_name + ".nrrd"
                    path = os.path.join(self.working_dir, dir_name) 
                    nrrd_path = os.path.join(path,filename)
                    os.mkdir(path)
                    self.DICOM_source_dir = source_dir
                    self.load_patient_ID = dir_name
                    
                    # start new thread to read dicom and report progress (prevent freezing)
                    self.action_load_new_DICOM.setEnabled(False)
                    self.pbar = QProgressBar() 
                    self.pbar.setMinimum(0)
                    self.pbar.setMaximum(len(os.listdir(source_dir)) + 1)
                    self.statusbar.addWidget(self.pbar)

                    self.thread = QtCore.QThread()
                    self.worker = DICOMReaderWorker()
                    self.worker.source_dir = source_dir
                    self.worker.nrrd_path = nrrd_path
                    self.worker.moveToThread(self.thread)

                    self.worker.progress[int, str].connect(self.report_DICOM_Progress)  
                    self.worker.data_processed.connect(lambda: self.statusbar.removeWidget(self.pbar))
                    self.worker.data_processed.connect(lambda: self.statusbar.showMessage("Saving "+filename+" ..."))
                    self.worker.finished.connect(self.thread.quit)
                    self.worker.finished.connect(self.worker.deleteLater)

                    self.thread.started.connect(self.worker.run)
                    self.thread.finished.connect(self.thread.deleteLater)
                    self.thread.finished.connect(self.statusbar.clearMessage)
                    self.worker.finished.connect(self.updateTree)
                    self.thread.finished.connect(lambda: self.action_load_new_DICOM.setEnabled(True))
                    self.thread.start()

    
    def loadNewNifti(self):
        # set path for nifit file 
        filter = "NIfTI files (*.nii *.nii.gz)"
        nifti_path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "", filter)
        
        # userinput for target filename if nifti path set
        if nifti_path:
            dir_name, ok = QInputDialog.getText(self, "Set Patient Directory", "Enter name of directory for patient data:")
            # check if directory exists, if yes -> open new dialog and check again 
            if dir_name and ok:
                while os.path.exists(os.path.join(self.working_dir, dir_name)):
                    dir_name, ok = QInputDialog.getText(self, "Set patient Directory", "Directory/Case allready exists! Please choose another name:")
                    # break if dialog canceled by user 
                    if not ok: 
                        break

                if dir_name and ok: 
                    self.load_patient_ID = dir_name
                   # create directory 
                    path = os.path.join(self.working_dir, self.load_patient_ID) 
                    os.mkdir(path)
                    # load image and get metadata for header
                    nifti_img = nib.load(nifti_path)
                    nifti_header = nifti_img.header
                    nifti_data = nifti_img.get_fdata().astype(int)  # np array of image data 
                    dim_x, dim_y, dim_z = nifti_data.shape
                    affine = nifti_img.affine
                    s_x, s_y, s_z = np.diag(affine)[:3]  
                    ox,oy,oz = nifti_header["qoffset_x"],nifti_header["qoffset_y"],nifti_header["qoffset_z"]
                    
                    # save as nrrd
                    filename = self.load_patient_ID + ".nrrd"
                    nrrd_path = os.path.join(path,filename)
                    header = OrderedDict()
                    header['dimension'] = 3
                    header['space'] = 'left-posterior-superior'
                    header['sizes'] =  str(dim_x) + ' ' + str(dim_y) + ' ' + str(dim_z) 
                    header['space directions'] = [[s_x, 0.0, 0.0], [0.0, s_y, 0.0], [0.0, 0.0, s_z]]
                    header['kinds'] = ['domain', 'domain', 'domain']
                    header['endian'] = 'little'
                    header['encoding'] = 'gzip'
                    header['space origin'] = [ox,oy,oz]  
                    self.write_nrrd(nrrd_path, nifti_data, header,filename)

                    
    def write_nrrd(self, path, array, header, filename):
        # start new thread to save data in nrrd file (prevent freezing)
        self.button_load_file.setEnabled(False)
        self.statusbar.showMessage("Saving "+filename+" ...")
        
        self.thread = QtCore.QThread()
        self.worker = NrrdWriterWorker()
        self.worker.path = path 
        self.worker.array = array
        self.worker.header = header
        self.worker.moveToThread(self.thread)
        
        self.worker.finished.connect(self.updateTree)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        
        self.thread.started.connect(self.worker.run) 
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.statusbar.clearMessage)
        self.thread.finished.connect(lambda: self.button_load_file.setEnabled(True))
        self.thread.start()
    
    
    def setWorkingDir(self, dir):
        if len(dir) <= 0:
            return
        self.working_dir = dir
        self.patient_data = []
        self.tree_widget_data.clear()
        self.active_patient_tree_widget_item = None
    
        for patient_folder in os.listdir(dir):
            base_path = os.path.join(dir,patient_folder)
            patient_dict = {}
            # dirs should be named letter + number e.g. R12, if pathology has to be specified add whitespace and pathology in () e.g. R14 (AAA)
            if "(" in patient_folder:
                patient_dict['patient_ID'] = patient_folder.split(" ")[0]
            else:    
                patient_dict['patient_ID'] = patient_folder
            patient_dict['base_path'] = base_path
        
            def add_if_exists(dict_key, file_tail):
                if dict_key in ["model","centerlines"]:  
                    path = os.path.join(dir,patient_folder, "models", patient_dict['patient_ID'] + file_tail)
                elif dict_key == "capping":
                    path_stl = os.path.join(dir,patient_folder, "models", "capping", patient_dict['patient_ID'] + file_tail[0])
                    path_obj = os.path.join(dir,patient_folder, "models", "capping", patient_dict['patient_ID'] + file_tail[1])
                else:
                    path = os.path.join(dir,patient_folder, patient_dict['patient_ID'] + file_tail)
                # catch different file formats
                if dict_key == "capping":
                    if os.path.exists(path_stl):
                        path = path_stl
                    else:
                        path = path_obj
                        
                if os.path.exists(path):
                    patient_dict[dict_key] = path
                else:
                    patient_dict[dict_key] = False

            # Fills the patient dict with all existing filepaths.
            # Non-existing filepaths are marked with 'False'.
            # First param is the dict key used to retrieve the entry.
            # Second param is the file tail after patientID.
            add_if_exists("volume", ".nrrd") 
            add_if_exists("seg", ".seg.nrrd")
            add_if_exists("model", ".stl")
            add_if_exists("centerlines", ".vtp")
            add_if_exists("metrics", "_metrics.csv")
            add_if_exists("capping", ["_lumen_capped.stl","_lumen_capped.obj"])  # TODO: check if right amount of caps?
            self.patient_data.append(patient_dict)

            entry_volume = ["Volume", ""]
            entry_volume[1] = SYM_YES if patient_dict["volume"] else SYM_NO

            entry_seg = ["Segmentation", ""]
            entry_seg[1] = SYM_YES if patient_dict["seg"] else SYM_NO

            entry_model = ["Model", ""]
            entry_model[1] = SYM_YES if patient_dict["model"] else SYM_NO
            
            entry_centerlines = ["Centerlines", ""]
            entry_centerlines[1] = SYM_YES if patient_dict["centerlines"] else SYM_NO
            
            entry_metrics = ["Metrics", ""]
            entry_metrics[1] = SYM_YES if patient_dict["metrics"] else SYM_NO
            
            entry_capping = ["Capping", ""]
            entry_capping[1] = SYM_YES if patient_dict["capping"] else SYM_NO
            
            # show content of data in data inspector
            entry_patient = QTreeWidgetItem([patient_folder, ""])
            entry_patient.addChild(QTreeWidgetItem(entry_volume))
            entry_patient.addChild(QTreeWidgetItem(entry_seg))
            entry_patient.addChild(QTreeWidgetItem(entry_model))
            entry_patient.addChild(QTreeWidgetItem(entry_centerlines))
            entry_patient.addChild(QTreeWidgetItem(entry_metrics))
            entry_patient.addChild(QTreeWidgetItem(entry_capping))
            self.tree_widget_data.addTopLevelItem(entry_patient)
            entry_patient.setExpanded(True)
        self.tree_widget_data.resizeColumnToContents(0)
        self.tree_widget_data.resizeColumnToContents(1)
        
    
    def openWorkingDirDialog(self):
        # direct to working directory via file explorer 
        dir = QFileDialog.getExistingDirectory(self, "Set Working Directory")
        if len(dir) > 0:
            self.setWorkingDir(dir)
    
    
    def setPatientTreeItemColor(self, item, color):
        if item is None:
            return
        c = QColor(color[0], color[1], color[2])
        for i in range(3):  
            item.setBackground(i, c)
        for i in range(item.childCount()):
            for j in range(3):
                item.child(i).setBackground(j, c)

    def loadSelectedPatient(self): 
        # make sure that no unsave changes that would get lost 
        if self.unsaved_changes:
            return

        # get top parent item
        selected = self.tree_widget_data.currentItem()
        if selected == None:
            return
        
        while selected.parent() != None:
            selected = selected.parent()

        # set colors of last and current item
        self.setPatientTreeItemColor(self.active_patient_tree_widget_item, COLOR_UNSELECTED)
        self.setPatientTreeItemColor(selected, COLOR_SELECTED)

        # save selected item, load new patient
        self.active_patient_tree_widget_item = selected
        patient_ID = selected.text(0)
        if "(" in patient_ID: 
            patient_ID = patient_ID.split(" ")[0]
        for patient in self.patient_data:
            if (patient['patient_ID'] == patient_ID):
                # update patient in all modules
                self.active_patient_dict = patient
                self.__updatePatientInModules()
                break

    
    def deleteSelectedPatient(self):
        # get selected top parent item
        selected = self.tree_widget_data.currentItem()
        if selected == None:
            return

        while selected.parent() != None: 
            selected = selected.parent()

        # delete patient dierectory, reset modules and remove patient from tree widget if user confirms patient
        patient = selected.text(0)
        delete = QMessageBox.question(self,
                                      "Delete patient",
                                      "Do you want to delete the data of " + patient + "?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        if delete == QMessageBox.StandardButton.Yes:
            patient_idx = self.tree_widget_data.indexOfTopLevelItem(selected)
            shutil.rmtree(os.path.join(self.working_dir, patient))
            del self.patient_data[patient_idx]
            if patient == self.active_patient_dict['patient_ID']:
                self.active_patient_dict = dict.fromkeys(self.active_patient_dict, False)
                self.__updatePatientInModules()
                self.active_patient_dict = {}
                if self.unsaved_changes == True:
                    self.discardChanges()
            self.tree_widget_data.takeTopLevelItem(patient_idx)


    def __updatePatientInModules(self):
        if self.active_patient_dict["centerlines"]:
            self.processed_centerline.reader_centerline.SetFileName("") # forces a reload
            self.processed_centerline.reader_centerline.SetFileName(self.active_patient_dict["centerlines"])
            self.processed_centerline.reader_centerline.Update()
            self.processed_centerline.preprocess()       
        # load patient in other modules, give active patient dict 
        self.segmentation_module.loadPatient(self.active_patient_dict)
        self.centerline_module.loadPatient(self.active_patient_dict)
        self.metrics_module.loadPatient(self.active_patient_dict, self.processed_centerline)
        self.capping_module.loadPatient(self.active_patient_dict, self.processed_centerline)
    
    
    #################### manage stack 
    def viewDataInspector(self, on:bool):
        self.dock_data_inspector.setVisible(on)
        """
        # carotidanalyzer
        if on:
            self.dock_data_inspector.show()
        else:
            self.dock_data_inspector.close()"""
            
    def uncheckInactiveModules(self, active_module):
        for m in self.processing_modules + self.vis_modules: 
            if not m == active_module:
                m.setChecked(False)
        if active_module in self.processing_modules:
            self.action_save_and_propagate.setText("Save and Propagate")
        else: 
            self.action_save_and_propagate.setText("Export Data")


    def viewSegmentationModule(self, on:bool):
        if on:
            self.uncheckInactiveModules(self.action_segmentation_module)
            self.module_stack.setCurrentWidget(self.segmentation_module)
        else:
            self.module_stack.setCurrentWidget(self.empty_module)


    
    def viewCenterlineModule(self, on:bool):
        if on:
            self.uncheckInactiveModules(self.action_centerline_module)
            self.module_stack.setCurrentWidget(self.centerline_module)
        else:
            self.module_stack.setCurrentWidget(self.empty_module)
    
    
    def viewMetricsModule(self, on:bool):
        if on:
            self.uncheckInactiveModules(self.action_metrics_module)
            self.module_stack.setCurrentWidget(self.metrics_module)
        else:
            self.module_stack.setCurrentWidget(self.empty_module)
            
            
    def viewCappingModule(self, on:bool):
        if on:
            self.uncheckInactiveModules(self.action_capping_module)
            self.module_stack.setCurrentWidget(self.capping_module)
        else:
            self.module_stack.setCurrentWidget(self.empty_module)
    
    
    ##################### internal data pipeline
    def setModulesClickable(self, state:bool):
        self.action_segmentation_module.setEnabled(state)
        self.action_centerline_module.setEnabled(state)
        self.action_metrics_module.setEnabled(state)
        self.action_capping_module.setEnabled(state)
    
    
    def changesMade(self):
        self.unsaved_changes = True
        self.setModulesClickable(False)
        self.button_load_file.setEnabled(False)
        self.action_discard_changes.setEnabled(True)
        self.action_save_and_propagate.setEnabled(True)


    def discardChanges(self):
        self.module_stack.currentWidget().discard()
        self.action_discard_changes.setEnabled(False)
        self.action_save_and_propagate.setEnabled(False)
        self.button_load_file.setEnabled(True)
        self.unsaved_changes = False
        self.setModulesClickable(True)

    
    def saveAndPropagate(self):
        # calls save on the current widget
        # propagation must be called through widget signals of type "newX"
        cancel = self.module_stack.currentWidget().save()
        if cancel: 
            return
        self.action_discard_changes.setEnabled(False)
        self.action_save_and_propagate.setEnabled(False)
        self.button_load_file.setEnabled(True)
        self.unsaved_changes = False
        self.setModulesClickable(True)

    ######## pipeline for modules
    # update tree, propagate data if necessary
    def newSegmentation(self):  
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path = os.path.join(base_path, patient_ID + ".seg.nrrd")
        seg_item = self.active_patient_tree_widget_item.child(1)
        if os.path.exists(path):
            self.active_patient_dict['seg'] = path
            seg_item.setText(1, SYM_YES)
        
        
    def newModels(self):  
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_lumen = os.path.join(base_path, "models", patient_ID + ".stl")
        lumen_item = self.active_patient_tree_widget_item.child(2)
        if os.path.exists(path_lumen):
            self.active_patient_dict['model'] = path_lumen
            lumen_item.setText(1, SYM_YES)
    
        # propagate
        self.centerline_module.loadPatient(self.active_patient_dict)
        self.metrics_module.loadPatient(self.active_patient_dict,self.processed_centerline)
        self.capping_module.loadPatient(self.active_patient_dict,self.processed_centerline)
        

    def newCenterlines(self):  # from old 
        # at save and propagate (if new centerline): went into this method but not in load Patient?!
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_centerlines = os.path.join(base_path, "models", patient_ID + ".vtp")
        centerlines_item = self.active_patient_tree_widget_item.child(3)
        if os.path.exists(path_centerlines):
            self.active_patient_dict['centerlines'] = path_centerlines
            centerlines_item.setText(1, SYM_YES)

        # propagate
        self.processed_centerline.reader_centerline.SetFileName("") # forces a reload
        self.processed_centerline.reader_centerline.SetFileName(path_centerlines)
        self.processed_centerline.reader_centerline.Update()
        self.processed_centerline.preprocess() 
        self.metrics_module.loadPatient(self.active_patient_dict,self.processed_centerline)
        self.capping_module.loadPatient(self.active_patient_dict,self.processed_centerline)
    
    
    def newMetrics(self):
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_metrics = os.path.join(base_path, patient_ID + "_metrics.csv")
        metrics_item = self.active_patient_tree_widget_item.child(4)
        if os.path.exists(path_metrics):
            self.active_patient_dict['metrics'] = path_metrics
            metrics_item.setText(1, SYM_YES)
    
    
    def newCapping(self,format_lumen):
        patient_ID = self.active_patient_dict['patient_ID']
        base_path  = self.active_patient_dict['base_path']
        path_capping = os.path.join(base_path, "models","capping", patient_ID + "_lumen_capped"+format_lumen) 
        capping_item = self.active_patient_tree_widget_item.child(5)
        if os.path.exists(path_capping):
            self.active_patient_dict["capping"] = path_capping
            capping_item.setText(1, SYM_YES)

    ######## ensure save exit 
    def okToClose(self):
        if self.unsaved_changes:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Unsaved Changes")
            dlg.setText("Close application? Changes will be lost.")
            dlg.setStandardButtons(QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel)
            button = dlg.exec()
            if button == QMessageBox.StandardButton.Cancel:
                return False

        if self.compute_threads_active > 0:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Threads Running")
            dlg.setText("Close application? Active computations will be lost.")
            dlg.setStandardButtons(QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel)
            button = dlg.exec()
            if button == QMessageBox.StandardButton.Cancel:
                return False

        return True
    
    
    def closeEvent(self, event):
        # save exit of tool 
        if self.okToClose():
            settings = QtCore.QSettings()
            
            # save last opened working dir
            dirname = QtCore.QVariant(self.working_dir) #if len(self.working_dir) is not 0 else QVariant()
            settings.setValue("LastWorkingDir", dirname)
            
            # save main window position and size
            settings.setValue("MainWindow/Geometry", QtCore.QVariant(self.saveGeometry()))
            
            # call Finalize() for all vtk interactors
            self.segmentation_module.close()
            self.centerline_module.close()
            self.metrics_module.close()
            self.capping_module.close()
            super(AortaFramework, self).closeEvent(event)
        else:
            event.ignore()


class DICOMReaderWorker(QtCore.QObject):  
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int, str)
    data_processed = QtCore.pyqtSignal()
    source_dir = None 
    nrrd_path = None
   

    def run(self):
        data = []
        locations = []
        path = os.listdir(self.source_dir)
        for idx, file in enumerate(path):
            # read in each dcm and save pixel data, emit progress and data when finished 
            self.progress.emit(idx, "Loading " + file)
            ds = pydicom.dcmread(os.path.join(self.source_dir, file))
            hu = pydicom.pixel_data_handlers.util.apply_modality_lut(ds.pixel_array, ds)
            locations.append(ds[0x0020, 0x1041].value) # slice location
            data.append(hu)

        # sort slices if required
        self.progress.emit(idx + 1, "Sorting slices...")
        if not (all(locations[i] <= locations[i + 1] for i in range(len(locations)-1))):
            data = [x for _, x in sorted(zip(locations, data))] 
        data_array = np.transpose(np.array(data, dtype=np.int16))
        self.data_processed.emit()
        # save data 
        self.write_nrrd(data_array)
        
        self.finished.emit()
    
    def write_nrrd(self,data_array):
        # get metadata for header/vtkImage
        dicomdata = pydicom.dcmread(os.path.join(self.source_dir, os.listdir(self.source_dir)[0]))
        dim_x, dim_y, dim_z = data_array.shape
        s_z = float(dicomdata[0x0018, 0x0088].value)  # spacing between slices 
        s_x_y = dicomdata[0x0028, 0x0030].value  # pixel spacing 
        pos = dicomdata[0x0020, 0x0032].value  # image position
        
        header = OrderedDict()
        header['dimension'] = 3
        header['space'] = 'left-posterior-superior'
        header['sizes'] =  str(dim_x) + ' ' + str(dim_y) + ' ' + str(dim_z) 
        header['space directions'] = [[s_x_y[0], 0.0, 0.0], [0.0, s_x_y[1], 0.0], [0.0, 0.0, s_z]]
        header['kinds'] = ['domain', 'domain', 'domain']
        header['endian'] = 'little'
        header['encoding'] = 'gzip'
        header['space origin'] = pos
        nrrd.write(self.nrrd_path, data_array, header)
        


class NrrdWriterWorker(QtCore.QObject):  
    finished = QtCore.pyqtSignal()
    path = None
    array = None 
    header = None 
   
    def run(self): 
        nrrd.write(self.path, self.array, self.header)
        self.finished.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("VisGroup Uni Jena")
    #app.setOrganizationDomain("vis.uni-jena.de")
    app.setApplicationName("AortaFramework")
    app.setStyle("Fusion")  
    win = AortaFramework()
    win.show()
    sys.exit(app.exec())
