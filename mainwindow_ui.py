import sys 

from PyQt6 import QtCore
from PyQt6.QtGui import QAction 
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QHBoxLayout, 
    QLabel,
    QMenu,
    QMenuBar,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QSizePolicy,
    QToolBar,
    QTreeWidget,
    QVBoxLayout,
    QWidget)

class Ui_MainWindow():
    def setupUI(self,MainWindow):
        # setup structure of ui main window 
        MainWindow.setWindowTitle("AortaAnalyzer")
        MainWindow.resize(1920, 1080)
        #MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.central_widget = QWidget(MainWindow)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.central_widget.setSizePolicy(sizePolicy)
        self.central_widget.setObjectName("central_widget")
        
        # top level layout 
        self.pagelayout = QVBoxLayout(self.central_widget)
        self.pagelayout.setContentsMargins(0, 0, 0, 0)
        self.pagelayout.setSpacing(0)
        self.pagelayout.setObjectName("page_layout")
        
        # stack modules to select one at a time
        self.module_stack = QStackedWidget(MainWindow)
        self.module_stack.setObjectName("module_stack")
        self.empty_module = QWidget()
        self.empty_module.setObjectName("empty_module")
        self.horizontalLayout = QHBoxLayout(self.empty_module)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.empty_module_label = QLabel(self.empty_module)
        self.empty_module_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_module_label.setObjectName("empty_module_label")
        self.empty_module_label.setText("Select a module.")
        self.horizontalLayout.addWidget(self.empty_module_label)
        self.module_stack.addWidget(self.empty_module)
        self.pagelayout.addWidget(self.module_stack)
        
        # set central widget 
        self.central_widget.setLayout(self.pagelayout)
        self.setCentralWidget(self.central_widget)

        # menubar (load data)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 31))
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile") 
        self.menuFile.setTitle("File")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuView.setTitle("View")
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)   
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # toolbars to switch modules and to save/discard
        self.toolbar_modules = QToolBar(MainWindow)
        self.toolbar_modules.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.toolbar_modules.setMovable(False)
        self.toolbar_modules.setFloatable(False)
        self.toolbar_modules.setObjectName("toolbar_modules")
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar_modules)
        # setwindowtitle?
        
        self.toolbar_save = QToolBar(MainWindow)
        self.toolbar_save.setLayoutDirection(QtCore.Qt.LayoutDirection.RightToLeft)
        self.toolbar_save.setMovable(False)
        self.toolbar_save.setFloatable(False)
        self.toolbar_save.setObjectName("toolbar_save")
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar_save)

        # docker for data inspection
        self.dock_data_inspector = QDockWidget(MainWindow)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dock_data_inspector.sizePolicy().hasHeightForWidth())
        self.dock_data_inspector.setSizePolicy(sizePolicy)
        self.dock_data_inspector.setMaximumSize(QtCore.QSize(524287, 524287))
        self.dock_data_inspector.setBaseSize(QtCore.QSize(200, 0))  # 300
        self.dock_data_inspector.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable|QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.dock_data_inspector.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea|QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock_data_inspector.setObjectName("dock_data_inspector")
        #self.dock_data_inspector.setWindowTitle("Data Inspector")
        self.data_inspector_contents = QWidget()
        self.data_inspector_contents.setObjectName("data_inspector_contents")
        self.verticalLayout = QVBoxLayout(self.data_inspector_contents)
        self.verticalLayout.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tree_widget_data = QTreeWidget(self.data_inspector_contents)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tree_widget_data.sizePolicy().hasHeightForWidth())
        self.tree_widget_data.setSizePolicy(sizePolicy)
        self.tree_widget_data.setMinimumSize(QtCore.QSize(300, 0))  # 400 
        self.tree_widget_data.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tree_widget_data.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree_widget_data.setIndentation(20)
        self.tree_widget_data.setUniformRowHeights(True)
        self.tree_widget_data.setAllColumnsShowFocus(True)
        self.tree_widget_data.setColumnCount(2)
        self.tree_widget_data.setObjectName("tree_widget_data")
        self.tree_widget_data.header().setDefaultSectionSize(90)
        self.tree_widget_data.header().setHighlightSections(True)
        self.tree_widget_data.headerItem().setText(0,"Patient ID")
        self.tree_widget_data.headerItem().setText(1,"")  # Data available
        self.verticalLayout.addWidget(self.tree_widget_data)
        self.button_load_file = QPushButton(self.data_inspector_contents)
        self.button_load_file.setObjectName("button_load_file")
        self.button_load_file.setText("Load Selected Data")
        self.verticalLayout.addWidget(self.button_load_file)
        self.dock_data_inspector.setWidget(self.data_inspector_contents)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dock_data_inspector)
        
        ## define actions 
        # menubar 
        self.action_load_new_DICOM = QAction(MainWindow)
        self.action_load_new_DICOM.setObjectName("action_load_new_DICOM")
        self.action_load_new_DICOM.setText("Load New DICOM...")
        self.action_load_new_nifti = QAction(MainWindow)
        self.action_load_new_nifti.setObjectName("action_load_new_nifti")
        self.action_load_new_nifti.setText("Load New nifti...")
        self.action_set_working_directory = QAction(MainWindow)
        self.action_set_working_directory.setObjectName("action_set_working_directory")
        self.action_set_working_directory.setText("Set Working Directory ...")
        # toolbar save
        self.action_save_and_propagate = QAction(MainWindow)
        self.action_save_and_propagate.setEnabled(False)
        self.action_save_and_propagate.setObjectName("action_save_and_propagate")
        self.action_save_and_propagate.setText("Save and Propagate")
        self.action_save_and_propagate.setShortcut("Ctrl+S")
        self.action_quit = QAction(MainWindow)
        self.action_quit.setCheckable(True)
        self.action_quit.setObjectName("action_quit")
        self.action_quit.setText("Close")
        self.action_delete_selected_patient = QAction(MainWindow)
        self.action_delete_selected_patient.setObjectName("action_delete_selected_patient")
        self.action_delete_selected_patient.setText("Delete selected data")
        self.action_delete_selected_patient.setShortcut("Ctrl+X")
        self.action_discard_changes = QAction(MainWindow)
        self.action_discard_changes.setEnabled(False)
        self.action_discard_changes.setObjectName("action_discard_changes")
        self.action_discard_changes.setText("Discard Changes")
        self.action_data_inspector = QAction(MainWindow)
        self.action_data_inspector.setObjectName("action_data_inspector")
        self.action_data_inspector = QAction(MainWindow)
        self.action_data_inspector.setCheckable(True)
        self.action_data_inspector.setChecked(True)
        self.action_data_inspector.setText("Data Inspector")
        # toolbar menu 
        self.action_segmentation_module = QAction(MainWindow)
        self.action_segmentation_module.setCheckable(True)
        self.action_segmentation_module.setObjectName("action_segmentation_module")
        self.action_segmentation_module.setText("Segmentation Module")
        self.action_centerline_module = QAction(MainWindow)
        self.action_centerline_module.setCheckable(True)
        self.action_centerline_module.setObjectName("action_centerline_module")
        self.action_centerline_module.setText("Centerline Module")
        self.action_metrics_module = QAction(MainWindow)
        self.action_metrics_module.setCheckable(True)
        self.action_metrics_module.setObjectName("action_metrics_module")
        self.action_metrics_module.setText("Metrics Module")
        self.action_capping_module = QAction(MainWindow)
        self.action_capping_module.setCheckable(True)
        self.action_capping_module.setObjectName("action_capping_module")
        self.action_capping_module.setText("Capping Module")

        # connect actions
        self.menuFile.addAction(self.action_load_new_DICOM)  
        self.menuFile.addAction(self.action_load_new_nifti)  
        self.menuFile.addAction(self.action_set_working_directory)
        self.menuFile.addAction(self.action_delete_selected_patient)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_save_and_propagate)
        self.menuFile.addAction(self.action_discard_changes)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_quit)
        self.menuView.addAction(self.action_data_inspector)
        self.menuView.addSeparator()
        self.menuView.addAction(self.action_segmentation_module)
        self.menuView.addAction(self.action_centerline_module)
        self.menuView.addAction(self.action_metrics_module)
        self.menuView.addAction(self.action_capping_module)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.toolbar_modules.addAction(self.action_data_inspector)
        self.toolbar_modules.addSeparator()
        self.toolbar_modules.addAction(self.action_segmentation_module)
        self.toolbar_modules.addAction(self.action_centerline_module)
        self.toolbar_modules.addSeparator()
        self.toolbar_modules.addAction(self.action_metrics_module)
        self.toolbar_modules.addAction(self.action_capping_module)
        self.toolbar_modules.addSeparator()
        self.toolbar_save.addAction(self.action_save_and_propagate)
        self.toolbar_save.addAction(self.action_discard_changes)

