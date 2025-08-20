import numpy as np 

import vtk 
from vtk.util.numpy_support import vtk_to_numpy

class CenterlinePreprocessor():
    """
    Preprocessing of centerline to access points, radii and relations between branches.
    """
    def __init__(self):
        self.patient_dict = None 
        self.c_radii_lists = []     # processed centerline radii
        self.c_pos_lists = []       # processed centerline positions
        self.c_arc_lists = []       # processed centerline arc length (cumulated)
        self.c_parent_indices = []  # processed centerline bifurcation points
        self.c_child_branches = {}  
        self.min_branch_len = 20 # minimal length of a branch in mm
        self.branch_cutoff = 1  # length to be cut from branch ends in mm
        
        self.reader_centerline = vtk.vtkXMLPolyDataReader()
        self.picker = vtk.vtkPropPicker() 
    
    def preprocess(self):
        # adapted version of stenosis classifier method
        # lists for each line in centerlines
        # lines are ordered source->outlet
        self.c_pos_lists = []       # 3xn numpy arrays with point positions
        self.c_arc_lists = []       # 1xn numpy arrays with arc length along centerline (accumulated)
        self.c_radii_lists = []     # 1xn numpy arrays with maximal inscribed sphere radius
        self.c_parent_indices = []  # tuple per list: (parent idx, branch point idx), id in list dependent on child branch number
        self.c_child_branches = {}  # list per parent branch: all child branches 

        # iterate all (global) lines
        # each line is a vtkIdList containing point ids in the right order
        centerlines = self.reader_centerline.GetOutput()	
        radii_flat = centerlines.GetPointData().GetArray('MaximumInscribedSphereRadius')
        l = centerlines.GetLines()
        l.InitTraversal()
        for i in range(l.GetNumberOfCells()):
            pointIds = vtk.vtkIdList()
            l.GetNextCell(pointIds)
            # retrieve position data
            points = vtk.vtkPoints()
            centerlines.GetPoints().GetPoints(pointIds, points)
            p = vtk_to_numpy(points.GetData())

            # calculate arc len
            arc = p - np.roll(p, 1, axis=0)
            arc = np.sqrt((arc*arc).sum(axis=1))
            arc[0] = 0
            arc = np.cumsum(arc)

            # retrieve radius data
            radii = vtk.vtkDoubleArray()
            radii.SetNumberOfTuples(pointIds.GetNumberOfIds())
            radii_flat.GetTuples(pointIds, radii)
            r = vtk_to_numpy(radii)

            # add to centerlines
            self.c_pos_lists.append(p)
            self.c_arc_lists.append(arc)
            self.c_radii_lists.append(r)
            self.c_parent_indices.append((i,0)) # points to own origin

        # cleanup branch overlaps
        # (otherwise each line starts at the inlet)
        for i in range(0, len(self.c_pos_lists)):
            for j in range(i+1, len(self.c_pos_lists)):
                len0 = self.c_pos_lists[i].shape[0]
                len1 = self.c_pos_lists[j].shape[0]
                if len0 < len1:
                    overlap_mask = np.not_equal(self.c_pos_lists[i], self.c_pos_lists[j][:len0])
                else:
                    overlap_mask = np.not_equal(self.c_pos_lists[i][:len1], self.c_pos_lists[j])
                overlap_mask = np.all(overlap_mask, axis=1) # AND over tuples
                split_index = np.searchsorted(overlap_mask, True) # first position where lines diverge

                if split_index <= 0:
                    continue # no new parent was found

                # save parent and position
                self.c_parent_indices[j] = (i,split_index)
                
                # clip line to remove overlaps
                self.c_pos_lists[j] = self.c_pos_lists[j][split_index:]
                self.c_arc_lists[j] = self.c_arc_lists[j][split_index:]
                self.c_radii_lists[j] = self.c_radii_lists[j][split_index:]

        #TODO: sometimes child = own parent or b1 parent of b2 + b2 parent of b1 --> why??
        # remove branches below the minimum length
        for i in range(len(self.c_arc_lists)-1, -1, -1):
            if self.c_arc_lists[i][-1] - self.c_arc_lists[i][0] < self.min_branch_len:
                del self.c_pos_lists[i]
                del self.c_arc_lists[i]
                del self.c_radii_lists[i]
                del self.c_parent_indices[i]
        
        # save direct child branches
        for i in range(len(self.c_parent_indices)):
            parent = self.c_parent_indices[i][0]
            if parent in self.c_child_branches:
                self.c_child_branches[parent].append(i)
            else:
                self.c_child_branches[parent] = [i]

        # clip branch ends 
        for i in range(len(self.c_arc_lists)):
            #start = self.c_arc_lists[i][0] + self.branch_cutoff
            start = 0
            end = self.c_arc_lists[i][-1] - self.branch_cutoff
            clip_ids = np.searchsorted(self.c_arc_lists[i], [start, end])
            self.c_pos_lists[i] = self.c_pos_lists[i][clip_ids[0]:clip_ids[1]]
            self.c_arc_lists[i] = self.c_arc_lists[i][clip_ids[0]:clip_ids[1]]
            self.c_radii_lists[i] = self.c_radii_lists[i][clip_ids[0]:clip_ids[1]]
    
    
    def getClosestCenterlinePoint(self,x,y,renderer):
        pick = self.picker.Pick(x,y,0,renderer) 
        position = np.array(self.picker.GetPickPosition())
        #if click on surface: determine closest point on centerline and set diameter
        if pick: 
            min_d_id = None
            min_d = float('inf')
            for i in range(len(self.c_pos_lists)):
                d = ((np.array(self.c_pos_lists[i])-np.array(position))**2).sum(axis=1) 
                min_d_branch = np.min(d)
                if min_d_branch < min_d:
                    min_d = min_d_branch
                    min_d_id = [i,d.argmin()]   
        else:
            min_d_id = None 
        
        return min_d_id
