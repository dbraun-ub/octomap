from __future__ import print_function

from multiprocessing import Process, Pool
from functools import partial

import numpy as np
import open3d
import time
from octomap_utils import constructRay
import torch
# import octomap_utils

# From: https://code.google.com/p/pynastran/source/browse/trunk/pyNastran/general/octree.py?r=949
#       http://code.activestate.com/recipes/498121-python-octree-implementation/

# UPDATED:
# Is now more like a true octree (ie: partitions space containing objects)

# Important Points to remember:
# The OctNode positions do not correspond to any object position
# rather they are seperate containers which may contain objects
# or other nodes.

# An OctNode which which holds less objects than MAX_OBJECTS_PER_CUBE
# is a LeafNode; it has no branches, but holds a list of objects contained within
# its boundaries. The list of objects is held in the leafNode's 'data' property

# If more objects are added to an OctNode, taking the object count over MAX_OBJECTS_PER_CUBE
# Then the cube has to subdivide itself, and arrange its objects in the new child nodes.
# The new octNode itself contains no objects, but its children should.

'''
Octomap implementation
'''

class OctNode(object):
    """
    New Octnode Class, can be appended to as well i think
    """
    def __init__(self, position, size, depth, data):
        """
        OctNode Cubes have a position and size
        position is related to, but not the same as the objects the node contains.

        Branches (or children) follow a predictable pattern to make accesses simple.
        Here, - means less than 'origin' in that dimension, + means greater than.
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
        """
        self.position = position
        self.size = size
        self.depth = depth

        ## All OctNodes will be leaf nodes at first
        ## Then subdivided later as more objects get added
        self.isLeafNode = True

        ## store our object, typically this will be one, but maybe more
        self.data = data

        ## might as well give it some emtpy branches while we are here.
        self.branches = [None, None, None, None, None, None, None, None]

        half = size / 2

        ## The cube's bounding coordinates
        # self.lower = (position[0] - half, position[1] - half, position[2] - half)
        # self.upper = (position[0] + half, position[1] + half, position[2] + half)

    def __str__(self):
        data_str = u", ".join((str(x) for x in self.data))
        return u"position: {0}, size: {1}, depth: {2} leaf: {3}, data: {4}".format(
            self.position, self.size, self.depth, self.isLeafNode, data_str
        )

class Point(object):
    def __init__(self, position, size=None, depth=None, occupancy=None, color=None):
        self.position = position
        self.size = size
        self.depth = depth
        self.occupancy = occupancy
        self.color = color

    def __str__(self):
        return u"[position: {0}, size: {1}, depth: {2}, occupancy: {3}, color: {4}]".format(
            self.position, self.size, self.depth, self.occupancy, self.color
        )

class Octomap(object):
    """
    The octree itself, which is capable of adding and searching for nodes.
    """
    def __init__(self, worldSize, origin, limit_nodes=1, limit_depth=10, depthMap=None, intrinsic=None, image=None, rayCast=True):
        """
        Init the world bounding root cube
        all world geometry is inside this
        it will first be created as a leaf node (ie, without branches)
        this is because it has no objects, which is less than MAX_OBJECTS_PER_CUBE
        if we insert more objects into it than MAX_OBJECTS_PER_CUBE, then it will subdivide itself.

        """

        self.root = OctNode(origin, worldSize, 0, [])
        self.worldSize = worldSize
        self.origin = origin
        self.device = worldSize.device
        # if limit_nodes = 0, there is no limit
        self.limit_nodes = limit_nodes
        # if limit_depth = 0, there is no limit
        self.limit_depth = limit_depth

        self.m = torch.tensor([[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
             [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]], dtype=torch.float32).to(self.device)

        if depthMap is not None:
            if intrinsic is not None:
                self.insertFromDepthMap(depthMap, intrinsic, image=image, rayCast=rayCast)
            else:
                print("intrinsic parameters are missing.")

        self.logodds_max = torch.tensor(3.5, dtype=torch.float32).to(self.device)
        self.logodds_min = torch.tensor(-3.5, dtype=torch.float32).to(self.device)
        self.free_prob = torch.exp(self.logodds_min) / (1 + torch.exp(self.logodds_min))

        self.time_insertFromPtCloud = 0
        self.time___insertPointCloud = 0
        self.time___sortByBranches = 0
        self.start_time = 0
        self.call_time = 0

    @staticmethod
    def CreateNode(position, size, objects):
        """This creates the actual OctNode itself."""
        return OctNode(position, size, objects)

    def insertNode(self, position, objData=None):
        """
        Add the given object to the octree if possible

        Parameters
        ----------
        position : array_like with 3 elements
            The spatial location for the object
        objData : optional
            The data to store at this position. By default stores the position.

            If the object does not have a position attribute, the object
            itself is assumed to be the position.

        Returns
        -------
        node : OctNode or None
            The node in which the data is stored or None if outside the
            octree's boundary volume.

        """
        if np:
            if np.any(position < self.root.lower):
                return None
            if np.any(position > self.root.upper):
                return None
        else:
            if position < self.root.lower:
                return None
            if position > self.root.upper:
                return None

        if objData is None:
            objData = position

        return self.__insertNode(self.root, self.root.size, self.root, position, objData)

    def __insertNode(self, root, size, parent, position, objData):
        """Private version of insertNode() that is called recursively"""
        if root is None:
            # we're inserting a single object, so if we reach an empty node, insert it here
            # Our new node will be a leaf with one object, our object
            # More may be added later, or the node maybe subdivided if too many are added
            # Find the Real Geometric centre point of our new node:
            # Found from the position of the parent node supplied in the arguments
            pos = parent.position

            ## offset is halfway across the size allocated for this node
            offset = size / 2

            ## find out which direction we're heading in
            branch = self.__findBranch(parent, position)

            ## new center = parent position + (branch direction * offset)
            newCenter = (0, 0, 0)

            if branch == 0:
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] - offset )
            elif branch == 1:
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] + offset )
            elif branch == 2:
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] - offset )
            elif branch == 3:
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] + offset )
            elif branch == 4:
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] - offset )
            elif branch == 5:
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] + offset )
            elif branch == 6:
                newCenter = (pos[0] + offset, pos[1] + offset, pos[2] - offset )
            elif branch == 7:
                newCenter = (pos[0] + offset, pos[1] + offset, pos[2] + offset )

            # Now we know the centre point of the new node
            # we already know the size as supplied by the parent node
            # So create a new node at this position in the tree
            # print "Adding Node of size: " + str(size / 2) + " at " + str(newCenter)
            return OctNode(newCenter, size, parent.depth + 1, [objData])

        #else: are we not at our position, but not at a leaf node either
        elif (
            not root.isLeafNode
            and
            (
                (np and np.any(root.position != position))
                or
                (root.position != position)
            )
        ):

            # we're in an octNode still, we need to traverse further
            branch = self.__findBranch(root, position)
            # Find the new scale we working with
            newSize = root.size / 2
            # Perform the same operation on the appropriate branch recursively
            root.branches[branch] = self.__insertNode(root.branches[branch], newSize, root, position, objData)

        # else, is this node a leaf node with objects already in it?
        elif root.isLeafNode:
            # We've reached a leaf node. This has no branches yet, but does hold
            # some objects, at the moment, this has to be less objects than MAX_OBJECTS_PER_CUBE
            # otherwise this would not be a leafNode (elementary my dear watson).
            # if we add the node to this branch will we be over the limit?
            if (
                # (self.limit_nodes and len(root.data) < self.limit)
                # or
                # (not self.limit_nodes and root.depth >= self.limit)

                (not self.limit_nodes) # if no limit
                or
                (len(root.data) < self.limit_nodes) # if under the limit
                or
                (root.depth >= self.limit_depth) # if we reached the depth limit
            ):
                # No? then Add to the Node's list of objects and we're done
                root.data.append(objData)
                #return root
            else:
                # Adding this object to this leaf takes us over the limit
                # So we have to subdivide the leaf and redistribute the objects
                # on the new children.
                # Add the new object to pre-existing list
                root.data.append(objData)
                # copy the list
                objList = root.data
                # Clear this node's data
                root.data = None
                # It is not a leaf node anymore
                root.isLeafNode = False
                # Calculate the size of the new children
                newSize = root.size / 2
                # distribute the objects on the new tree
                # print "Subdividing Node sized at: " + str(root.size) + " at " + str(root.position)
                for ob in objList:
                    # Use the position attribute of the object if possible
                    if hasattr(ob, "position"):
                        pos = ob.position
                    else:
                        pos = ob
                    branch = self.__findBranch(root, pos)
                    root.branches[branch] = self.__insertNode(root.branches[branch], newSize, root, pos, ob)
        return root

    def findPosition(self, position):
        """
        Basic lookup that finds the leaf node containing the specified position
        Returns the child objects of the leaf, or None if the leaf is empty or none
        Also return the node size
        """
        if np:
            if np.any(position < self.root.lower):
                return None
            if np.any(position > self.root.upper):
                return None
        else:
            if position < self.root.lower:
                return None
            if position > self.root.upper:
                return None
        return self.__findPosition(self.root, position)

    @staticmethod
    def __findPosition(node, position, count=0, branch=0):
        """Private version of findPosition """
        if node.isLeafNode:
            #print("The position is", position, " data is", node.data)
            return node
        branch = Octomap.__findBranch(node, position)
        child = node.branches[branch]
        if child is None:
            return None
        return Octomap.__findPosition(child, position, count + 1, branch)

    @staticmethod
    def __findBranch(root, position):
        """
        helper function
        returns an index corresponding to a branch
        pointing in the direction we want to go
        """
        index = 0
        if (position[0] >= root.position[0]):
            index |= 4
        if (position[1] >= root.position[1]):
            index |= 2
        if (position[2] >= root.position[2]):
            index |= 1
        return index

    def iterateDepthFirst(self):
        """Iterate through the octree depth-first"""
        gen = self.__iterateDepthFirst(self.root)
        for n in gen:
            yield n

    @staticmethod
    def __iterateDepthFirst(root):
        """Private (static) version of iterateDepthFirst"""

        for branch in root.branches:
            if branch is None:
                continue
            for n in Octomap.__iterateDepthFirst(branch):
                yield n
            if branch.isLeafNode:
                yield branch

    # No loop, don't manipulate array of Point object
    def __insertPointCloud(self, pointArray, branch, parent, branchPosition, colorArray):
        if self.call_time != 0:
            self.time___sortByBranches += time.time() - self.call_time
        if self.start_time != 0:
            self.time___insertPointCloud += time.time() - self.start_time
        self.start_time = time.time()
        # if len(pointArray) == 0:
        if pointArray.shape[0] == 0:
            branch = None
        # elif (len(pointArray) == 1) or (self.limit_depth > 0 and parent.depth + 1 >= self.limit_depth):
        elif parent.depth + 1 >= self.limit_depth:
            # branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, {"nodeCenter": branchPosition, "position": torch.mean(pointArray,0), "occupancy":torch.tensor([1], dtype=torch.float32), "color":colorArray[0]})
            l = torch.min(self.logodds_max, torch.sum(pointArray[:,3]))
            branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, {"occupancy": torch.exp(l) / (1 + torch.exp(l)), "color":colorArray[0]})
        else:
            if branch is None:
                l = torch.min(self.logodds_max, torch.sum(pointArray[:,3]))
                branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, {"occupancy":torch.exp(l) / (1 + torch.exp(l)), "color":colorArray[0]})
                # {"nodeCenter": branchPosition, "position": torch.mean(pointArray,0), "occupancy":torch.tensor([1], dtype=torch.float32), "color":colorArray[0]})
            else:
                if branch.data is not None:
                    l = torch.min(self.logodds_max, torch.sum(pointArray[:,3]))
                    branch.data["occupancy"] = torch.exp(l) / (1 + torch.exp(l))
            branch.isLeafNode = False

            # dispatch points in the eight branches
            # pointsBranch, branchPosition, colorBranch = self.__sortByBranches(pointArray, branch, colorArray)
            mask, branchPosition = self.__sortByBranches(pointArray, branch, colorArray)


            for k in range(8):
                self.call_time = time.time()
                branch.branches[k] = self.__insertPointCloud(pointArray[mask[k]], branch.branches[k], branch, branchPosition[k], colorArray[mask[k]])

        return branch

    # @staticmethod
    def __sortByBranches(self, pointArray, branch, colorArray):
        # start_time = time.time()
        # dispatch points in the eight branches
        # branch: 0 1 2 3 4 5 6 7
        # x:      - - - - + + + +
        # y:      - - + + - - + +
        # z:      - + - + - + - +
        offset = branch.size / 4
        offset = offset.to(self.device)
        newCenter = branch.position + offset * self.m
        # print(f"newCenter.shape: {newCenter.shape}")
        newCenter = newCenter.reshape(8,1,3).to(self.device)
        isInsideBranch = (pointArray[:,:3] <= newCenter + offset) & (pointArray[:,:3] >= newCenter - offset)
        mask = torch.all(isInsideBranch,2)

        return mask, newCenter

    @staticmethod
    def __sortByBranches_noColor(pointArray, branch):
        # dispatch points in the eight branches
        # branch: 0 1 2 3 4 5 6 7
        # x:      - - - - + + + +
        # y:      - - + + - - + +
        # z:      - + - + - + - +
        m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
             [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
        pointsBranch = [[], [], [], [], [], [], [], []]
        offset = branch.size / 4
        newCenter = np.add(branch.position, np.multiply(offset, m))
        isInsideBorders = [np.multiply(pointArray < c + offset,pointArray > c - offset) for c in newCenter]
        mergedStates = [a[:,0] * a[:,1] * a[:,2] for a in isInsideBorders]

        for i in range(8):
            pointsBranch[i] = pointArray[mergedStates[i]]

        return pointsBranch, newCenter

    # Insert DepthMap into the octomap
    def insertFromDepthMap(self, depthMap, intrinsic, depthScale=1, image=None, rayCast=False, maxDepth=0):
        # cx = intrinsic[0,2]
        # cy = intrinsic[1,2]
        # fx = intrinsic[0,0]
        # fy = intrinsic[1,1]
        h,w = depthMap.shape

        if image is None:
            image = np.zeros((h,w,3))
        if len(image.shape) < 3:
            colorArray = image.reshape(h*w)
        else:
            colorArray = image.reshape(h*w,3)


        # starting from root, dispatch points in the eight root branches
        # branch: 0 1 2 3 4 5 6 7
        # x:      - - - - + + + +
        # y:      - - + + - - + +
        # z:      - + - + - + - +
        # m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
        #      [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
        # offset = self.root.size / 2
        # pointsBranch = [[], [], [], [], [], [], [], []]
        # ptCloud = []
        # u,v = np.meshgrid(np.arange(0,w),np.arange(0,h))
        # u = u.reshape(h*w,1)
        # v = v.reshape(h*w,1)
        # depthMap = depthMap.reshape(h*w,1)
        # z = depthMap / depthScale
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        #
        # ptCloud = np.concatenate((x,y,z), axis=1)

        # depth to 3d point cloud
        h,w = depthMap.shape
        x,y = np.meshgrid(range(w), range(h), indexing='xy')
        # id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

        pix_coords = np.stack([x.reshape(1,h*w), y.reshape(1,h*w), np.ones((1,h*w))]).astype(np.float32).reshape(3,h*w)

        K_inv = np.linalg.inv(intrinsic)

        cam_points = np.matmul(K_inv[:3,:3], pix_coords)
        cam_points = depthMap.reshape(1,h*w) * cam_points

        ptCloud= cam_points[:3,:]
        ptCloud = np.transpose(ptCloud)

        # Filter out points outside max depth

        # if maxDepth > 0:
        #     idx = (x**2+y**2+z**2)**(1/2) < maxDepth
        #     ptCloud = ptCloud[idx[:,0]]
        #     colorArray = colorArray[idx[:,0]]
        #
        idx = ptCloud[:,2] > 0
        ptCloud = ptCloud[idx]
        colorArray = colorArray[idx]

        if len(ptCloud) > 0:
            self.root.isLeafNode = False

        # allocate the point to the corresponding branch
        pointsBranch, branchPosition, colorBranch = Octomap.__sortByBranches(ptCloud, self.root, colorArray)

        for k in range(8):
            self.root.branches[k] = self.__insertPointCloud(pointsBranch[k], self.root.branches[k], self.root, branchPosition[k], colorBranch[k])

        if rayCast:
            self.rayCast()

    def insertFromPtCloud(self, ptCloud, colorArray, rayCast=False):
        # start_time = time.time()
        if len(ptCloud) > 0:
            self.root.isLeafNode = False

        if ptCloud.shape[0] == 3:
            ptCloud = torch.transpose(ptCloud,0,1)
        # allocate the point to the corresponding branch
        # pointsBranch, branchPosition, colorBranch = self.__sortByBranches(ptCloud, self.root, colorArray)
        #
        # for k in range(8):
        #     self.root.branches[k] = self.__insertPointCloud(pointsBranch[k], self.root.branches[k], self.root, branchPosition[k], colorBranch[k])


        # dispatch points in the eight branches
        # pointsBranch, branchPosition, colorBranch = self.__sortByBranches(pointArray, branch, colorArray)
        mask, branchPosition = self.__sortByBranches(ptCloud, self.root, colorArray)

        for k in range(8):
            self.root.branches[k] = self.__insertPointCloud(ptCloud[mask[k]], self.root.branches[k], self.root, branchPosition[k], colorArray[mask[k]])

        if rayCast:
            self.rayCast()

        # self.time_insertFromPtCloud += time.time() - start_time

    def getMaxDepth(self):
        max_depth = 0
        for i, x in enumerate(self.iterateDepthFirst()):
            if x.depth > max_depth:
                max_depth = x.depth
        return max_depth

    def cutTree(self, maxDepth):
        # Navigate through the tree. When we reach max depth,
        # we get the max value occupancy value of its branches

        # Duplicate tree
        self.__cutTree(maxDepth, self.root)

    def __cutTree(self, maxDepth, root):
        for i in range(8):
            branch = root.branches[i]
            if branch is None or branch.isLeafNode:
                continue
            elif branch.depth == maxDepth:
                # Cut the children branches
                root.branches[i].isLeafNode = True
                root.branches[i].branches = [None, None, None, None, None, None, None, None]
                continue
            else:
                self.__cutTree(maxDepth, branch)

    def visualize(self, max_depth=0, split=False, displayFree=False):
        # Extract nodes and convert them to point cloud
        pt = []
        # updateMaxDepth = False
        if max_depth == 0:
            max_depth = self.limit_depth

        for i, x in enumerate(self.iterateDepthFirst()):
            if displayFree:
                pt.append(Point(x.position, size=x.size, depth=x.depth, color=x.data["color"]))
            else:
                if x.data["occupancy"] != 0:
                    pt.append(Point(x.position.squeeze().cpu().detach().numpy(), size=x.size, depth=x.depth, color=x.data["color"].squeeze().detach().cpu().numpy()))
            # if x.data[0].occupancy == 0:
            # if x.depth > max_depth and updateMaxDepth:
            #     max_depth = x.depth

        # Split point cloud in order to have points in every cube util mx depth.
        if split:
            pt = self.__split2maxDepth(pt, max_depth)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector([p.position for p in pt])
        if pt[0].color is None:
            pcd.colors = open3d.utility.Vector3dVector([[0.2, p.depth / max_depth, 0.2] for p in pt])
        else:
            pcd.colors = open3d.utility.Vector3dVector([p.color if np.max(p.color) <= 1 else p.color / 255 for p in pt])
        # pcd.colors = open3d.utility.Vector3dVector([[0.2, p.depth / max_depth, 0.2] for p in pt])
        # pcd.colors = open3d.utility.Vector3dVector([p.color for p in pt])
        voxels = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self.worldSize/(2**(max_depth)))
        # Tree = open3d.geometry.Octree(max_depth)
        # Tree.create_from_voxel_grid(voxels)
        open3d.visualization.draw_geometries([voxels])

    def __split2maxDepth(self, pointCloud, max_depth, output=[]):
        if pointCloud == []:
            return output

        ptCloud = []
        matrix = [[-1,-1,-1],
               [-1,-1, 1],
               [-1, 1,-1],
               [-1, 1, 1],
               [ 1,-1,-1],
               [ 1,-1, 1],
               [ 1, 1,-1],
               [ 1, 1, 1]]

        for p in pointCloud:
            if p.depth >= max_depth:
                output.append(p)
            else:
                newSize = p.size / 4
                newDepth = p.depth + 1
                pos = np.add(np.multiply(matrix, newSize), p.position)
                if newDepth == max_depth:
                    for i in range(8):
                        output.append(Point(pos[i], size=p.size / 2, depth=newDepth, color=p.color))
                else:
                    for i in range(8):
                        ptCloud.append(Point(pos[i], size=p.size / 2, depth=newDepth, color=p.color))

        return self.__split2maxDepth(ptCloud, max_depth, output)

    def rayCast(self):
        # Insert new point along the line between the point P and the origin O with a step of the minimal resolution.
        resolution = self.worldSize / 2**self.getMaxDepth()
        itDepth = self.iterateDepthFirst()

        pointArray = []
        for it in itDepth:
            pointArray.append(constructRay(resolution, self.origin, it))

        pointArray = np.vstack(pointArray)

        pointsBranch, branchPosition = Octomap.__sortByBranches_noColor(pointArray, self.root)

        for k in range(8):
            self.root.branches[k] = self.__rayCastTree(pointsBranch[k], self.root.branches[k], self.root, branchPosition[k])


    def __rayCastTree(self, pointArray, branch, parent, branchPosition):
        if len(pointArray) == 0:
            return branch

        color = (0.5,0.5,0.5)
        if branch is None:
            branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, [Point(branchPosition, occupancy=0, color=color)])
            # branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, [Point(point, occupancy=0, color=color) for point in pointArray])
        elif not branch.isLeafNode:
            pointsBranch, branchPosition = Octomap.__sortByBranches_noColor(pointArray, branch)
            for k in range(8):
                branch.branches[k] = self.__rayCastTree(pointsBranch[k], branch.branches[k], branch, branchPosition[k])
        return branch

    def compareToOctomap(self, octomapBis, limit_depth=None):
        # compare the octomap to another.
        # We want it to be as generic as possible. So no notion of size is considered here.
        # We evaluate the Octree. The evalutation is based on the data.occupancy value.

        if limit_depth is None:
            limit_depth = self.limit_depth

        # If an octomap is empty or contains a single element we stop the process
        if self.root.isLeafNode or octomapBis.root.isLeafNode:
            print("One of the octomap is too small to be evaluated")
            return 0
        # Accurcy: percentage of correctly matched nodes
        # cross entropy loss
        self.loss = torch.zeros(2, dtype=torch.float32).to(self.device)
        self.__digIntoDepth(self.root.branches, octomapBis.root.branches, limit_depth)
        loss = self.loss[1] / self.loss[0]
        return loss

    def __digIntoDepth(self, branches1, branches2, limit_depth):
        for i in range(8):
            branch1 = branches1[i]
            branch2 = branches2[i]
            if not (branch1 is None or branch2 is None): # if both of them are not None
                if branch1.depth == limit_depth:
                    self.__evaluateNode(branch1.data["occupancy"], branch2.data["occupancy"], branch1.depth,limit_depth)
                elif not (branch1.isLeafNode or branch2.isLeafNode):
                    self.__digIntoDepth(branch1.branches, branch2.branches, limit_depth)

            # At this step, it means that at least one branch is a LeafNode

            # Test NoneTuype for an empty (unknown) node:
            if branch1 is None and branch2 is None:
                # no information so we ignore
                continue
                # accuracy += 1 / (8**depth)
            elif branch1 is None and not branch2.isLeafNode:
                if branch2.depth == limit_depth:
                    self.__evaluateNode(self.free_prob, branch2.data["occupancy"], branch2.depth,limit_depth)
                else:
                    self.__digIntoOneTree(self.free_prob, branch2.branches, branch2.depth + 1, limit_depth, False)
            elif branch2 is None and not branch1.isLeafNode:
                if branch1.depth == limit_depth:
                    self.__evaluateNode(branch1.data["occupancy"], self.free_prob, branch1.depth,limit_depth)
                else:
                    self.__digIntoOneTree(self.free_prob, branch1.branches, branch1.depth + 1, limit_depth, True)
            # Test is they are both leaf node. In that case, we compare the values

            elif branch1 is None: # branch2 is necessary a leafe node != None -> can only be a miss
                self.__evaluateNode(self.free_prob, branch2.data["occupancy"], branch2.depth,limit_depth)
            elif branch2 is None: # branch1 is necessary a leafe node != None
                self.__evaluateNode(branch1.data["occupancy"], self.free_prob, branch1.depth,limit_depth)


            elif branch1.isLeafNode and branch2.isLeafNode:
                self.__evaluateNode(branch1.data["occupancy"], branch2.data["occupancy"], branch1.depth,limit_depth)
                # accuracy += int(branch1.data[0].occupancy == branch2.data[0].occupancy) / (8**depth)
                # miss += int(branch1.data["occupancy"] != branch2.data["occupancy"]) / (8**branch1.depth)
                # match += int(branch1.data["occupancy"] == branch2.data["occupancy"]) / (8**branch1.depth)
            # If a single one is a leafe node, we get the corresponding value
            # and keep digging into the other tree to compare it with the childens values
            elif branch1.isLeafNode:
                if branch1.depth >= limit_depth:
                    self.__evaluateNode(branch1.data["occupancy"], branch2.data["occupancy"], branch1.depth,limit_depth)
                    # miss += int(branch1.data["occupancy"] != branch2.data["occupancy"]) / (8**branch1.depth)
                    # match += int(branch1.data["occupancy"] == branch2.data["occupancy"]) / (8**branch1.depth)
                else:
                    self.__digIntoOneTree(branch1.data["occupancy"], branch2.branches, branch2.depth + 1, limit_depth, False)
            elif branch2.isLeafNode:
                if branch2.depth >= limit_depth:
                    self.__evaluateNode(branch1.data["occupancy"], branch2.data["occupancy"], branch2.depth,limit_depth)
                    # miss += int(branch1.data["occupancy"] != branch2.data["occupancy"]) / (8**branch2.depth)
                    # match += int(branch1.data["occupancy"] == branch2.data["occupancy"]) / (8**branch2.depth)
                else:
                    self.__digIntoOneTree(branch2.data["occupancy"], branch1.branches, branch1.depth + 1, limit_depth, True)

    # cross entropy loss between two probability distributions
    def __evaluateNode(self, ref, pred, depth, limit_depth):
        # counter
        self.loss[0] += 2**(limit_depth - depth)
        # cross entropy
        alpha = 2
        self.loss[1] += -(alpha * ref * torch.log(pred) + (1 - ref) * torch.log(1 - pred)) * 2**(limit_depth - depth)


    def __digIntoOneTree(self, comparisonValue, branches, depth, limit_depth, digIntoBranch1):
        assert(depth is not None)
        if comparisonValue is None:
            comparisonValue = self.free_prob
        for branch in branches:
            if branch is not None:
                # force the evaluation if we reach the limit depth
                if depth >= limit_depth or branch.isLeafNode:
                    if digIntoBranch1:
                        self.__evaluateNode(branch.data["occupancy"], comparisonValue, depth,limit_depth)
                    else:
                        self.__evaluateNode(comparisonValue, branch.data["occupancy"], depth,limit_depth)
                    # miss += int(comparisonValue != branch.data["occupancy"]) / (8**depth)
                    # match += int(comparisonValue == branch.data["occupancy"]) / (8**depth)
                else:#elif not branch.isLeafNode:
                    self.__digIntoOneTree(comparisonValue, branch.branches, depth + 1, limit_depth, digIntoBranch1)
            # If we reach this point, the branch is a node leaf or None
            else: #branch is None:
                if comparisonValue != self.free_prob:
                    if digIntoBranch1:
                        self.__evaluateNode(self.free_prob, comparisonValue, depth,limit_depth)
                    else:
                        self.__evaluateNode(comparisonValue, self.free_prob, depth,limit_depth)


    # Evaluate the likelihood of an octomap with the self octomap used as a GT reference.
    def evaluateOctomap(self, octomapToEvaluate, limit_depth=None):
        if limit_depth is None:
            limit_depth = self.limit_depth

        if self.root.isLeafNode:
            print("The octomap of reference is empty. It can't be used as an evaluation.")
            return False

        if octomapToEvaluate.root.isLeafNode:
            print("The octomap is too small to be evaluated. Accuracy = 0")
            return 0
        # Accurcy: percentage of correctly matched nodes
        miss, match = Octomap.__digIntoDepthSingleWay(self.root.branches, octomapToEvaluate.root.branches, 0, 0, limit_depth)
        accuracy = match / (miss + match)
        print(f"match: {match}")
        print(f"miss: {miss}")
        return accuracy

    @staticmethod
    def __digIntoDepthSingleWay(refBranches, evalBranches, miss, match, limit_depth):
        # Similar to digIntoDepth but do an oriented evaluation.
        # Only evaluate nodes with known information from the reference branch

        # Rules :
        # If refBranch is None : Ignore
        # Else evaluate or dig deeper

        for i in range(8):
            refBranch = refBranches[i]
            evalBranch = evalBranches[i]

            # If they are both not leafnodes we dig deeper
            if not (refBranch is None or evalBranch is None):
                if refBranch.depth >= limit_depth:
                    miss += int(refBranch.data["occupancy"] != evalBranch.data["occupancy"]) / (8**refBranch.depth)
                    match += int(refBranch.data["occupancy"] == evalBranch.data["occupancy"]) / (8**refBranch.depth)
                elif not (refBranch.isLeafNode or evalBranch.isLeafNode):
                    miss, match = Octomap.__digIntoDepthSingleWay(refBranch.branches, evalBranch.branches, miss, match, limit_depth)

            # If ref branch is None, we ignore
            elif refBranch is None:
                continue
            # If eval branch is None but ref branch is not
            elif evalBranch is None:
                # if ref is leafNode we evaluate
                if refBranch.isLeafNode:
                    miss += int(None != refBranch.data["occupancy"]) / (8**refBranch.depth)
                    match += int(None == refBranch.data["occupancy"]) / (8**refBranch.depth)
                # Else we dig into the tree
                else:
                    miss, match = Octomap.__digIntoOneTree(None, refBranch.branches, miss, match, refBranch.depth + 1, limit_depth)
            #If they are both leaf node, we evaluate
            elif refBranch.isLeafNode and evalBranch.isLeafNode:
                miss += int(refBranch.data["occupancy"] != evalBranch.data["occupancy"]) / (8**refBranch.depth)
                match += int(refBranch.data["occupancy"] == evalBranch.data["occupancy"]) / (8**refBranch.depth)
            # If a single one is a leafe node, we get the corresponding value
            # and keep digging into the other tree to compare it with the childens values
            elif refBranch.isLeafNode:
                miss, match = Octomap.__digIntoOneTree(refBranch.data["occupancy"], evalBranch.branches, miss, match, evalBranch.depth + 1, limit_depth)
            elif evalBranch.isLeafNode:
                miss, match = Octomap.__digIntoOneTree(evalBranch.data["occupancy"], refBranch.branches, miss, match, refBranch.depth + 1, limit_depth)

        # print("Accuracy =", accuracy)
        return miss, match


    @staticmethod
    def __getDepthAtBranches(branches):
        for b in branches:
            if b is not None:
                return b.depth
        return None

    def rayCast2(self):
        ## navigate through the tree and replace unknown area with free area when necessary

        # Construit 8 points clouds correspondant aux noeuds occupés de la première
        # subdivision en 8 du cube.
        # Conversion des points de coordonnées cartériennes en coordonnées sphériques

        # On parcourt ensuite l'arbre de chacun des 8 cubes.
        # Pour chaque noeud inconnu, on calcul les huits points correspondant
        # aux sommets du cube. On teste ensuite si il y a des points du point cloud
        # qui rentre dans les critères suivant :
        # -> r_occ > r_max
        # -> phi_min < phi_occ < phi_max
        # -> theta_min < theta_occ < theta_max
        # Si au moins un point du point cloud rempli les critères, le noeud est considéré comme libre.

        for i, branch in enumerate(self.root.branches):
            ptCloud = []
            ptCloud = Octomap.getPointCloudInBranch(ptCloud, branch)
            if len(ptCloud):
                ptCloud = np.array(ptCloud)
                self.root.branches[i] = self.rayCastTree2(np.array(ptCloud), branch, self.root, i)



    @staticmethod
    def getPointCloudInBranch(ptCloud, branch):
        if branch is None:
            return ptCloud
        elif branch.isLeafNode:
            x,y,z = branch.position
            r = (x**2 + y**2 + z**2)**(1/2)
            phi = np.arctan(abs(y / x))
            theta = np.arctan((x**2 + y**2)**(1/2) / abs(z))
            ptCloud.append([r, phi, theta])
        else:
            for b in branch.branches:
                ptCloud = Octomap.getPointCloudInBranch(ptCloud, b)

        return ptCloud


    def rayCastTree2(self, ptCloud, branch, parent, branchNumber):
        if branch is None:
            m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
                 [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
            offset = parent.size / 4
            branchCenter = np.add(parent.position, np.multiply(offset, m[branchNumber]))
            branchBorders = np.add(branchCenter, np.multiply(offset, m))
            x = branchBorders[:,0]
            y = branchBorders[:,1]
            z = branchBorders[:,2]
            r = (x**2 + y**2 + z**2)**(1/2)
            phi = np.arctan(abs(y)/np.maximum(abs(x),1e-7))
            theta = np.arctan((x**2 + y**2)**(1/2) / np.maximum(abs(z),1e-7))

            # if np.any(ptCloud[:,0] > np.max(r)):
            #     if np.any((ptCloud[:,1] > np.min(phi)) * (ptCloud[:,1] < np.max(phi))):
            #         if np.any((ptCloud[:,2] > np.min(theta)) * (ptCloud[:,2] < np.max(theta))):
            #             branch = OctNode(branchCenter, parent.size / 2, parent.depth + 1, [Point(branchCenter, occupancy=0, color=(0.5,0.5,0.5))])

            if np.any((ptCloud[:,0] > np.max(r))
                    * (ptCloud[:,1] > np.min(phi)) * (ptCloud[:,1] < np.max(phi))
                    * (ptCloud[:,2] > np.min(theta)) * (ptCloud[:,2] < np.max(theta))):
                branch = OctNode(branchCenter, parent.size / 2, parent.depth + 1, [Point(branchCenter, occupancy=0, color=(0.5,0.5,0.5))])

            # np.any(ptCloud < np.array([0, np.max(phi), np.max(theta)])) and np.any(ptCloud > np.array([np.max(r), np.min(phi), np.min(theta)])):


        elif not branch.isLeafNode:
            m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
                 [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
            offset = parent.size / 4
            branchCenter = np.add(parent.position, np.multiply(offset, m[branchNumber]))
            branchBorders = np.add(branchCenter, np.multiply(offset, m))
            x = branchBorders[:,0]
            y = branchBorders[:,1]
            z = branchBorders[:,2]
            r = (x**2 + y**2 + z**2)**(1/2)
            phi = np.arctan(abs(y)/np.maximum(abs(x),1e-7))
            theta = np.arctan((x**2 + y**2)**(1/2) / np.maximum(abs(z),1e-7))

            # if np.any(ptCloud[:,0] > np.max(r)):
            #     if np.any((ptCloud[:,1] > np.min(phi)) * (ptCloud[:,1] < np.max(phi))):
            #         if np.any((ptCloud[:,2] > np.min(theta)) * (ptCloud[:,2] < np.max(theta))):
            #             branch = OctNode(branchCenter, parent.size / 2, parent.depth + 1, [Point(branchCenter, occupancy=0, color=(0.5,0.5,0.5))])

            idx = ((ptCloud[:,0] > np.min(r)) # Difference here : we compare to the minimum value
                * (ptCloud[:,1] > np.min(phi)) * (ptCloud[:,1] < np.max(phi))
                * (ptCloud[:,2] > np.min(theta)) * (ptCloud[:,2] < np.max(theta)))

            if np.any(idx):
                for i, b in enumerate(branch.branches):
                    branch.branches[i] = self.rayCastTree2(ptCloud[idx], b, branch, i)

        return branch

    def getTime(self):
        print(f"insertFromPtCloud : {self.time_insertFromPtCloud}")
        print(f"__insertPointCloud : {self.time___insertPointCloud}")
        print(f"__sortByBranches : {self.time___sortByBranches}")
