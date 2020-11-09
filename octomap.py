from __future__ import print_function

from multiprocessing import Process, Pool
from functools import partial

import numpy as np
import open3d
import time
import octomap_utils

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
        self.lower = (position[0] - half, position[1] - half, position[2] - half)
        self.upper = (position[0] + half, position[1] + half, position[2] + half)

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
    def __init__(self, worldSize, origin=(0, 0, 0), limit_nodes=1, limit_depth=10):
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
        # if limit_nodes = 0, there is no limit
        self.limit_nodes = limit_nodes
        # if limit_depth = 0, there is no limit
        self.limit_depth = limit_depth


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

    # # Manipulate Point object
    # def insertPointCloud(self, ptCloud, occupancy=None):
    #     # Insert point cloud to the Octomap
    #     # If the occupancy occupancy is specified, it will be used to define the occupancy.
    #     # Otherwise we use standard occupancy value (or none at first)
    #     if occupancy == None:
    #         occupancy = [1 for i in range(len(ptCloud))]
    #
    #     if len(ptCloud) == 1:
    #         return self.insertNode(ptCloud[0], Point(ptCloud[0], occupancy=occupancy[0]))
    #
    #     # There is more than one point, so root is not a leafenode anymore.
    #     self.root.isLeafNode = False
    #
    #     # self.__insertPointCloud(ptCloud, self.root, self.root)
    #     # dispatch points in the eight branches
    #     # branch: 0 1 2 3 4 5 6 7
    #     # x:      - - - - + + + +
    #     # y:      - - + + - - + +
    #     # z:      - + - + - + - +
    #     m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
    #          [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
    #     offset = self.root.size / 2
    #     pointsBranch = [[], [], [], [], [], [], [], []]
    #     for i in range(len(ptCloud)):
    #         # We start at root and dispatch points in the corresponding branches
    #
    #
    #         for j in range(8):
    #             newCenter = np.add(self.root.position, np.multiply(offset, m[j]))
    #             # Inside x bound
    #             if (ptCloud[i][0] > newCenter[0] - offset) and (ptCloud[i][0] < newCenter[0] + offset):
    #                 # Inside y bound
    #                 if (ptCloud[i][1] > newCenter[1] - offset) and (ptCloud[i][1] < newCenter[1] + offset):
    #                     # Inside z bound
    #                     if (ptCloud[i][2] > newCenter[2] - offset) and (ptCloud[i][2] < newCenter[2] + offset):
    #                         pointsBranch[j].append(Point(ptCloud[i], occupancy=occupancy[i]))
    #                         break
    #
    #     for k in range(8):
    #         process = Process(target=self.__insertPointCloud, args=(pointsBranch[k], self.root.branches[k], self.root))
    #         process.start()

    # # Manipulate point objects
    # def __insertPointCloud(self, PointArray, branch, parent):
    #     if len(PointArray) == 0:
    #         branch = None
    #     elif len(PointArray) == 1:
    #         branch = OctNode(parent.position, parent.size / 2, parent.depth + 1, PointArray[0])
    #     else:
    #         if branch is None:
    #             branch = OctNode(parent.position, parent.size / 2, parent.depth + 1, [])
    #         branch.isLeafNode = False
    #
    #         # dispatch points in the eight branches
    #         # branch: 0 1 2 3 4 5 6 7
    #         # x:      - - - - + + + +
    #         # y:      - - + + - - + +
    #         # z:      - + - + - + - +
    #         m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
    #              [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
    #         pointsBranch = [[], [], [], [], [], [], [], []]
    #         offset = branch.size / 2
    #
    #         pointsBranch = Octomap.__sortByBranches(PointArray, branch)
    #
    #         for k in range(8):
    #             process = Process(target=self.__insertPointCloud, args=(pointsBranch[k], branch.branches[k], branch))
    #             process.start()
    #         # for k in range(8):
    #         #     self.__insertPointCloud(pointsBranch[k], branch.branches[k], branch)
    #     return branch

    # No loop, don't manipulate array of Point object
    def __insertPointCloud_noloop(self, pointArray, branch, parent, branchPosition, colorArray):
        if len(pointArray) == 0:
            # print("DEBUG - octomap.py - line 409 : branch None")
            branch = None
        # elif len(pointArray) == 1:
        #     branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, [Point(point, occupancy=1, color) for point, color in zip(pointArray, colorArray)])
        elif (len(pointArray) == 1) or (self.limit_depth > 0 and parent.depth + 1 >= self.limit_depth):
            branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, [Point(point, occupancy=1, color=color) for point, color in zip(pointArray, colorArray)])
            # print(f"Branch created - position:{branch.data[0].position}, color:{branch.data[0].color}")
        else:
            if branch is None:
                # print("DEBUG - octomap.py - line 421 : Create Branch OctNode")
                branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, [])
            else:
                if branch.data is not None:
                    pointArray.append(branch.data[0].position)
                    branch.data = []
            branch.isLeafNode = False

            # dispatch points in the eight branches
            pointsBranch, branchPosition, colorBranch = Octomap.__sortByBranches(pointArray, branch, colorArray)

            for k in range(8):
                branch.branches[k] = self.__insertPointCloud_noloop(pointsBranch[k], branch.branches[k], branch, branchPosition[k], colorBranch[k])
            # for k in range(8):
            #     process = Process(target=self.__insertPointCloud_noloop, args=(pointsBranch[k], branch.branches[k], branch, branchPosition[k], colorBranch[k]))
            #     process.start()
        return branch

    @staticmethod
    def __sortByBranches(pointArray, branch, colorArray):
        # dispatch points in the eight branches
        # branch: 0 1 2 3 4 5 6 7
        # x:      - - - - + + + +
        # y:      - - + + - - + +
        # z:      - + - + - + - +
        m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
             [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
        pointsBranch = [[], [], [], [], [], [], [], []]
        colorBranch = [[], [], [], [], [], [], [], []]
        offset = branch.size / 4
        newCenter = np.add(branch.position, np.multiply(offset, m))
        isInsideBorders = [np.multiply(pointArray < c + offset,pointArray > c - offset) for c in newCenter]
        mergedStates = [a[:,0] * a[:,1] * a[:,2] for a in isInsideBorders]

        for i in range(8):
            pointsBranch[i] = pointArray[mergedStates[i]]
            colorBranch[i] = colorArray[mergedStates[i]]

        return pointsBranch, newCenter, colorBranch

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

    # anchor to insertFromDepthMap_noloop
    def insertFromDepthMap(self, depthMap, intrinsic, depthScale=1, image=None, rayCast=False, maxDepth=0):
        return self.insertFromDepthMap_noloop(depthMap, intrinsic, depthScale, image, rayCast, maxDepth)

    # # Vanilla solution using Insert Point
    # def insertFromDepthMap(self, depthMap, intrinsic, depthScale=1, image=None, rayCast=False, maxDepth=0):
    #     cx = intrinsic[0,2]
    #     cy = intrinsic[1,2]
    #     fx = intrinsic[0,0]
    #     fy = intrinsic[1,1]
    #     h,w = depthMap.shape
    #
    #     ptCloud = []
    #     color = []
    #     for v in range(h):
    #         for u in range(w):
    #             d = depthMap[v,u]
    #             if d > 0:
    #                 z = d / depthScale
    #                 x = (u-cx) * z / fx
    #                 y = (v - cy) * z / fy
    #                 #print((x**2+y**2+z**2)**(1/2))
    #                 if (maxDepth > 0) and ((x**2+y**2+z**2)**(1/2) > maxDepth):
    #                         continue
    #                 ptCloud.append((x,y,z))
    #                 if not (image is None):
    #                     c = image[v][u]
    #                     if len(image.shape)<3:
    #                         color.append([c/255,c/255,c/255])
    #                     else:
    #                         color.append([c[2]/255,c[1]/255,c[0]/255])
    #
    #     for i in range(len(ptCloud)):
    #         obj = Point(ptCloud[i], color=color[i], occupancy=1)
    #         self.insertNode(obj.position, obj)

    # # Solution with insert point cloud sithout inserting them one by one. And a small amount of multiprocessing
    # def insertFromDepthMap2(self, depthMap, intrinsic, depthScale=1, image=None, rayCast=False, maxDepth=0):
    #     cx = intrinsic[0,2]
    #     cy = intrinsic[1,2]
    #     fx = intrinsic[0,0]
    #     fy = intrinsic[1,1]
    #     h,w = depthMap.shape
    #
    #     ptCloud = []
    #     color = []
    #     start_time = time.time()
    #     for v in range(h):
    #         for u in range(w):
    #             d = depthMap[v,u]
    #             if d > 0:
    #                 z = d / depthScale
    #                 x = (u-cx) * z / fx
    #                 y = (v - cy) * z / fy
    #                 #print((x**2+y**2+z**2)**(1/2))
    #                 if (maxDepth > 0) and ((x**2+y**2+z**2)**(1/2) > maxDepth):
    #                     continue
    #                 if np.any((x,y,z) < self.root.lower):
    #                     continue
    #                 if np.any((x,y,z) > self.root.upper):
    #                     continue
    #                 ptCloud.append((x,y,z))
    #                 if not (image is None):
    #                     c = image[v][u]
    #                     if len(image.shape)<3:
    #                         color.append([c/255,c/255,c/255])
    #                     else:
    #                         color.append([c[2]/255,c[1]/255,c[0]/255])
    #     print(f"DEBUG octomap.py line 475 : Loop executed in {time.time() - start_time}")
    #
    #     self.insertPointCloud(ptCloud)
    #
    # # Advance solution of solution 2 which gets rid of one loop by directly storing it in the correct branch
    # def insertFromDepthMap3(self, depthMap, intrinsic, depthScale=1, image=None, rayCast=False, maxDepth=0):
    #     cx = intrinsic[0,2]
    #     cy = intrinsic[1,2]
    #     fx = intrinsic[0,0]
    #     fy = intrinsic[1,1]
    #     h,w = depthMap.shape
    #
    #
    #     self.root.isLeafNode = False
    #     # starting from root, dispatch points in the eight root branches
    #     # branch: 0 1 2 3 4 5 6 7
    #     # x:      - - - - + + + +
    #     # y:      - - + + - - + +
    #     # z:      - + - + - + - +
    #     m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
    #          [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
    #     offset = self.root.size / 2
    #     pointsBranch = [[], [], [], [], [], [], [], []]
    #     for v in range(h):
    #         for u in range(w):
    #             d = depthMap[v,u]
    #             if d > 0:
    #                 z = d / depthScale
    #                 x = (u-cx) * z / fx
    #                 y = (v - cy) * z / fy
    #
    #                 # test if inside depth limits (if there is one)
    #                 if (maxDepth > 0) and ((x**2+y**2+z**2)**(1/2) > maxDepth):
    #                     continue
    #                 if np.any((x,y,z) < self.root.lower):
    #                     continue
    #                 if np.any((x,y,z) > self.root.upper):
    #                     continue
    #
    #                 # allocate the point to the corresponding branch
    #                 for j in range(8):
    #                     newCenter = np.add(self.root.position, np.multiply(offset, m[j]))
    #                     # Inside x bound
    #                     if (x > newCenter[0] - offset) and (x < newCenter[0] + offset):
    #                         # Inside y bound
    #                         if (y > newCenter[1] - offset) and (y < newCenter[1] + offset):
    #                             # Inside z bound
    #                             if (z > newCenter[2] - offset) and (z < newCenter[2] + offset):
    #                                 color = [0,0,0]
    #                                 if not (image is None):
    #                                     c = image[v][u]
    #                                     if len(image.shape)<3:
    #                                         color = [c/255,c/255,c/255]
    #                                     else:
    #                                         color = [c[2]/255,c[1]/255,c[0]/255]
    #                                 pointsBranch[j].append(Point((x,y,z), occupancy=1, color=color))
    #                                 break
    #
    #
    #     for k in range(8):
    #         process = Process(target=self.__insertPointCloud, args=(pointsBranch[k], self.root.branches[k], self.root))
    #         process.start()
    #         # for k in range(8):
    #         #     self.__insertPointCloud(pointsBranch[k], branch.branches[k], branch)


    # Solution without big loops
    def insertFromDepthMap_noloop(self, depthMap, intrinsic, depthScale=1, image=None, rayCast=False, maxDepth=0):
        cx = intrinsic[0,2]
        cy = intrinsic[1,2]
        fx = intrinsic[0,0]
        fy = intrinsic[1,1]
        h,w = depthMap.shape

        if image is None:
            image = np.zeros((h,w,3))
        if len(image.shape) < 3:
            colorArray = image.reshape(h*w)
        else:
            colorArray = image.reshape(h*w,3)



        self.root.isLeafNode = False
        # starting from root, dispatch points in the eight root branches
        # branch: 0 1 2 3 4 5 6 7
        # x:      - - - - + + + +
        # y:      - - + + - - + +
        # z:      - + - + - + - +
        m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
             [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
        offset = self.root.size / 2
        pointsBranch = [[], [], [], [], [], [], [], []]
        ptCloud = []
        u,v = np.meshgrid(np.arange(0,w),np.arange(0,h))
        u = u.reshape(h*w,1)
        v = v.reshape(h*w,1)
        depthMap = depthMap.reshape(h*w,1)
        z = depthMap / depthScale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        ptCloud = np.concatenate((x,y,z), axis=1)
        # print(f"ptCloud length : {len(ptCloud)}")
        # Filter out points outside max depth
        if maxDepth > 0:
            idx = (x**2+y**2+z**2)**(1/2) > maxDepth
            ptCloud = ptCloud[idx[:,0]]
            colorArray = colorArray[idx[:,0]]

        idx = ptCloud[:,2] > 0
        ptCloud = ptCloud[idx]
        colorArray = colorArray[idx]

        # allocate the point to the corresponding branch
        pointsBranch, branchPosition, colorBranch = Octomap.__sortByBranches(ptCloud, self.root, colorArray)

        for k in range(8):
            self.root.branches[k] = self.__insertPointCloud_noloop(pointsBranch[k], self.root.branches[k], self.root, branchPosition[k], colorBranch[k])


    # # Vanilla solution using Insert Point
    # def insertFromDepthMap5(self, depthMap, intrinsic, depthScale=1, image=None, rayCast=False, maxDepth=0):
    #     cx = intrinsic[0,2]
    #     cy = intrinsic[1,2]
    #     fx = intrinsic[0,0]
    #     fy = intrinsic[1,1]
    #     h,w = depthMap.shape
    #
    #     m = [[-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
    #          [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1]]
    #     offset = self.root.size / 2
    #     pointsBranch = [[], [], [], [], [], [], [], []]
    #     ptCloud = []
    #     u,v = np.meshgrid(np.arange(0,w),np.arange(0,h))
    #     u = u.reshape(h*w,1)
    #     v = v.reshape(h*w,1)
    #     depthMap = depthMap.reshape(h*w,1)
    #     z = depthMap / depthScale
    #     x = (u - cx) * z / fx
    #     y = (v - cy) * z / fy
    #
    #     ptCloud = np.concatenate((x,y,z), axis=1)
    #     # Filter out points outside max depth
    #     if maxDepth > 0:
    #         idx = (x**2+y**2+z**2)**(1/2) > maxDepth
    #         # print(ptCloud)
    #         ptCloud = ptCloud[idx[:,0]]
    #
    #     idx = ptCloud[:,2] > 0
    #     ptCloud = ptCloud[idx]
    #
    #     for i in range(len(ptCloud)):
    #         obj = Point(ptCloud[i], occupancy=1)
    #         self.insertNode(obj.position, obj)

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

    def visualize(self, max_depth=0, split=False):
        # Extract nodes and convert them to point cloud
        pt = []
        # updateMaxDepth = False
        if max_depth == 0:
            max_depth = self.limit_depth

        for i, x in enumerate(self.iterateDepthFirst()):
            # if x.data[0].occupancy == 0:
            pt.append(Point(x.position, size=x.size, depth=x.depth, color=x.data[0].color))
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
        voxels = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self.worldSize/(2**max_depth))
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

    # def rayCast(self):
    #     # Insert new point along the line between the point P and the origin O with a step of the minimal resolution.
    #     if self.limit_depth > 0:
    #         maxDepth = self.limit_depth
    #     else:
    #         maxDepth = self.getMaxDepth()
    #
    #     resolution = self.worldSize / 2**(maxDepth)
    #     itDepth = self.iterateDepthFirst()
    #     O = self.origin
    #     print("maxDepth:", maxDepth)
    #     print("step:",resolution)
    #     for i, x in enumerate(itDepth):
    #         P = x.position
    #         OP = [P[0] - O[0], P[1] - O[1], P[2] - O[2]]
    #         step = np.multiply(resolution, OP / np.linalg.norm(OP))
    #         # all points between the endpoint and the origin
    #         pointArray = np.array([np.arange(O[0], P[0], step[0]), np.arange(O[1], P[1], step[1]) , np.arange(O[2], P[2], step[2])]).T
    #         # print(f"{i}: step={step}")
    #         self.__rayCast(pointArray[1:])
    #         # process = Process(target=self.__rayCast, args=(x.position, step))
    #         # process.start()

    # def rayCast_2(self):
    #     # Insert new point along the line between the point P and the origin O with a step of the minimal resolution.
    #     if self.limit_depth > 0:
    #         maxDepth = self.limit_depth
    #     else:
    #         maxDepth = self.getMaxDepth()
    #
    #     resolution = self.worldSize / 2**(maxDepth)
    #     itDepth = self.iterateDepthFirst()
    #     O = self.origin
    #     print("maxDepth:", maxDepth)
    #     print("step:",resolution)
    #     for i, x in enumerate(itDepth):
    #         P = x.position
    #         OP = [P[0] - O[0], P[1] - O[1], P[2] - O[2]]
    #         step = np.multiply(resolution, OP / np.linalg.norm(OP))
    #         # all points between the endpoint and the origin
    #         pts = np.array([np.arange(O[0], P[0], step[0]), np.arange(O[1], P[1], step[1]) , np.arange(O[2], P[2], step[2])]).T
    #         # pointArray.append(pts[1:])
    #         if i == 0:
    #             pointArray = pts[1:]
    #         else:
    #             pointArray = np.concatenate((pointArray, pts[1:]))
    #         # print(f"{i}: step={step}")
    #
    #     self.__rayCast(pointArray)

    def rayCast(self):
        # Insert new point along the line between the point P and the origin O with a step of the minimal resolution.
        resolution = self.worldSize / 2**self.getMaxDepth()
        itDepth = self.iterateDepthFirst()
        p = Pool()
        pointArray = p.map(partial(octomap_utils.constructRay, resolution, self.origin), itDepth)
        p.close()
        p.join()
        pointArray = np.vstack(pointArray)

        pointsBranch, branchPosition = Octomap.__sortByBranches_noColor(pointArray, self.root)

        for k in range(8):
            self.root.branches[k] = self.__rayCastTree(pointsBranch[k], self.root.branches[k], self.root, branchPosition[k])


    def __rayCastTree(self, pointArray, branch, parent, branchPosition):
        if len(pointArray) == 0:
            return branch

        color = (0.5,0.5,0.5)
        if branch is None:
            branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, [Point(point, occupancy=0, color=color) for point in pointArray])
        elif not branch.isLeafNode:
            pointsBranch, branchPosition = Octomap.__sortByBranches_noColor(pointArray, branch)
            for k in range(8):
                branch.branches[k] = self.__rayCastTree(pointsBranch[k], branch.branches[k], branch, branchPosition[k])
        return branch

    # def __rayCastTree(self, pointArray, branch, parent, branchPosition):
    #     if len(pointArray) == 0:
    #         return branch
    #     color = (0.5,0.5,0.5)
    #     if (len(pointArray) == 1) or (self.limit_depth > 0 and parent.depth + 1 >= self.limit_depth):
    #         branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, [Point(point, occupancy=0, color=color) for point in pointArray])
    #
    #     elif branch is None:
    #         # branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, [Point(point, occupancy=0, color=color) for point in pointArray])
    #         branch = OctNode(branchPosition, parent.size / 2, parent.depth + 1, [])
    #         branch.isLeafNode = False
    #
    #     pointsBranch, branchPosition = Octomap.__sortByBranches_noColor(pointArray, branch)
    #
    #     for k in range(8):
    #         branch.branches[k] = self.__rayCastTree(pointsBranch[k], branch.branches[k], branch, branchPosition[k])
    #     # elif branch.data is None:
    #     #     return OctNode(branchPosition, parent.size / 2, parent.depth + 1, [Point(point, occupancy=0, color=color) for point in pointArray])
    #     return branch

    # def __rayCast(self, pointArray):
    #     # If empty array
    #     if pointArray.shape[0] == 0:
    #         return True
    #     # Study the first point only:
    #     p = tuple(pointArray[0])
    #     # Locate point in the octree
    #     node = self.findPosition(p)
    #
    #     if node is None:
    #         # print("is None")
    #         node = self.insertNode(p, Point(p, occupancy=0, color=(0.5,0.5,0.5)))
    #         node = self.findPosition(p)
    #         # node = self.findPosition(p)
    #     elif node.data is None:
    #         node = self.insertNode(p, Point(p, occupancy=0, color=(0.5,0.5,0.5)))
    #         # print("Node is None")
    #
    #     # Test how many points are inside this node
    #     size = node.size
    #     center = node.position
    #     isInsideBorders = np.multiply(pointArray < center + size / 2, pointArray > center - size / 2)
    #     mergedStates = isInsideBorders[:,0] * isInsideBorders[:,1] * isInsideBorders[:,2]
    #     pointArray = pointArray[mergedStates == 0, :]
    #     # print(pointArray)
    #
    #     self.__rayCast(pointArray)

        # # move point toward the origin
        # # compute manhattan distance at the border of the cube
        # m1 = [[ 0, 0,-1],[ 0, 0, 1],[ 0,-1, 0],[ 0, 1, 0],[-1, 0, 0],[ 1, 0, 0]]
        # m2 = [[ 0, 0, 1],[ 0, 0, 1],[ 0, 1, 0],[ 0, 1, 0],[ 1, 0, 0],[ 1, 0, 0]]
        # o = self.origin
        #
        # # a = p + ( m1 * size / 2 + m2 * (origin - p))
        # #
        # # ----o-----------
        # # |   ^          |
        # # |   |          |
        # # |   |          |
        # # o<--x--------->o
        # # |   |          |
        # # |   v          |
        # # ----o-----------
        # #
        # # We then compute the distance to the origin to see in which direction we must go
        # limits = np.add(p, np.add(np.multiply(m1,size / 2),np.multiply(m2,np.add(o,np.multiply(-1,p)) )))
        #
        # # euclidean distance
        # # d = [ ((p[0]-o[0])**2 + (p[1]-o[1])**2 + (p[2]-o[2])**2)**(1/2) for a in limits ]
        #
        # dmin = np.inf
        # idx = None
        # for i in range(6):
        #     d = ((limits[i,0]-o[0])**2 + (limits[i,1]-o[1])**2 + (limits[i,2]-o[2])**2)**(1/2)
        #     if d < dmin:
        #         dmin = d
        #         idx = i
        #
        # # We want to be just outside the current cube
        # epsilon = self.worldSize / 2**self.getMaxDepth()
        #
        # # [x,y,z] = t * [xp,yp,zp] + [xo,yo,zo]
        # # t = ([x,y,z] - [xo,yo,zo]) / [xp,yp,zp]
        # # We fixed one of the value to identify t
        # top = np.multiply((limits[idx] - o), m2[idx])
        # btm = np.multiply(p, m1[idx])
        # # Extract the non-zero value
        # top = np.sum(top)
        # btm = np.sum(top)
        #
        # offset = (top + epsilon) / btm
        #
        # position = np.multiply(offset, p)
        #
        # if ((position[0] - o[0])**2 + (position[1] - o[1])**2 + (position[2] - o[2])**2)**(1/2) < epsilon:
        #     return True
        #
        # octnode = self.insertNode(position, Point(position, occupancy=0, color=(1,1,1)))

        # node = self.findPosition(position)
        # if node == None:
        #     self.insertNode(position, Point(position, occupancy=0, color=(1,1,1)))
        #     node = self.findPosition(position)

        # # new point
        # P = np.add(position, step)
        # nodeData = self.findPosition(P)
        # count = 0
        # if nodeData is None:
        #     self.insertNode(P, Point(P, occupancy=0, color=1))
        #
        # O = self.origin
        # PO = [O[0] - P[0], O[1] - P[1], O[2] - P[2]]
        # # print("point P:",P)
        # # print("distance to origin:", np.linalg.norm(PO))
        # if np.linalg.norm(PO) < np.linalg.norm(step):
        #     return True
        # else:
        #     return self.__rayCast(P, step)


    def compareToOctomap(self, octomapBis):
        # compare the octomap to another.
        # We want it to be as generic as possible. So no notion of size is considered here.
        # We evaluate the Octree. The evalutation is based on the data.occupancy value.

        # If an octomap is empty or contains a single element we stop the process
        if self.root.isLeafNode or octomapBis.root.isLeafNode:
            print("One of the octomap is too small to be evaluated")
            return False
        # Accurcy: percentage of correctly matched nodes
        miss, match = Octomap.__digIntoDepth(self.root.branches, octomapBis.root.branches, miss=0, match=0)
        accuracy = match / (miss + match)
        return accuracy

    @staticmethod
    def __digIntoDepth(branches1, branches2, miss, match):
        depth = Octomap.__getDepthAtBranches(branches1)
        for i in range(8):
            branch1 = branches1[i]
            branch2 = branches2[i]
            if not (branch1 is None or branch2 is None):
                if not (branch1.isLeafNode or branch2.isLeafNode):
                    # print('dig!')
                    # print(depth)
                    miss, match = Octomap.__digIntoDepth(branch1.branches, branch2.branches, miss, match)
            # while not (branch1.isLeafNode or branch2.isLeafNode):
            #     accuracy = Octomap.__digIntoDepth(branch1.branches, branch2.branches, accuracy)

            # At this step, it means that at least one branch is a LeafNode

            # Test NoneTuype for an empty (unknown) node:
            if branch1 is None and branch2 is None:
                # no information so we ignore
                continue
                # accuracy += 1 / (8**depth)
            elif branch1 is None and not branch2.isLeafNode:
                miss, match = Octomap.__digIntoOneTree(None, branch2.branches, miss, match)
            elif branch2 is None and not branch1.isLeafNode:
                miss, match = Octomap.__digIntoOneTree(None, branch1.branches, miss, match)
            # Test is they are both leaf node. In that case, we compare the values

            elif branch1 is None:
                # accuracy += int(None == branch2.data[0].occupancy) / (8**depth)
                miss += int(None != branch2.data[0].occupancy) / (8**depth)
                match += int(None == branch2.data[0].occupancy) / (8**depth)
            elif branch2 is None:
                # accuracy += int(branch1.data[0].occupancy == None) / (8**depth)
                miss += int(None != branch1.data[0].occupancy) / (8**depth)
                match += int(None == branch1.data[0].occupancy) / (8**depth)

            elif branch1.isLeafNode and branch2.isLeafNode:
                # accuracy += int(branch1.data[0].occupancy == branch2.data[0].occupancy) / (8**depth)
                miss += int(branch1.data[0].occupancy != branch2.data[0].occupancy) / (8**depth)
                match += int(branch1.data[0].occupancy == branch2.data[0].occupancy) / (8**depth)
            # If a single one is a leafe node, we get the corresponding value
            # and keep digging into the other tree to compare it with the childens values
            elif branch1.isLeafNode:
                miss, match = Octomap.__digIntoOneTree(branch1.data[0].occupancy, branch2.branches, miss, match)
            elif branch2.isLeafNode:
                miss, match = Octomap.__digIntoOneTree(branch2.data[0].occupancy, branch1.branches, miss, match)

        # print("Accuracy =", accuracy)
        return miss, match

    @staticmethod
    def __digIntoOneTree(comparisonValue, branches, miss, match):
        depth = Octomap.__getDepthAtBranches(branches)
        for branch in branches:
            if branch is not None:
                if not branch.isLeafNode:
                    miss, match = Octomap.__digIntoOneTree(comparisonValue, branch.branches, miss, match)
            # while not branch.isLeafNode:
            #     accuracy = Octomap.__digIntoOneTree(comparisonValue, branch.branches, accuracy)
            # If we reach this point, the branch is a node leaf or None
            elif branch is None:
                # accuracy += int(comparisonValue == None) / (8**depth)
                miss += int(comparisonValue != None) / (8**depth)
                match += int(comparisonValue == None) / (8**depth)
            elif branch.data is None:
                # accuracy += int(comparisonValue == None) / (8**depth)
                miss += int(comparisonValue != None) / (8**depth)
                match += int(comparisonValue == None) / (8**depth)
            else:
                # accuracy += int(comparisonValue == branch.data[0].occupancy) / (8**depth)
                miss += int(comparisonValue != branch.data[0].occupancy) / (8**depth)
                match += int(comparisonValue == branch.data[0].occupancy) / (8**depth)

        return miss, match

    @staticmethod
    def __getDepthAtBranches(branches):
        for b in branches:
            if b is not None:
                return b.depth
        return None
