"""
Implementation of annotation tools for annotation of the four corner points and ruler in ultrasound images.
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils.conversion_utils import polar2cartesian_image


class CornerAnnotationEventTracker:
    """
    Tool for annotating the four corner points in the ultrasound videos.
    o  Use the left mouse button to click on the corner points. 
    Subsequent scripts expect the following order of clicked points:
    1. top left, 2. bottom left, 3. top right, 4. bottom right.
    o  Using the scrolling wheel, the user can scroll through the frames in a clip.
    o  Key commands (only work if non-rectified image is shown):
    -  'c' = copy the previous annotation
    -  'n' = skip the current clip and continue to the next one 
                (useful when a corner is not visible in one clip, but is visible in another with the same shape)
    -  'd' = delete previous point that was clicked
    -  't' = rectify the current image (useful to identify inaccuracies in the annotation)
    -  '=' = increase the intensities (useful for low contrast regions, often at the bottom of the ultrasound image)
    -  '-' = decrease the intensities
    -  ']' = add padding to the outside (useful when corners fall just outside of the image) 
    -  '[' = remove padding (only works if padding was already added)
    -  'ENTER' = accept annotation (only works if four points were selected)
    -  'SHIFT' = increases scrolling speed when pressed
    """
    
    # define the class attributes 
    intensity_factor = 1.5
    padding_step = 5 # px
    scroll_speed = 2
    cart_shape = (256, 384)

    def __init__(
        self, 
        ax: np.ndarray, 
        image_array: np.ndarray, 
        cmap: str = 'gray', 
        vmin: int = 0, 
        vmax: int = 1, 
        init_slice: int = 0, 
        prev_annotation: list = None
    ) -> None:
        """ 
        Args:
            ax:  matplotlib.pyplot figure axis.
            image_array:  image series with the following expected shape: (frame, height, width).
            cmap:  colormap recognized by matplotlib.pyplot module.
            vmin:  image intensity used as minimum for the colormap.
            vmin:  image intensity used as maximum for the colormap.
            init_slice:  slice that is displayed first when the figure is created.
            prev_annotation:  if not None, a tuple with four annotated points is expected,
                              which can easily be loaded in as annotation for the next image.
        """
        # define the general instance attributes
        self.ax = ax
        self.image_array = image_array
        self.padded_image_array = image_array
        self.shape = image_array.shape
        self.prev_annotation = prev_annotation
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap
        
        # initialize the state instance attributes
        self.idx = init_slice
        self.speed = 1
        self.factor_power = 0
        self.pad = 0
        self.polar = True

        # define the attributes related to plotting and storing points and lines
        self.points = []
        self.point_objs = []
        self.line_objs = []
      
        # create the image axis object
        self.image = self.ax.imshow(self.image_array[self.idx, :, :], vmin=self.vmin, vmax=self.vmax, cmap=cmap)

        # create the figure
        self.update()

    def onscroll(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """ 
        Add or subtract 'self.speed' from the image index when scrolling up or down respectively.
        Update the image frame afterwards. Do not update the index when scolling up for the last or down for the first image.

        Args:
            event:  mouse event (up or down scroll).
        """
        # check if the image is currently displayed in polar form 
        # and if the mouse is inside the figure        
        if self.polar and (event.inaxes == self.ax or event.inaxes == None):
            if event.button == 'up':
                # update the index after scolling
                if self.idx + self.speed < self.shape[0]:
                    self.idx += self.speed
                    self.update()
                # else if due to the speed, the index would surpass the max number
                # set the index equal to the max number
                elif self.idx + self.speed >= self.shape[0]:
                    self.idx = self.shape[0]-1
                    self.update()

            elif event.button == 'down': 
                # update the index after scolling
                if self.idx - self.speed > 0:
                    self.idx -= self.speed
                    self.update()
                # else if due to the speed, the index would surpass the min number
                # set the index equal to the min number
                elif self.idx - self.speed <= 0:
                    self.idx = 0
                    self.update()

    def keypress(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Args:
            event:  key event (press) that is checked for a connected action.
                    If there is one, the implemented action is performed.
        """
        # check if the polar state is true
        if self.polar:
            # increase the scrolling speed while shift is pressed
            if event.key == 'shift':
                self.speed = self.scroll_speed

            # delete the last point that was clicked
            elif event.key == 'd':
                # check if there are any points
                if len(self.points) > 0:
                    # remove the point and the point object for plotting
                    del self.points[-1]
                    self.point_objs[-1].remove()
                    del self.point_objs[-1]
                    # if the deleted point is the second point of a line,
                    # also delete the line object
                    if len(self.points)%2 == 1:
                        self.line_objs[-1].remove()
                        del self.line_objs[-1]
                self.update()
            
            # use the previously annotated points if they were given as input argument
            # and if there are currently no other points clicked
            elif event.key == 'c':
                if self.prev_annotation != None and len(self.points) == 0:
                    self.points = self.prev_annotation.copy()
                    self.update()
                        
            # increase the image intensity by one step
            elif event.key == '=':
                self.factor_power += 1
                self.update()
            
            # decrease the image intensity by one step
            elif event.key == '-':
                self.factor_power -= 1
                self.update()
            
            # increate the image padding by one step
            elif event.key == ']':
                self.pad += self.padding_step
                self.points = [(point[0]+self.padding_step, point[1]+self.padding_step) for point in self.points]
                for point in self.point_objs:
                    point.remove()
                for line in self.line_objs:
                    line.remove()   
                self.point_objs, self.line_objs = [], []
                self.change_ax()
                self.update()
            
            # decrease the image padding by one step
            elif event.key == '[':
                if self.pad != 0:
                    self.pad -= self.padding_step
                    self.points = [(point[0]-self.padding_step, point[1]-self.padding_step) for point in self.points]
                    for point in self.point_objs:
                        point.remove()
                    for line in self.line_objs:
                        line.remove()  
                    self.point_objs, self.line_objs = [], []
                    self.change_ax()
                    self.update()

            # use the previously annotated points if they were given as input argument
            # and if there are currently no other points clicked
            elif event.key == 'n':
                self.points = None
                plt.close()

            # close the figure when enter is pressed
            elif event.key == 'enter':
                if len(self.points)!= 4:
                    print('Less than 4 points were selected')
                else:
                    self.points = [(point[0]-self.pad, point[1]-self.pad) for point in self.points]
                    plt.close()

            # show the image after transformation from polar to cartesian using the four annotated corner points
            elif event.key == 't':
                if len(self.points)!= 4:
                    print('Less than 4 points were selected')
                elif self.polar:
                    self.polar = False
                    points = [(point[0]-self.pad, point[1]-self.pad) for point in self.points]
                    cartesian_image = polar2cartesian_image(self.image_array[self.idx,...], points, self.cart_shape)
                    self.show_cartesian(cartesian_image)
        
        # if currently the cartesian image is shown
        else:
            if event.key == 't':
                self.polar = True
                self.update()

    def keyrelease(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Args:
            event:  key event (release) that is checked for a connected action.
                    If there is one, the implemented action is performed.
        """
        # check if the polar state is true
        if self.polar:
            # reset the scrolling speed to 1
            if event.key == 'shift':
                self.speed = 1

    def onclick(self,  event: matplotlib.backend_bases.MouseEvent) -> None:
        """ 
        Save the coordinate of a clicked point if there are not yet 4 points clicked.

        Args:
            event:  mouse event (left button click).
        """
        # check if the polar state is true
        if self.polar:
            if event.button == 1 and len(self.points) < 4 and event.inaxes == self.ax:
                x = event.xdata
                y = event.ydata
                self.points.append((x,y))
        
        self.update()

    def change_ax(self) -> None:
        """ 
        After the padding is altered, a new axis mus be created to handle the change in image shape.
        """
        self.padded_image_array = np.pad(self.image_array, [(0,0), (self.pad, self.pad), (self.pad, self.pad)], mode='constant')
        self.image.remove()
        self.image = self.ax.imshow(self.padded_image_array[self.idx, :, :], vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)

    def update(self) -> None:
        """ 
        Update the image and corresponding labels.
        """
        # load the new image
        frame = self.padded_image_array[self.idx, :, :]*(self.intensity_factor**self.factor_power)
        self.ax.set_title(f'Frame: {self.idx}/{self.shape[0]-1}')
        self.image.set_data(frame)

        # draw the points and lines
        while len(self.points) > len(self.point_objs):
            i = len(self.point_objs)
            p = self.ax.scatter(self.points[i][0], self.points[i][1], facecolors='none', s=40, edgecolors='red', lw=1)
            self.point_objs.append(p)
        
        while len(self.points)//2 > len(self.line_objs):
            i = len(self.line_objs)*2
            l = self.ax.plot([self.points[i][0], self.points[i+1][0]], [self.points[i][1], self.points[i+1][1]], color='red', lw=0.75)
            self.line_objs.append(l[0])

        # update the canvas
        self.ax.figure.canvas.draw()

    def show_cartesian(self, cartesian_image: np.ndarray) -> None:
        """ 
        Remove plotted points and lines. Show the Cartesian image and remove the plotted points.

        Args:
            cartesian_image:  currently displayed frame after transformation to Cartesian system
                              using the annotated points.
        """
        # load the new image
        self.image.set_data(cartesian_image)

        # remove all points and lines from the figure
        for point in self.point_objs:
            point.remove()
        for line in self.line_objs:
            line.remove()  
        self.point_objs, self.line_objs = [], []

        # update the canvas
        self.ax.figure.canvas.draw()

def corner_annotator(array: np.ndarray, prev_annotation: list = None) -> list:
    """ 
    Creates matplotlib figure to annotate the four corner points in lung ultrasound clips.

    Args:
        array:  image series with the following expected shape: (frame, height, width).
        prev_annotation:  if not None, a tuple with four annotated points is expected,
                          which can easily be loaded in as annotation for the next image.
    
    Returns:
        tracker.points:  annotated points in the following order: (top left, bottom left, top right, bottom right)
    """
    # create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.axis('off')

    # create a tracker object and connect it to the figure
    tracker = CornerAnnotationEventTracker(ax, array, prev_annotation=prev_annotation)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.keypress)
    fig.canvas.mpl_connect('key_release_event', tracker.keyrelease)
    fig.canvas.mpl_connect('button_press_event', tracker.onclick)
    plt.show()

    return tracker.points


# --------------------------  PHYSICAL SCALE ANNOTATION  ------------------------------

class ScaleAnnotationEventTracker:
    """
    Tool for annotation of the two corner points in the polar ultrasound data for calculation of the pixel/cm ratio.
    o  Use the left mouse button to click on the corner points. The order of the points is not of importance. 
    o  Using the scrolling wheel, the user can change the distance over which the ratio will be calculated.
    o  Key commands:
        -  'c' = copy the previous annotation
        -  'r' = remove all annotated points
        -  'ENTER' = accept annotation (only works if two points were selected)
        -  'CTRL' = enable point annotation or distance toggling when pressed
    """

    def __init__(
        self,
        ax: np.ndarray,
        prev_annotation: list = None,
        prev_distance: int = None
    ) -> None:
        """ 
        Args:
            ax:  matplotlib.pyplot figure axis.
            prev_annotation:  if not None, a tuple with four annotated points is expected,
                              which can easily be loaded in as annotation for the next image.
        """
        # define the general instance attributes
        self.ax = ax
        self.prev_annotation = prev_annotation
        self.prev_distance = prev_distance

        # define the state instance attributes
        self.vertical = True
        self.active = False
        self.distance = 10

        # define the attributes related to plotting and storing the points and lines
        self.points = []
        self.point_obj = []
        self.line_obj = []

        self.update()

    def onscroll(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """ 
        Add or subtract one to or from the specified distance that is used for the calculation for the pixel/cm ratio.

        Args:
            event:  mouse event (up or down scroll).
        """       
        if self.active:
            if event.button == 'up':
                self.distance += 1
                self.update()

            elif event.button == 'down': 
                self.distance -= 1
                self.update()

    def onclick(self,  event: matplotlib.backend_bases.MouseEvent) -> None:
        """ 
        Save the coordinate of a clicked point if there are not yet 2 points clicked.

        Args:
            event:  mouse event (left button click).
        """
        if self.active and event.button == 1 and len(self.points) < 2:
            x = self.points[0][0] if len(self.points) == 1 and self.vertical else event.xdata
            y = event.ydata
            
            self.points.append((x,y))
            self.update()

    def keypress(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Args:
            event:  key event (press) that is checked for a connected action.
                    If there is one, the implemented action is performed.
        """
        # allow point clicking and toggling of the distance used for the calculation in control mode
        if event.key == 'control' and self.active == False:
            self.active = True
            self.update()
        
        # specify if the line between the two points must be vertical
        # i.e. the x-coordinate of the second point equals that of the first
        elif event.key == 'v':
            self.vertical = not self.vertical
            self.update()

        # delete the plotted points and the line
        elif event.key == 'r':
            # remove points
            for obj in self.point_obj:
                obj.remove()
            self.point_obj = []
            self.points = []

            # remove line
            for obj in self.line_obj:
                obj.remove()
            self.line_obj = []

            self.update()
        
        # use the previously annotated points if they were given as input argument
        # and if there are currently no other points clicked
        elif event.key == 'c':
            if self.prev_annotation != None and self.prev_distance != None and len(self.points) == 0:
                self.points = self.prev_annotation.copy()
                self.distance = self.prev_distance
                self.update()
                    
        # use the previously annotated points if they were given as input argument
        # and if there are currently no other points clicked
        elif event.key == 'n':
            self.points = None
            plt.close()

        # close the figure when enter is pressed
        elif event.key == 'enter':
            if len(self.points)!= 2:
                print('Less than 2 points were selected')
            else:
                plt.close()    
    
    def keyrelease(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Args:
            event:  key event (release) that is checked for a connected action.
                    If there is one, the implemented action is performed.
        """
        # inactivate point clicking and distance toggling
        if event.key == 'control' and self.active:
            self.active = False
            self.update()

    def update(self) -> None:
        """
        Update plotted points and line.       
        """
        # plot the points that were not yet displayed
        while len(self.points) > len(self.point_obj):
            i = len(self.point_obj)
            self.point_obj.append(self.ax.scatter(self.points[i][0], self.points[i][1], color='red'))

        # plot the line if it is not yet displayed
        if len(self.points) == 2 and self.line_obj == []:
            self.line_obj.append(self.ax.plot([point[0] for point in self.points], [point[1] for point in self.points], color='red')[0])

        # update the title
        self.ax.set_title(f'Annotation mode: {self.active}, Vertical: {self.vertical}, Distance: {self.distance}')

        # update the canvas
        self.ax.figure.canvas.draw()

def scale_annotator(image: np.ndarray, prev_annotation: list = None, prev_distance: int = None) -> tuple:
    """ 
    Creates matplotlib figure to annotate the four corner points in lung ultrasound clips.

    Args:
        image:  image with the following expected shape: (height, width).
        prev_annotation:  if not None, a tuple with four annotated points is expected,
                          which can easily be loaded in as annotation for the next image.
    
    Returns:
        tracker.points:  two annotated points
        tracker.distance:  distance in cm
    """
    # create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    ax.imshow(image)

    # create a tracker object and connect it to the figure
    tracker = ScaleAnnotationEventTracker(ax, prev_annotation=prev_annotation, prev_distance=prev_distance)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.keypress)
    fig.canvas.mpl_connect('key_release_event', tracker.keyrelease)
    fig.canvas.mpl_connect('button_press_event', tracker.onclick)
    plt.tight_layout()
    plt.show()

    return tracker.points, tracker.distance