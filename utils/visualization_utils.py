import os 
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.widgets import Slider

class Index:
    """
    Simple class to track index.
    """

    def __init__(self, initial: int, minimum: int, maximum: int) -> None:
        """
        Args:
            initial:  initial index
            min:  minimum index
            max:  maximum index
        """
        self.idx = initial
        self.minimum = minimum
        self.maximum = maximum

        # check if the current index is not outside of the range
        if self.idx < self.minimum or self.idx > self.maximum:
            raise ValueError('Initialized index outside of specified range.')
    
    def current(self) -> int:
        return self.idx
    
    def add(self, step: int = 1) -> None:
        """
        Args:
            step:  number to add to idx.
        """
        self.idx = min(self.idx+step, self.maximum)

    def subtract(self, step: int = 1) -> None:
        """
        Args:
            step:  number to subtract from idx.
        """
        self.idx = max(self.idx-step, self.minimum)


class EventTracker:
    """
    Tool for visualization of a serie of images.
    Use the scroll wheel to scroll through the images.
    Press 'SHIFT' to scroll at a higher scrolling speed.
    """
    # define the class attributes
    scroll_speed = 2
    class_mode = False

    def __init__(
        self, 
        ax: np.ndarray, 
        image_tensor: torch.Tensor, 
        idx: int = 0,
        class_idx: int = 0, 
        cmap: str = 'gray',
        vmin: float = 0.0,
        vmax: float = 1.0
    ) -> None:
        """ 
        Args:
            ax:  matplotlib.pyplot figure axis.
            image_tensor:  image series with the following expected shape: (instance, channel, height, width).
            idx:  slice that is displayed first when the figure is created.
            class_idx:  index for corresponding channel to show.
            cmap:  colormap recognized by matplotlib.pyplot module.
            vmin:  image intensity used as minimum for the colormap.
            vmax:  image intensity used as maximum for the colormap.
        """
        # define instance attributes
        self.ax = ax
        self.image_tensor = image_tensor
        self.speed = 1

        self.image = self.ax.imshow(self.image_tensor[idx, class_idx, :, :], vmin=vmin, vmax=vmax, cmap=cmap)

        # initialize objects to track the class and instance index
        shape = image_tensor.shape
        self.idx = Index(idx, 0, shape[0]-1)
        self.class_idx = Index(class_idx, 0, shape[1]-1)
        self.selected_idx = self.idx

        # set the first image
        self.update()

    def onscroll(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """ 
        Add or subtract 'self.speed' from the selected index (either instance or class) when scrolling up or down respectively.
        Update the image frame afterwards. Do not update the index when scolling up for the last or down for the first image.

        Args:
            event:  mouse event (up or down scroll).
        """
        if event.button == 'up':
            # update the index after scolling
            self.selected_idx.add(self.speed)
            self.update()

        elif event.button == 'down': 
            # update the index after scolling
            self.selected_idx.subtract(self.speed)
            self.update()

    def keypress(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Args:
            event:  key event (press) that is checked for a connected action.
                    If there is one, the implemented action is performed.
        """
        for key in event.key.split('+'): # handles multiple keys when pressed at once
            # increases the scrolling speed
            if key == 'shift' and self.speed == 1:
                self.speed = self.scroll_speed
            # scrolling now influences class index
            if key == 'control' and self.selected_idx == self.idx:
                self.selected_idx = self.class_idx

    def keyrelease(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Args:
            event:  key event (release) that is checked for a connected action.
                    If there is one, the implemented action is performed.
        """
        for key in event.key.split('+'): # handles multiple keys when pressed at once
            # decreases scrolling speed
            if key == 'shift' and self.speed == self.scroll_speed:
                self.speed = 1
            # scrolling now influences instance index
            if key == 'control' and self.selected_idx == self.class_idx:
                self.selected_idx = self.idx

    def update(self) -> None:
        """ 
        Update the image and corresponding labels.
        """
        # load the new image
        image = self.image_tensor[self.idx.current(), self.class_idx.current(), :, :]
        self.ax.set_title(f'Instance: {self.idx.current()}/{self.idx.maximum}, Channel: {self.class_idx.current()}/{self.class_idx.maximum}')
        self.image.set_data(image)

        # update the canvas
        self.ax.figure.canvas.draw()


def image_viewer(
    tensor: torch.Tensor,
    idx: int = 0,
    class_idx: int = 0,
    cmap: str = 'gray',
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """ 
    Tool to view an image volume by scrolling through the slices.

    Args:
        tensor:  image series with the following expected shape: (instance, channel, height, width).
        idx:  slice that is displayed first when the figure is created.
        class_idx:  index for corresponding channel to show.
        cmap:  colormap recognized by matplotlib.pyplot module.
        vmin:  image intensity used as minimum for the colormap.
        vmax:  image intensity used as maximum for the colormap.
    """
    # create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')

    # create tracker object and connect it to the figure
    tracker = EventTracker(ax, tensor, idx, class_idx, cmap, vmin, vmax)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.keypress)
    fig.canvas.mpl_connect('key_release_event', tracker.keyrelease)

    plt.show()