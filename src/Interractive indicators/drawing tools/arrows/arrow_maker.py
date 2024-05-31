import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

%matplotlib notebook

class ArrowPlot:
    def __init__(self, dataframe):
        self.df = dataframe
        self.ax, self.fig = plt.subplot()
        self.ax.scatter(self.df[x], self.df[y])
        self.arrow = None
        self.start = None
        self.end = None

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid.motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_click(self, event):
        if self.start is None:
            self.start = (event.xdata, event.ydata)
        else:
            self.end = (event.xdata, event.ydata)
            self.draw_arrow()

    def on_motion(self, event):
        if self.start and not self.end:
            self.end = (event.xdata, event.ydata)
            self.update_arrow()
        elif self.arrow and event.button == 1:
            self.end = (event.xdata, event.ydata)
            self.update_arrow()

    def on_release(self, event):
        """
        if self.start and self.end:
            self.end = (event.xdata, event.ydata)
            self.update_arrow()
            """
        if self.arrow:
            self.start = None
            self.end = None
    
    def draw_arrow(self):
        if self.arrow:
            self.arrow.remove()
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        width = np.hypot(dx, dy) * 0.01 # 1% of the hypotenuse
        self.arrow = FancyArrow(self.start[0], self.start[1], dx, dy, width=width, color='black')
        self.ax.add_patch(self.arrow)
        self.fig.canvas.draw()

    def update_arrow(self):
        if self.arrow:
            self.arrow.remove()
        if self.start and self.end:
            self.draw_arrow()
    
    def show_results(df):
        arrowplot = ArrowPlot(df)
        plot.show()

## needs checking