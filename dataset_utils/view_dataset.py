import argparse
import logging
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

mplstyle.use('seaborn-notebook')


class ScrollView(object):
    def __init__(self, data):
        self.data = np.round(data)
        self.samples = self.data.shape[0]
        self.ind = self.samples // 2

        self.fig = plt.figure(figsize=(4, 4), tight_layout=True)
        gs = gridspec.GridSpec(1, 1)

        self.lastind = 0
        self.axes = [self.fig.add_subplot(gs[0, 0])]
        self.text_scroll = self.axes[0].text(
            0.95, 0.95, 'use mouse wheel to scroll', transform=self.axes[0].transAxes, va='top', ha='right', color='white')
        self.ims = [axis_i.imshow(self.data[self.ind],
                                  cmap='cividis') for n_axis, axis_i in enumerate(self.axes)]
        for axis_i in self.axes:
            axis_i.get_xaxis().set_visible(False)
            axis_i.get_yaxis().set_visible(False)

        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        plt.show()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.samples
        else:
            self.ind = (self.ind - 1) % self.samples
        self.text_scroll.set_text(
            'sample {} of {}'.format(self.ind + 1, self.samples))
        for n_axis, im in enumerate(self.ims):
            im.set_data(self.data[self.ind])
            im.axes.figure.canvas.draw()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='dataset.npy')
    args = parser.parse_args()
    dataset = np.load(args.data_file)
    _ = ScrollView(dataset)
