from field import Field, norm
import matplotlib.pyplot as plt
import numpy as np
import rasterio

class FieldPlotter:
    def __init__(self,field : Field):
        self.field=field

    """ PLOT FUNCTIONS """

    def plot(self, indicator, mask):
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = None
        ax.imshow(indicator, cmap=cmap)
        if mask == 1:
            mask = self.field.return_mask()
            ax.imshow(mask, alpha=.33)
        plt.show()

    def bi_plot(self, indicator, before, after, cmap, mask):

        fig, axs = plt.subplots(1, 2, figsize=(15, 15))

        axs[0].imshow(after, cmap=cmap)
        axs[0].set_title(f'{indicator} Scene 0 | Fold {self.field.batch_nr} | Image {self.field.img_nr}')
        axs[1].imshow(before, cmap=cmap)
        if mask == 1:
            mask = self.field.return_mask()
            axs[1].imshow(mask, alpha=.3)
        axs[1].set_title(f'{indicator} Scene 1 | Fold {self.field.batch_nr} | Image {self.field.img_nr}')
        plt.show()

    def bi_plot_mask(self, img):
        mask = self.field.return_mask()
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        cmap = None
        axs[0].imshow(img, cmap=cmap, vmin=-1, vmax=1)
        axs[0].set_title(f'RESULT | Fold {self.field.batch_nr} | Image {self.field.img_nr}')
        axs[1].imshow(mask)
        axs[1].set_title(f'MASK | Fold {self.field.batch_nr} | Image {self.field.img_nr}')
        plt.show()

    def plot_watermask(self):
        water_mask = self.field.get_water_mask()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(water_mask)

    def plot_hist(self, metric):

        counts, bins = np.histogram(metric.flatten())
        plt.hist(bins[:-1], bins, weights=counts)
        plt.ylim(0, 200)

    """ PLOT INDICES FUNCTIONS"""

    def plot_rgb(self, mask, factor):
        r, g, b = self.field.get_rgb(0)

        # Stack the bands to create an RGB image
        RGB_after = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])

        r, g, b = self.field.get_rgb(1)
        RGB_before = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])

        cmap = None
        self.bi_plot("RGB", RGB_before * factor, RGB_after * factor, cmap=cmap, mask=mask)

    def plot_abai(self, mask):

        B03 = self.field.bands(0, 2)
        B11 = self.field.bands(0, 10)
        B12 = self.field.bands(0, 11)
        ABAI_after = (3 * B12 - 2 * B11 - 3 * B03) / ((3 * B12 + 2 * B11 + 3 * B03) + 1e-10)
        B03 = self.field.bands(1, 2)
        B11 = self.field.bands(1, 10)
        B12 = self.field.bands(1, 11)
        ABAI_before = (3 * B12 - 2 * B11 - 3 * B03) / ((3 * B12 + 2 * B11 + 3 * B03) + 1e-10)

        cmap = None
        self.bi_plot("ABAI", ABAI_before, ABAI_after, cmap=cmap, mask=mask)

    def plot_metric(self, mask=0):
        metric = self.field.calculate_metric()

        # Plot the output
        self.plot(metric, mask)

        # Plot output with mask comparison
        self.bi_plot_mask(metric)
