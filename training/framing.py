"""Moderate crops measured relative to the source aspect ratio."""
import torch


class RandomFrameCrop:
    def __init__(self, probability=0.7, minimum_fraction=0.75):
        if not 0 <= probability <= 1 or not 0 < minimum_fraction <= 1:
            raise ValueError('Invalid crop probability or minimum fraction')
        self.probability = probability
        self.minimum_fraction = minimum_fraction

    def box(self, size):
        width, height = size
        if float(torch.rand(())) >= self.probability:
            return (0, 0, width, height)
        # Independent width/height retention varies framing and aspect ratio.
        w = max(1, round(width * float(torch.empty(()).uniform_(self.minimum_fraction, 1))))
        h = max(1, round(height * float(torch.empty(()).uniform_(self.minimum_fraction, 1))))
        x = int(torch.randint(width-w+1, ()))
        y = int(torch.randint(height-h+1, ()))
        return (x, y, x+w, y+h)

    def __call__(self, image):
        return image.crop(self.box(image.size))
