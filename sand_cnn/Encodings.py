import numpy as np

class Encodings:

    def __init__(self):
        pass

    def rle_encode(self, mask):
        '''
        Encode pixel runs into rle format to save space.
        Args:
            mask: image that contains an an array of the pixel runs.
        Returns:
            A string of the rle encoding
        '''
        pixels = mask.T.flatten()
        use_padding = False
        if pixels[0] or pixels[-1]:
            use_padding = True
            pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
            pixel_padded[1:-1] = pixels
            pixels = pixel_padded
        rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
        if use_padding:
            rle = rle - 1
        rle[1::2] = rle[1::2] - rle[:-1:2]
        return rle

    def rle_decode(self, rle_str, mask_shape, mask_dtype):
        '''
        Return the full array of data given an rle string
        Args:
            rle_str: String that represents the pixel runs
            mask_shape: The array shape of the mask you wish to return
            mask_dtype: The dtype of the array
        Returns:
            A numpy array of the actual pixel values
        '''
        s = rle_str.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 1
        return mask.reshape(mask_shape[::-1]).T

    def test_mask(self, mask):
        '''
        The mask image should only contain 2 colors. Make sure all values are 
        either 0 or 255.
        Args:
            mask: the mask image
        Returns:
            Print to console if a mask has a problem.
        '''
        desired_values = [0, 255]
        mask_correct = np.isin(mask, desired_values).all
        
        if mask_correct:
            pass
        
        else:
            print('Found non-binary pixels in mask')