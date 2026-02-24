import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
from skimage.segmentation import slic, mark_boundaries
import time
# from PIL import Image
# import cv2

class PerturbationBase(nn.Module):
    def __init__(self, input_size, device='cpu', N=100, p1=0.1):
        super(PerturbationBase, self).__init__()
        self.input_size = input_size
        self.device = device
        self.p1 = p1
        self.N = N

    def generate_masks_rise(self, N, s, p1, savepath='masks.npy'):
        """
        Generate random masks for the RISE algorithm.
        
        Parameters:
        - N: The number of masks to generate.
        - s: The size of the grid that the image is divided into (s x s), resolution.
        - p1: Probability of a grid cell being set to 1 (not occluded). This should be a float value in the [0, 1] range
        - savepath: The path where the generated masks are saved.
        """
        
        # set timer
        _init_time = time.time()
        
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size
        print('Cell size:',cell_size)
        
        # Generate random grid
        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')
        print('Grid shape:', grid.shape)

        self.masks = np.empty((N, *self.input_size))
        
        # Generate masks with random shifts
        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping (with skimage)
            _upsampling = resize(grid[i], up_size, order=1, mode='reflect',anti_aliasing=False)
            self.masks[i, :, :] = _upsampling[x:x + self.input_size[0], y:y + self.input_size[1]]
            # Linear upsampling and cropping (with PIL)
            # _upsampling = Image.fromarray(grid[i]).resize(up_size.astype(int), Image.BILINEAR)
            # self.masks[i, :, :] = np.array(_upsampling)[x:x + self.input_size[0], y:y + self.input_size[1]]
            # Linear upsampling and cropping (with cv2)
            # _upsampling = cv2.resize(grid[i], (int(up_size[1]), int(up_size[0])), interpolation=cv2.INTER_LINEAR)
            # self.masks[i, :, :] = _upsampling[x:x + self.input_size[0], y:y + self.input_size[1]]
    
        # Reshape and save the masks    
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        
        # monitor time
        print('Total time: {:.2f}seg'.format(time.time() - _init_time))

        
        # save
        if savepath is not None:
            np.save(savepath, self.masks)
        
        # Load masks to the specified device
        self.masks = torch.from_numpy(self.masks).float()
        self.N = N
        self.p1 = p1
    
    def generate_sliding_window_masks(self, window_size, stride, savepath='sliding_window_masks.npy'):
        """
        Generate masks using the sliding window method.

        Parameters:
        - window_size (tuple): Size of the sliding window, (height, width).
        - stride (int): Stride or step size for the sliding window; it can be extended to be a tuple if (stride_y, stride_x) want to be different.
        - savepath (str, optional): File path to save the generated masks (default: 'sliding_window_masks.npy').

        Returns:
        - None. Modifies the class attribute `self.masks`, storing the generated masks.
        """
        # set timer
        _init_time = time.time()
        
        window_height, window_width = window_size
        input_height, input_width = self.input_size

        stride_y = stride
        stride_x = stride

        vertical_steps = (input_height - window_height) // stride_y + 1
        horizontal_steps = (input_width - window_width) // stride_x + 1

        print(f'vertical steps: {vertical_steps} input height {input_height}, window height {window_height}, stride {stride_y}')
        print(f'horizontal steps: {horizontal_steps} input height {input_width}, window height {window_height}, stride {stride_x}')
        self.masks = np.empty((vertical_steps * horizontal_steps, *self.input_size), dtype='float32')
        k=0

        for i in tqdm(range(vertical_steps), desc='Vertical sliding'):
            for j in range(horizontal_steps):
                mask = np.zeros((input_height, input_width), dtype='float32')
                start_x = i * stride_y
                start_y = j * stride_x
                end_x = start_x + window_height
                end_y = start_y + window_width
                mask[start_x:end_x, start_y:end_y] = 1
                self.masks[k] = mask
                k += 1

        print(f'Generation finished with {vertical_steps}x{horizontal_steps} masks')
        # self.masks = np.array(self.masks)
        # print('Transformed into numpy')
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        print('Reshaped to correct dims')
        
        # monitor time
        print('Total time: {:.2f}seg'.format(time.time() - _init_time))
        
        if savepath is not None:
            print('saving...')
            np.save(savepath, self.masks)
            print('saving completed')
        self.masks = torch.from_numpy(self.masks).float()
        self.N=vertical_steps*horizontal_steps
        
    def generate_mask_mfpp(self, img_np, N, p1, num_levels, savepath='mfpp_masks.npy'):
        """
            Generate masks using the MFPP approach.

            Parameters:
            - img_np: Input image as a NumPy array.
            - N: The number of masks to generate.
            - p1: Proportion of pixels to be set to 1 (not occluded).
            - savepath: The path where the generated masks are saved.
        """
        # Start the timer to measure the time taken for mask generation
        _init_time = time.time()
        
        # Define different levels of segmentation
        segment_levels = [50, 100, 200, 400, 800, 1600]
        # Total number of masks to be generated (a) in total (b) in each level
        num_masks_per_level = int(N)//num_levels
        num_masks = num_masks_per_level*num_levels
        # Preallocate an array to store the masks
        self.masks = np.empty((num_masks, *self.input_size), dtype='float32')

        # Initialize a counter for mask indexing
        k = 0
        # Calculate the total number of pixels in the image
        total_pixels = img_np.shape[0] * img_np.shape[1]
        # Calculate the target number of pixels to be set to 1 based on p1
        target_num_pixels = self.p1 * total_pixels

        # Iterate over the specified number of segment levels
        for level_segments in segment_levels[:num_levels]:
            # Perform SLIC segmentation on the image for the current segment level
            segments = slic(image=img_np, n_segments=level_segments)
            # Get the unique superpixel labels
            unique_segments = np.unique(segments)

            # Generate N masks for each segment level
            for _ in tqdm(range(num_masks_per_level), desc=f'Generating filters for {level_segments} segments'):
                
                # Initialize a mask with zeros (all pixels occluded)
                mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype='float32')        
                # Initialize a counter for the number of pixels set to 1
                current_num_pixels = 0
                # Shuffle the order of superpixel labels to ensure randomness
                np.random.shuffle(unique_segments)

                # Iterate over the shuffled superpixel labels
                for superpixel in unique_segments:
                    # Break the loop if the target number of pixels is reached
                    if current_num_pixels >= target_num_pixels:
                        break
                    # Get the pixel indices for the current superpixel
                    superpixel_mask = (segments == superpixel)
                    # Set the pixels in the current superpixel to 1
                    mask[superpixel_mask] = 1
                    # Update the counter for the number of pixels set to 1
                    current_num_pixels += np.sum(superpixel_mask)
                    
                
                # Store the generated mask in the preallocated array
                self.masks[k] = mask
                k += 1
        
        # Reshape the masks array to include a channel dimension (num_masks, 1, height, width)
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        print(self.masks.shape)
        
        # Print the total time taken for mask generation
        segs = time.time() - _init_time
        mins = int(segs // 60)
        remaining_segs = segs % 60
        print('Total time: {:.2f} seconds ({} min {:.2f} seconds)'.format(segs, mins, remaining_segs))
        
        # Save the generated masks to the specified file if a savepath is provided
        if savepath is not None:
            np.save(savepath, self.masks)
        
        # Convert the masks to a PyTorch tensor and load them onto the specified device
        self.masks = torch.from_numpy(self.masks).float()
        self.N = num_masks
        self.p1 = p1
        
    def load_masks(self, filepath):
        """
        Load masks from a specified file path.
        
        Parameters:
        - filepath: The path from where to load the masks.
        """
        # Load the masks
        self.masks = np.load(filepath)
        # Transfer them to the specified device (depends on if we are going to use batched implem or not)
        if True: 
            # batched --> load firts in cpu, then it would move them in batches to GPU
            self.masks = torch.from_numpy(self.masks).float().to('cpu')
        else:
            # everything in GPU
            self.masks = torch.from_numpy(self.masks).float().to(self.device)
        self.N = self.masks.shape[0] # Update the number of masks
        print(f'Loaded {self.N} masks')