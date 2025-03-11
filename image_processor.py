"""
Made by Daniel B. and Varick H. as a project for DSC20 at UCSD
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2

        >>> pixels = [
        ...    [[255, 255, 255], [0, 0, 0]],
        ...    [[200, 200, 200], [1, 1, 1]]
        ...    ]
        >>> img = RGBImage(pixels)
        >>> img.num_rows
        2
        >>> img.num_cols
        2

        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255], [0, 2, 256]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        ValueError
        """
        if not all([isinstance(pixels,list) and len(pixels)>=1,
                all(isinstance(row,list) and len(row)>0 for row in pixels),
                all(len(row) == len(pixels[0]) for row in pixels),
                all(isinstance(col,list) for row in pixels for col in row),
                all(len(col) == 3 for row in pixels for col in row)
                ]):
            raise TypeError()
        
        if any([x<0 or x>255 for row in pixels for col in row for x in col]):
            raise ValueError()
        
        self.pixels = pixels
        self.num_rows = len(self.pixels)
        self.num_cols = len(self.pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows,self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[list(col) for col in row] for row in list(self.pixels)]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError
        >>> img.get_pixel(0, 2)
        Traceback (most recent call last):
        ...
        ValueError
        >>> img.get_pixel(0, -1)
        Traceback (most recent call last):
        ...
        ValueError
        >>> img.get_pixel(0.4, 0)
        Traceback (most recent call last):
        ...
        TypeError
        >>> img.get_pixel(0, 0.4)
        Traceback (most recent call last):
        ...
        TypeError
        >>> img.get_pixel(0.4, 0.4)
        Traceback (most recent call last):
        ...
        TypeError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if not(isinstance(row,int) and isinstance(col,int)):
            raise TypeError()
        if row<0 or col<0:
            raise ValueError()
        try:
            return tuple(self.pixels[row][col])
        except IndexError as err:
            raise ValueError()

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if not(isinstance(row,int) and isinstance(col,int)):
            raise TypeError()
        if row<0 or col<0:
            raise ValueError()
        if not(isinstance(new_color,tuple) and len(new_color) == 3 and all([isinstance(x,int) for x in new_color])):
            raise TypeError()
        if any([x>255 for x in new_color]):
            raise ValueError()
        
        try:
            self.pixels[row][col] = [color if color>=0 else self.pixels[row][col][i] for i,color in enumerate(new_color)]
        except IndexError as err:
            raise ValueError()


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        # YOUR CODE GOES HERE #
        negated_pixels = [[[255 - col for col in row] for row in pixel] for pixel in image.pixels]
        return RGBImage(negated_pixels)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        grayscaled_pixels = [[[sum(row)//3, sum(row)//3, sum(row)//3] for row in pixel] for pixel in image.pixels]
        return RGBImage(grayscaled_pixels)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        # YOUR CODE GOES HERE #
        rotated_pixels = [pixel[::-1] for pixel in image.pixels[::-1]]
        return RGBImage(rotated_pixels)

    
    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        # YOUR CODE GOES HERE #
        pixel_list = [pixel for row in image.pixels for pixel in row]
        return sum(sum(pixel)//len(pixel) for pixel in pixel_list) // len(pixel_list)
    
    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(intensity, int):
            raise TypeError()
        if (intensity > 255 or intensity < -255):
            raise ValueError()
        adjusted_pixels = [[[255 if col+intensity > 255 else 0 if col+intensity < 0 else col+intensity \
                             for col in row] for row in pixel] for pixel in image.pixels]
        return RGBImage(adjusted_pixels)
    
    
    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        # YOUR CODE GOES HERE #
        blurred_pixels = [[[0, 0, 0] for y in range(image.num_cols)] for x in range(image.num_rows)]
        
        #defining neighbor index relative to pixel position in the middle
        neighbors = [(-1, -1), (-1, 0), (-1, 1), 
                    (0, -1), (0, 0), (0, 1), 
                    (1, -1), (1, 0), (1, 1)]
        
        for row in range(image.num_rows):
            for col in range(image.num_cols):
                sum_neighbors = [0, 0, 0]
                count = 0
                
                for neighbor_row, neighbor_col in neighbors:
                    c_row, c_col = row + neighbor_row, col + neighbor_col
                    
                    if 0 <= c_row < image.num_rows and 0 <= c_col < image.num_cols:
                        for color in range(3):
                            sum_neighbors[color] += image.pixels[c_row][c_col][color]
                        count += 1
                
                for color in range(3):
                    blurred_pixels[row][col][color] = sum_neighbors[color] // count
        
        return RGBImage(blurred_pixels)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.coupons = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #
        if self.coupons <= 0:
            self.cost = self.cost + 5
        else:
            self.coupons -= 1
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        # YOUR CODE GOES HERE #
        if self.coupons <= 0:
            self.cost = self.cost + 6
        else:
            self.coupons -= 1
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        # YOUR CODE GOES HERE #
        if self.coupons <= 0:
            self.cost = self.cost+10
        else:
            self.coupons -= 1
        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        # YOUR CODE GOES HERE #
        if self.coupons <= 0:
            self.cost = self.cost + 1
        else:
            self.coupons -= 1
        return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        # YOUR CODE GOES HERE #
        if self.coupons <= 0:
            self.cost = self.cost + 5
        else:
            self.coupons -= 1
        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # YOUR CODE GOES HERE #
        if not isinstance(amount, int):
            raise TypeError()
        if not amount > 0:
            raise ValueError()
        self.coupons += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)
        """
        # YOUR CODE GOES HERE #
        if not all([isinstance(chroma_image, RGBImage), isinstance(background_image, RGBImage)]):
            raise TypeError()
        if chroma_image.size() != background_image.size():
            raise ValueError()

        copy_pixels = chroma_image.copy()
        for i in range(len(copy_pixels.pixels)):
            for j in range(len(copy_pixels.pixels[i])):
                if tuple(copy_pixels.pixels[i][j]) == color:
                    copy_pixels.pixels[i][j] = background_image.pixels[i][j]
        return copy_pixels


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        if not all([isinstance(sticker_image,RGBImage), isinstance(background_image,RGBImage)]):
            raise TypeError()
        if sticker_image.size()[0] > background_image.size()[0] or sticker_image.size()[1] > background_image.size()[1]:
            raise ValueError()
        if not all([isinstance(x_pos,int), isinstance(y_pos, int)]):
            raise TypeError()
        if sticker_image.size()[0] > background_image.size()[0] - (y_pos) or sticker_image.size()[1] > background_image.size()[1] - (x_pos):
            raise ValueError()
        
        res = background_image.copy()
        for ith_row,jth_row in zip(range(y_pos,len(res.pixels)+y_pos),range(len(sticker_image.pixels))):
            for ith_col,jth_col in zip(range(x_pos,len(res.pixels[0])+x_pos), range(len(sticker_image.pixels[0]))):
                res.pixels[ith_row][ith_col] = sticker_image.pixels[jth_row][jth_col]

        return res

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        # YOUR CODE GOES HERE #
        nocolor_img = image.copy()

        # Convert the RGB image to a grayscale image using floor division
        for ith_row in range(len(nocolor_img.pixels)):
            for ith_col in range(len(nocolor_img.pixels[0])):
                nocolor_img.pixels[ith_row][ith_col] = sum(nocolor_img.pixels[ith_row][ith_col])//3

        #apply kernel
        nocolor_img_pixels = nocolor_img.pixels
        kernel = [[-1, -1, -1],
                 [-1, 8, -1],
                 [-1, -1, -1]
                 ]
        
        highlighted_pixels = []
        for ith_row in range(len(nocolor_img_pixels)):
            output_row = []
            for ith_col in range(len(nocolor_img_pixels[0])):
                value = 0
                for kern_x in range(len(kernel)):
                    for kern_y in range((len(kernel))):
                        #-1 cuz it has to be -1,-1 from the center
                        img_x_pos = ith_row - 1 + kern_x
                        img_y_pos = ith_col - 1 + kern_y
                        if 0 <= img_x_pos < nocolor_img.size()[0] and 0 <= img_y_pos < nocolor_img.size()[1]:
                            value += nocolor_img_pixels[img_x_pos][img_y_pos] * kernel[kern_x][kern_y]
                if value < 0:
                    value = 0
                if value > 255:
                    value = 255
                output_row.append(value)
            highlighted_pixels.append(output_row)
        
        #convert back to RGB
        for ith_row in range(len(highlighted_pixels)):
            for ith_col in range(len(highlighted_pixels[0])):
                value = highlighted_pixels[ith_row][ith_col]
                highlighted_pixels[ith_row][ith_col] = [value,value,value]

        return RGBImage(highlighted_pixels)

# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        self.k_neighbors = k_neighbors

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        # YOUR CODE GOES HERE #
        if len(data) < self.k_neighbors:
            raise ValueError()
        self.data=data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        if not (isinstance(image1,RGBImage) and isinstance(image2,RGBImage)):
            raise TypeError()
        if not (image1.size() == image2.size()):
            raise ValueError()
        
        sum_of_sq_diff = sum([(a-b)**2 for row1,row2 in zip(image1.pixels,image2.pixels) for col1,col2 in zip(row1,row2) for a,b in zip(col1,col2)])
        res = sum_of_sq_diff**0.5
        return res


    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'

        >>> knn.vote(['lol','aaaa','aaaa','bbbb'])
        'aaaa'
        """
        count_dict = {}
        for label in candidates:
            if label in count_dict:
                count_dict[label] += 1
            else:
                count_dict[label] = 1
        
        max_frequency = max(count_dict.values())
        for label,count in count_dict.items():
            if count == max_frequency:
                return label

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        try:
            self.data = self.data
        except AttributeError as e:
            raise ValueError()
        #data: A list of tuples, where every tuple is a (image, label)
        distances_and_labels = [(self.distance(image,image_data[0]), image_data[1]) for image_data in self.data]
        sorted_tuples = sorted(distances_and_labels, key=lambda x: x[0])

        smallest_labels = [label for distance,label in sorted_tuples[:self.k_neighbors]]
        return self.vote(smallest_labels)


def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label