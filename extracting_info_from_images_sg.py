### SETUP ------------------------------------------------------------------------------------------------------------------------------------

## Imports
import zipfile
import os.path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

## Paths
# used in the function explore_basic_image_handling
picture_path = "C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\Tests\\GiantApe.png"
# used in the function convert_image_folder_to_list_and_array
folder_path = "C:\\Users\\Simon\\Pictures\\Skærmbilleder"
# used in the function from_zip_to_image_list
zip_path = "C:\\Users\\Simon\\Desktop\\6 stages of deforestation\\jpg-zip.zip"

### FUNCTIONS --------------------------------------------------------------------------------------------------------------------------------
def get_imlist(path):

    """
    returns a list of filenames for all png images in a directory
    """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
def get_imlist_from_zip(path):
    """
    Construct a str list of all images from a zip.file. Each entry corresponds to a picture
    To open a picture, say picture k, with pil, write Image.open(im_list[k]).show()

    path: is the full path of the zip-file.
    """
    imzip = zipfile.ZipFile(path)
    infolist = imzip.infolist()

    #initialize image list
    im_list = [] 
    
    #add pictures to list
    for f in infolist:
        ifile = imzip.open(f)
        im_list.append(ifile)

    return im_list

def images_to_data_list (image_list):
    """
    Takes a list of images, potentially of different sizes, and returns a list of pixel data, in which each entry is a flattened array containing
    all pixel values. It also returns a list in which each entry contains the shape of a picture in the format (pixel height, pixel width,
    number of integers used to describe 1 pixel) [provided that more than 1 byte is used]
    If the picture has the pixel area NxN, each entry array of the returned list will have length (pixel area x bytes/pixel) ,where bytes/pixel is number of
    integers used to decribe a pixel (1 for greyscale, 3 for 24 bit pictures). 
    
    """
    #Find the number of images in the list
    no_images = len(image_list)
    #Store the shape of each image 
    im_shape_list = []
    #Initialize picture list
    pic_list = []

    for k in range(no_images):
        im = np.array(Image.open(image_list[k]))
        im_shape_list.append(im.shape)
        pic_list.append(im.flatten())
    
    return pic_list, im_shape_list
def images_to_data_arr (image_list, pixel_shape, color = True):

    """
    Takes a list of images, potentially of different sizes, reshapes them to pixel_shape, 
    and returns an array of pixel data, in which each row is a flattened array containing
    all pixel values. 
    The returned array has dmensions (No. of images) x (pixel_shape * bytes/pixel)
    params:
        image_list: a list of images
        pixel_shape: list holding dimensions that all pictures will be resized to in the format 
                    [pixel height, pixel width]
        color: Boolean. If true, the pictures will be assumed to be 24 bit RGB images. 
                        If False, the pictures will be assumed to be 8 bit grey images

    """
    #Find no. of pictures
    no_pictures = len(image_list)

    # Find length of each row
    if color:
        bytes_per_pixel = 3
    else:
        bytes_per_pixel = 1
    
    arr = np.empty([no_pictures, bytes_per_pixel * pixel_shape[0] * pixel_shape[1]])

    #Build matrix arr, with all pixels from one picture per row
    for image in range(no_pictures):
        im_resized = Image.open(image_list[image]).resize((pixel_shape[0], pixel_shape[1]))
        arr[image] = np.array(im_resized).flatten()
    return arr
def convert_to_grey_scale(image_list, pixel_shape): 
    """
    Takes a list of images, potentially of different sizes, reshapes them to pixel_shape, converts them to greyscale,
    and returns an array of pixel data, in which each row is a flattened array containing
    all pixel values. 
    The returned array has dmensions (No. of images) x (pixel_shape * bytes/pixel)
    params:
        image_list: a list of images
        pixel_shape: list holding dimensions that all pictures will be resized to in the format 
                    [pixel height, pixel width]
    """
    n_pictures = len(image_list)

    grey_arr = np.empty([n_pictures, pixel_shape[0] * pixel_shape[1] ])

    for image in range(n_pictures):
        im = Image.open(image_list[image]).convert('L').resize((pixel_shape[0], pixel_shape[1]))
        grey_arr [image] = np.array(im).flatten()
    return grey_arr

def make_histogram_from_data (image_data, image_shape, color = True, bins = 256):
    """
    image_data: A flat array or list containing all pixel values for the image. 
    image_shape: The image shape list has the form [pixel height, pixel width]. 
    color: Default = True. If true,   it is assumed to be a 24 bit RGB pictures, 
    and each color channel will be plotted separately in the histogram. If False, the picture is assumed to be 8bit greyscale
    
    After calling the function, you can add titles and change the layout of the histogram. To see the result, call plt.show()
    """
    plt.figure()
    #determine whether the image is greyscale or in color
    if not color:
        fig = plt.hist(image_data, bins)
    else:
        colors = 3
        RGB = ['r','g','b']
        for color in range(colors):
            pixel_area = image_shape[0] * image_shape[1]
            fig = plt.hist(image_data[int(color * pixel_area): int((color + 1) * pixel_area)], bins, color = RGB[color])
def get_features(data_arr, pixel_shape, color = True, histogram = True):
    """
    Takes an array of image data and extracts a number of features for each image, e.g. the sum of color channel values for each color,
    the average and standard deviation for each color channel
    params:
        data_arr: A (No. of pictures) x (No. of pixels x bytes/pixel) array, in which each row contains all pixels of an image.
        If color = True, the pixel values must be ordered as red, green, blue.
        pixel_shape: List holding the dimensions in units of pixels for each image in the format [pixel height, pixel width]. All images
                    must have the same resolution
        color: Default = True. If True, the pictures are assumed to be 24 bit RGB pictures. Otherwise, they will be assumed to be 8 bit
                greyscale.
        histogram: Default = True. If true, the histogram for each picture (and possibly color channel) will be returned as features.
                   If False, only the other parameters will be returned as feautres
    returns:    
                a feature matrix, in which each row has the form [red histogram, red average, red std, green histogram, ..., blue std].
                if histogram = False, the histogram features will not be returned.
    """

    if not color:
        colors = 1
    else:
        colors = 3

    # Extract relevant values
    no_pixels = pixel_shape[0] * pixel_shape[1]
    no_pictures = np.size(data_arr[:,0])
    # The no. of features (per color) is given by the color histogram along with the color average and std.

    if histogram == True:
        no_features = 256 + 2
        feature_arr = np.empty([no_pictures,colors * no_features])
    else:
        no_features = 2
        feature_arr = np.empty([no_pictures,colors * no_features])

    
    # loop over each color and construct a feature matrix of the form [red, green, blue]
    for color in range(colors):
        # Collect all pixel values of a given color
        color_pixels = data_arr[:,int(color * no_pixels): int((color + 1) * no_pixels)]
        # Calculate the color histogram for each row/picture by using apply_along_axis. The resultant matrix has a row for each picture, and
        # its columns are the corresponding color histogram
        color_histogram = np.apply_along_axis(lambda x: np.histogram(x, bins = 256)[0], axis = 1, arr = color_pixels  )

        # calculate the average color value for each row/picture using numpy.nexaxis to calculate all averages at once
        color_av = np.sum(color_histogram * np.linspace(0,255,256)[np.newaxis,:], axis = 1) / no_pixels
        # calculate the average color value for each row/picture using numpy.nexaxis to calculate all values at once
        color_std = np.sqrt (np.sum (color_histogram *  np.power(np.linspace(0,255,256)[np.newaxis,:] \
                    -np.ones([no_pictures, 256]) *  color_av [:, np.newaxis], 2), axis = 1 ) / no_pixels)
    
        # Collect the features in the feature matrix. The color_av and color_std vectors are added as columns, so that for each color, 
        # a feature row has the format [color histogram, color_av, color_std] 
        if histogram == True:
            feature_arr[:, int(color * no_features): int((color + 1) *  no_features)] = \
                    np.r_['1,2,0', color_histogram, color_av, color_std]
        else:
             feature_arr[:, int(color * no_features): int((color + 1) *  no_features)] = \
                    np.r_['1,2,0', color_av, color_std]

    return feature_arr, no_features


### MAIN -------------------------------------------------------------------------------------------------------------------------------------

## TO DO:


# Then move on to another algorithm. Maybe Bayes or multivariate or cnn neural or Xgit boosted.

# The different sections of main() are stored in the functions below. They can be run independently.
def explore_basic_image_handling():

    ##Some stuff you can do with PIL

    #open images
    pil_image = Image.open(picture_path)
    #pil_image.show()

    #create thumbnail
    #reduced = pil_image.thumbnail((64,64))
    #pil_image.show()

    # use crop to crop and image, and resize to resize
    pil_reduced = pil_image.resize((64,64))
    #pil_reduced.show()

    #convert to greyscale
    pil_grey = pil_image.convert('L')
    #pil_grey.show()

    pil_array = np.array(pil_reduced)
    #get pixel value at a coordinate position
    pil_reduced.getpixel((21,23))

    if 1:
        #make histograms over pixel distribution
        #the fourth row of the picture is all 255, and we disregard it
        im_array = pil_array[:,:,0:3]

        #convert back to image
        image_reobtained = Image.fromarray(im_array[:,:,0])


        plt.figure()
        plt.hist(im_array[:,:,0].flatten(), label = "red", bins = 256, color='r')
        plt.hist(im_array[:,:,1].flatten(), label = "green", bins = 256, color = 'g')
        plt.hist(im_array[:,:,2].flatten(), label = "blue", bins = 256, color = 'b')
        plt.legend()
        plt.show()
def convert_image_folder_to_list_and_array():

    #create list of all pictures in folder
    im_list = get_imlist(folder_path)


    #to get image from list, do
    ims = Image.open(im_list[0])
    #ims.show()

    #Construct data list and pictures shapes of a list of pictures
    im_data, im_shapes = images_to_data_list(im_list)

    #make histograms
    if 1:
        for picture in range (len(im_data)):
       
            make_histogram_from_data(im_data[picture],im_shapes[picture][0:2])
            plt.title(f'Histogram from list for image {picture}')
        plt.show(block=False)


    #construct data matrix from list of images
    pixel_shape = [256, 256]
    arr_data = images_to_data_arr(im_list, pixel_shape)
    
    #make histograms
    for picture in range(len(arr_data[:,0])):
        make_histogram_from_data(arr_data[picture],pixel_shape)
        plt.title(f'Histogram from array for image {picture}')
    plt.show(block = False)
def extract_features_from_images():
    #create list of all pictures in folder
    im_list = get_imlist("C:\\Users\\Simon\\Pictures\\Skærmbilleder")

    #define dimensions that images will get resized to and calculate data array
    picture_shape = [256, 256]
    pixel_number = 256 * 256
    data_arr = images_to_data_arr(im_list, picture_shape)
    no_pictures = np.size(data_arr[:,0])

    #extract features from array
    features_arr, features_no = get_features(data_arr, picture_shape)

    #print average values for red, green, blue channels for each pictures
    RGB = ['red', 'green', 'blue']
    for color in range(len(RGB)):
        print(f'{RGB[color]} av = ',  features_arr[:,int((color+1) * features_no ) - 2], \
            "  std = ", features_arr[:,int((color+1) * features_no ) - 1])
def from_zip_to_image_list():

    imlist = get_imlist(folder_path)
   # imlist[0].show()

    im_list = get_imlist_from_zip(zip_path)

    flat_list, shape = images_to_data_list(im_list)
  

    make_histogram_from_data(flat_list[0], shape[0])
    plt.show()

    flat_array = images_to_data_arr(im_list,[256,256])
    print(flat_array.shape)
    make_histogram_from_data(flat_array[0],[256,256])
    plt.title('array approach')
    plt.show()
def deforestation_features_examination():

    # In this section, we will compare 6 images of various degrees of deforestation. We wil compare their feautures and look for obvious
    # feature dependence on the deforestation degree

    #load images. They are ordered from least to most deforestation
    im_list = get_imlist_from_zip(zip_path)
    N_pictures = len(im_list)

    #construct data_arr from images, resizing to pixel_shape
    pixel_shape = [340, 340]
    im_arr = images_to_data_arr(im_list, pixel_shape, color = True)

    if 0:
        #construct grey_scale_image array
        grey_arr = np.empty([N_pictures,pixel_shape[0] * pixel_shape[1]])
        for image in range(N_pictures):
            im = Image.open(im_list[image]).convert('L').resize((pixel_shape[0],pixel_shape[1]))
            grey_arr[image,:] = np.array(im).flatten()
    grey_arr = convert_to_grey_scale(im_list, pixel_shape)

    #construct feature matrix    
    im_features, N_features = get_features(im_arr, pixel_shape)


    grey_features, _ = get_features(grey_arr, pixel_shape, color = False)
    RGB = ['r', 'g', 'b']

    #plot average and std values for each image
    plt.figure()
    #plot for grey:
    grey_av = grey_features[:, N_features-2]
    grey_std = grey_features[:, N_features-1 ]

    plt.plot(np.linspace(1,N_pictures,N_pictures ), grey_av, f'kx-', label = 'av')
    plt.plot(np.linspace(1,N_pictures,N_pictures ), grey_std, f'ko-', label = 'std')

    for color in range(len(RGB)):
        color_av = im_features[:,int( (color+1) * (N_features) - 2 )]
        color_std = im_features[:,int( (color+1) * (N_features) - 1 )]
        #print(f'av {color}', color_av, "\n \n")
        #print(f'std {color}', color_std, "\n \n")

        plt.plot(np.linspace(0,N_pictures-1,N_pictures ), color_av, f'{RGB[color]}x-', label = 'av')
        plt.plot(np.linspace(0,N_pictures-1,N_pictures ), color_std, f'{RGB[color]}o-', label = 'std')
    plt.legend()
    plt.title('Average and std for each colorchannel for various degrees of deforestation')
    plt.xlabel('Deforestation degree from small to large')
    plt.show(block = False)

    # plot color histogram for each image
    # plot for grey
    for image in range(N_pictures):
       make_histogram_from_data( im_arr[image], pixel_shape)
       plt.hist(grey_arr[image,:], color = 'grey', bins = 256)
       plt.title(f'histogram for image {image}')
       #plt.show(block = False)
    
    plt.show()
    pass

def main():
    [explore, convert, extract, open_zip, deforestation] = [False, False, False, False, True]

    if explore:
        explore_basic_image_handling()
    if convert:
        convert_image_folder_to_list_and_array()
    if extract:
        extract_features_from_images()
    if open_zip:
        from_zip_to_image_list()
    if deforestation:
        deforestation_features_examination()

if __name__ == "__main__":
    main()


