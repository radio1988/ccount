def read_czi(fname, Format="2019"):
    '''
    input: fname of czi file
    output: 2d numpy array, uint8 for 2019
    assuming input czi format (n, 1, :, :, 1)
    e.g. (4, 1, 70759, 65864, 1)

    '''
    from czifile import CziFile
    
    fname=str(fname)
    Format=str(Format)
    print('read_czi:', fname)
    if fname.endswith('czi'):
        with CziFile(fname) as czi:
            image_arrays = czi.asarray()  # 129s, Current memory usage is 735.235163MB; Peak was 40143.710599MB
            print(image_arrays.shape)
    elif fname.endswith('czi.gz'):
        raise Exception("todo")
    else:
        raise Exception("input czi/czi.gz file type error\n")

    return image_arrays


def parse_image_arrays (image_arrays, i = 0,  Format = '2019'):
    '''
    image_arrays: output from read_czi
    i: index of [0,1,2,3], only this image will be parsed
    Format: e.g. 2019
    '''
    import numpy as np
    import gc

    i = int(i)
    Format = str(Format).strip()
    if Format == "2018":
        # reading (need 38 GB RAM) todo: use uint8 if possible
        image = image_arrays[0, 1, 0, 0, :, :, 0]  # real image
        return image 
    elif Format == "2019":        
        # todo: Find Box faster by https://kite.com/python/docs/PIL.Image.Image.getbbox  

        image = image_arrays[i, 0, :,  :, 0] # 0s
        nz_image = np.nonzero(image)  # process_time(),36s, most time taken here, 1.4GB RAM with tracemalloc
        nz0 = np.unique(nz_image[0]) # 1.5s
        nz1 = np.unique(nz_image[1]) # 2.4s
        del nz_image
        n = gc.collect()
        if len(nz0) < 2 or len(nz1) < 2: 
            import warnings
            warnings.warn('area', i, 'is blank')
            return False
        image = image[min(nz0):max(nz0), min(nz1):max(nz1)]  # 0s
        return image
        
        # if concatenation:
        #     # padding
        #     heights = [x.shape[0] for x in lst]
        #     widths = [x.shape[1] for x in lst]
        #     max(heights)
        #     max(widths)
        #     for (i,image) in enumerate(lst):
        #         print(image.shape, i)
        #         pad_h = max(heights) - image.shape[0]
        #         pad_w = max(widths) - image.shape[1]
        #         lst[i] = np.pad(image, [[0,pad_h],[0,pad_w]], "constant")
                
        #     # concat: use a long wide image instead to adjust for unknown number of scanns
        #     image = np.hstack(lst)
        #     print("shape of whole picture {}: {}\n".format(fname, image.shape))
        #     return image
        # else:
        #     # return a list of single are images
        #     return lst #[image0, image1, image2 ..]
    else:
        raise Exception("image format error:", Format, "\n")
        return None