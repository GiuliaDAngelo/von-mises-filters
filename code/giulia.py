import sinabs
import sinabs.layers as sl
import torch
import torch.nn as nn
import numpy as np
import tonic
import matplotlib.pyplot as plt
import torchvision
import numpy.lib.recfunctions as rf
from bimvee.importRpgDvsRos import importRpgDvsRos
from bimvee.importIitYarp import importIitYarp
from bimvee.importProph import importProph
from torchvision.transforms import InterpolationMode
from skimage.transform import rescale, resize, downscale_local_mean




def run(angle_shift, fltr_resize_perc, time_wnd_frames, rec, polarity, batch_frames, new_size):
    angles = range(0, 360, angle_shift)
    filters = []
    for i in angles:
        filter = np.load(f"VMfilters/{i}_grad.npy")
        filter = rescale(filter, fltr_resize_perc, anti_aliasing=False)
        filters.append(filter)
    filters = torch.tensor(np.stack(filters).astype(np.float32))

    #plt filters
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle(f'VM filters size ({filter.shape[0]},{filter.shape[0]})', fontsize=16)

    #show filters
    show_fltFLAG = True
    if show_fltFLAG:
        for i in range(8):
            if i < 4:
                axes[0, i].set_title(f"{angles[i]} grad")
                axes[0, i].imshow(filters[i])
            else:
                axes[1, i-4].set_title(f"{angles[i]} grad")
                axes[1, i-4].imshow(filters[i])
        plt.show()

    # define our single layer network and load the filters
    net = nn.Sequential(
        nn.Conv2d(1, filters.shape[0], filters.shape[1], bias=False),
        sl.IAF()
    )
    net[0].weight.data = filters.unsqueeze(1)

    # find out maximum x and y
    max_x = rec['x'].max().astype(int)
    max_y = rec['y'].max().astype(int)
    # use single polarity
    rec['p'] = polarity
    sensor_size = (max_x+1, max_y+1, 1)
    print(f"sensor size is {sensor_size}")


    # We have to convert the raw events into frames so that we can feed those to our network
    # We use a library called tonic for that https://tonic.readthedocs.io/en/latest/ as well as torchvision
    # We use a 20ms (20000us) time window to bin events into frames and crop the center of the frame
    transforms = torchvision.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_wnd_frames),
        torch.tensor,
        torchvision.transforms.CenterCrop((sensor_size[1]-1, sensor_size[0]-1)),
    ])

    frames = transforms(rec)
    # here
    frames[0].resize_(new_size)
    # this leaves us with some 337 time steps.
    frames.shape
    plt.figure()
    plt.imshow(frames[10, 0])
    # now we feed the data to our network! Because my computer has little memory, I only feed 10 specific time steps
    with torch.no_grad():
        output = net(frames[batch_frames[0]:batch_frames[1]].float())
    output.shape
    # in the end we can plot the


    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    for i in range(8):
        if i < 4:
            axes[0, i].set_title(f"{angles[i]} grad")
            axes[0, i].imshow(output[0, i])
        else:
            axes[1, i-4].set_title(f"{angles[i]} grad")
            axes[1, i-4].imshow(output[0, i])
    plt.show()
    print('end')


def raw_data(filePathOrName):
    events = importProph(
        filePathOrName=filePathOrName)

    camera_events = 'ch0'

    xs = events['data'][camera_events]['dvs']['x']
    ys = events['data'][camera_events]['dvs']['y']
    ts = events['data'][camera_events]['dvs']['ts']
    ts *= 1e6  # convert time from seconds to microseconds
    pol = events['data'][camera_events]['dvs']['pol']
    pol = np.array(pol, dtype=bool)
    recording = np.column_stack((xs, ys, pol, ts))
    rec = rf.unstructured_to_structured(recording,
                                        dtype=np.dtype(
                                            [('x', np.int16), ('y', np.int16), ('p', bool), ('t', int)]))
    return rec


def npy_data(filePathOrName):
    recording = np.load(filePathOrName)
    recording[:, 3] *= 1e6  # convert time from seconds to microseconds
    rec = rf.unstructured_to_structured(recording,
                                        dtype=np.dtype(
                                            [('x', np.int16), ('y', np.int16), ('p', bool), ('t', int)]))
    return rec


def yarp_data(filePathOrName):
    events = importIitYarp(
        filePathOrName=filePathOrName,
        codec='20bit')
    # events = importIitYarp(filePathOrName='/Tesi/Datasets/icub_datasets/square/data')
    camera_events = 'right'

    xs = events['data'][camera_events]['dvs']['x']
    ys = events['data'][camera_events]['dvs']['y']
    ts = events['data'][camera_events]['dvs']['ts']
    ts *= 1e6  # convert time from seconds to microseconds
    pol = events['data'][camera_events]['dvs']['pol']
    pol = np.array(pol, dtype=bool)
    recording = np.column_stack((xs, ys, pol, ts))
    rec = rf.unstructured_to_structured(recording,
                                        dtype=np.dtype(
                                            [('x', np.int16), ('y', np.int16), ('p', bool), ('t', int)]))
    return rec

if __name__ == '__main__':


    # load all the filters and stack them to a 3d array of (filter number, width, height)
    fltr_resize_perc = 3
    angle_shift = 45
    time_wnd_frames = 20000 #us
    polarity = 0
    batch_frames = (100, 110)
    new_size = (200, 300)
    ############################################
    ################ data ######################
    ############################################

    RawDataFLAG = False
    npy_dataFLAG = True
    YarpDataFLAG = False

    if RawDataFLAG:
        filePathOrName='/home/giuliadangelo/workspace/code/von-mises-filters/data/twoobjects.raw'
        rec = raw_data(filePathOrName)
        run(angle_shift, fltr_resize_perc, time_wnd_frames, rec, polarity, batch_frames, new_size)

    elif npy_dataFLAG:
        # load the recording and convert it to a structured numpy array
        filePathOrName = "/home/giuliadangelo/workspace/code/von-mises-filters/data/twoobjects.npy"
        rec = npy_data(filePathOrName)
        run(angle_shift, fltr_resize_perc, time_wnd_frames, rec, polarity, batch_frames, new_size)


    elif YarpDataFLAG:
        filePathOrName = '/home/giuliadangelo/workspace/data/DATASETs/IROS_attention/paddle/moving_paddle/ATIS/data'
        rec = yarp_data(filePathOrName)
        run(angle_shift, fltr_resize_perc, time_wnd_frames, rec, polarity,batch_frames, new_size)