import matplotlib.pyplot as plt

class MultisliceViewer3():
    # constructor
    def __init__(self):
        # nothing to initialize
        pass

    # here, the scrolling is with mousewheel
    def run_3D_viewer(self,volume_dat,volume_seg,volume_cla):
        fig, axarr = plt.subplots(1,3)
        axarr[0].volume_dat = volume_dat
        axarr[0].volume_seg = volume_seg
        axarr[0].volume_cla = volume_cla
        axarr[0].index = volume_dat.shape[2]//2 #(x,y,z) configuration assumed, x-y plotted
        axarr[0].imshow(volume_dat[:,:,axarr[0].index])
        axarr[1].imshow(volume_seg[:,:,axarr[0].index])
        axarr[2].imshow(volume_cla[:,:,axarr[0].index])
        fig.canvas.mpl_connect('scroll_event',self.on_scroll)
        plt.show()
        
    def on_scroll(self,event):
        fig = event.canvas.figure #select the fig on which the event happened???
        axarr = fig.axes # what does this get???
        if event.button == 'up':
            self.prev_slice(axarr)
            #print('scroll up')
        else:
            self.next_slice(axarr)
            #print('scroll down')
        fig.canvas.draw()

    def prev_slice(self,axarr):
        # Go to previous slice
        volume_dat = axarr[0].volume_dat
        volume_seg = axarr[0].volume_seg
        volume_cla = axarr[0].volume_cla
        axarr[0].index = (axarr[0].index - 1) % volume_dat.shape[2] #warp around using %
        axarr[0].images[0].set_array(volume_dat[:,:,axarr[0].index])
        axarr[1].images[0].set_array(volume_seg[:,:,axarr[0].index])
        axarr[2].images[0].set_array(volume_cla[:,:,axarr[0].index])
        print('slice num: %d'%(axarr[0].index))

    def next_slice(self,axarr):
        # Go to next slice
        volume_dat = axarr[0].volume_dat
        volume_seg = axarr[0].volume_seg
        volume_cla = axarr[0].volume_cla
        axarr[0].index = (axarr[0].index + 1) % volume_dat.shape[2]
        axarr[0].images[0].set_array(volume_dat[:,:,axarr[0].index])
        axarr[1].images[0].set_array(volume_seg[:,:,axarr[0].index])
        axarr[2].images[0].set_array(volume_cla[:,:,axarr[0].index])
        print('slice num: %d'%(axarr[0].index))