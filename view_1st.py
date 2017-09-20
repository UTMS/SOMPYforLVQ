import matplotlib
from matplotlib import pyplot as plt
import numpy as np

class View(object):
    def __init__(self, width, height, title, show_axis=True, packed=True,
                 text_size=2.8, show_text=True, col_size=6, *args, **kwargs):
        self.width = width
        self.height = height
        self.title = title
        self.show_axis = show_axis
        self.packed = packed
        self.text_size = text_size
        self.show_text = show_text
        self.col_size = col_size

    def prepare(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def show(self, *args, **kwrags):
        raise NotImplementedError()


class MatplotView(View):

    def __init__(self, width, height, title, show_axis=True, packed=True,
                 text_size=2.8, show_text=True, col_size=6, *args, **kwargs):
        super(MatplotView, self).__init__(width, height, title, show_axis,
                                          packed, text_size, show_text,
                                          col_size, *args, **kwargs)
        self._fig = None

    def __del__(self):
        self._close_fig()

    def _close_fig(self):
        if self._fig:
            plt.close(self._fig)

    def prepare(self, *args, **kwargs):
        self._close_fig()
        # close previous windows
        self._fig = plt.figure(figsize=(self.width, self.height))
        # creat a figure of (self.width*self.height)
        plt.title(self.title)
        plt.axis('off')
        plt.rc('font', **{'size': self.text_size})

    def save(self, filename, transparent=False, bbox_inches='tight', dpi=400):
        self._fig.savefig(filename, transparent=transparent, dpi=dpi,
                          bbox_inches=bbox_inches)

    def show(self, *args, **kwrags):
        raise NotImplementedError()




class MapView(MatplotView):

    def _calculate_figure_params(self, som, which_dim, col_sz):
        codebook = som._normalizer.denormalize_by(som.data_raw,
                                                  som.codebook.matrix)

        indtoshow, sV, sH = None, None, None

        if which_dim == 'all':
            dim = len(som.codebook.matrix)
            row_sz = np.ceil(float(dim) / col_sz)
            # col_sz is 4, just number of figures in a row
            #msz_row, msz_col = som.codebook.mapsize
            #ratio_hitmap = msz_row / float(msz_col)
            #ratio of the hight and width of the figure
            #ratio_fig = row_sz / float(col_sz)
            #ratio_fig is 1/2
            indtoshow = np.arange(0, dim).T
            #sH, sV = 6,6
            
        elif type(which_dim) == int:
            dim = 1
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            indtoshow = np.zeros(1)
            indtoshow[0] = int(which_dim)
            #sH, sV = 6,6

        elif type(which_dim) == list:
            max_dim = codebook.shape[1]
            dim = len(which_dim)
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.asarray(which_dim).T
            width, height = 36, 36*ratio_fig*ratio_hitmap

        no_row_in_plot = dim / col_sz + 1  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = col_sz

        axis_num = 0

        #width = sH
        #height = sV
        width = self.width
        height = self.height
        

        return (width, height, indtoshow, no_row_in_plot, no_col_in_plot,
                axis_num)


class View2DN(MapView):

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None, desnormalize=False):
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        #self.prepare()

        if not desnormalize:
            A = som.codebook.matrix
            map_row,map_col = som.codebook.mapsize
            B = A.reshape(map_row,map_col,self.width*self.height)
            C = B.transpose((1,0,2))
            codebook = C.reshape(map_row*map_col,self.width*self.height)
        else:
            codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)

        if which_dim == 'all':
            names = som._component_names[0]
        elif type(which_dim) == int:
            names = [som._component_names[0][which_dim]]
        elif type(which_dim) == list:
            names = som._component_names[0][which_dim]


        while axis_num < len(indtoshow):
            axis_num += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axis_num)
            ind = int(indtoshow[axis_num-1])

            min_color_scale = np.mean(codebook[ind,:].flatten()) - 1 * np.std(codebook[ind,:].flatten())
            max_color_scale = np.mean(codebook[ind,:].flatten()) + 1 * np.std(codebook[ind,:].flatten())
            min_color_scale = min_color_scale if min_color_scale >= min(codebook[ind,:].flatten()) else \
                min(codebook[ind,:].flatten())
            max_color_scale = max_color_scale if max_color_scale <= max(codebook[ind,:].flatten()) else \
                max(codebook[ind,:].flatten())
            norm = matplotlib.colors.Normalize(vmin=min_color_scale, vmax=max_color_scale, clip=True)

            mp = codebook[ind,:].reshape(self.width,self.height)
            pl = plt.pcolor(mp[::-1], cmap=cmap)
            plt.axis([0, self.width, 0, self.height])
            plt.title(axis_num - 1)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            #plt.ax.set_cmap(cmap)
            #plt.colorbar(pl)
            #print(min_color_scale, max_color_scale)
        plt.show()
                
class View2DPacked(MapView):

    def _set_axis(self, ax, msz0, msz1):
        plt.axis([0, msz0, 0, msz1])
        plt.axis('off')
        ax.axis('off')

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None):
        if col_sz is None:
            col_sz = 6
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        codebook = som.codebook.matrix

        cmap = cmap or plt.cm.get_cmap('RdYlBu_r')
        msz0, msz1 = som.codebook.mapsize
        print(msz0)
        compname = som.component_names
        if what == 'codebook':
            h = .1
            w = .1
            self.width = no_col_in_plot*2.5*(1+w)
            self.height = no_row_in_plot*2.5*(1+h)
            self.prepare()

            while axis_num < len(indtoshow):
                axis_num += 1
                ax = self._fig.add_subplot(no_row_in_plot, no_col_in_plot,
                                           axis_num)
                ax.axis('off')
                ind = int(indtoshow[axis_num-1])
                mp = codebook[:, ind].reshape(msz0, msz1)
                plt.imshow(mp[::-1], norm=None, cmap=cmap)
                self._set_axis(ax, msz0, msz1)

                if self.show_text is True:
                    plt.title(compname[0][ind])
                    font = {'size': self.text_size}
                    plt.rc('font', **font)

        if what == 'cluster':
            try:
                codebook = getattr(som, 'cluster_labels')
            except:
                codebook = som.cluster()

            h = .2
            w = .001
            self.width = msz0/2
            self.height = msz1/2
            self.prepare()

            ax = self._fig.add_subplot(1, 1, 1)
            mp = codebook[:].reshape(msz0, msz1)
            plt.imshow(mp[::-1], cmap=cmap)

            self._set_axis(ax, msz0, msz1)

        plt.subplots_adjust(hspace=h, wspace=w)


class View1D(MapView):

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None):
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        self.prepare()

        codebook = som.codebook.matrix

        while axis_num < len(indtoshow):
            axis_num += 1
            plt.subplot(no_row_in_plot, no_col_in_plot, axis_num)
            ind = int(indtoshow[axis_num-1])
            mp = codebook[:, ind]
            plt.plot(mp, '-k', linewidth=0.8)

        plt.show()
