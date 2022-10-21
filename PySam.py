
def resize_scatterplot(s,s1=10,s2=100,log=False,extrema=[]):
    #----------------------------------------------------------------------
    # Rescale the size of a scatter plot points
    #
    # Parameters:  s : array_like
    #                  Array to be scaled.
    #              s1 : float
    #                   Minimum value for the marker size
    #              s2 : float
    #                   Maximum value for the marker size
    #              log : bool
    #                    if False (default), the marker size scales as s
    #                    if True, the marker size scales as log(s)
    #              extrema : 2-elements array, optional
    #                        Gives the range of s to rescale
    # Returns:  array
    #           Scaled markers size
    #----------------------------------------------------------------------

    import numpy as np
    s = np.asarray(s)
    if extrema==[]: r1,r2 = min(s),max(s)
    else:
        try: r1,r2 = min(extrema),max(extrema)
        except: raise TypeError("extrema must be a list of two values")
    if not log: return s1+(s-r1)*(s2-s1)/(r2-r1)
    else: return resize_scatterplot(np.log10(s),s1=s1,s2=s2,extrema=[np.log10(r1),np.log10(r2)])

#######################################################################

def get_colours_for_plot(Ncolors,cmap='gist_rainbow'):
    #----------------------------------------------------------------------
    # Generate a color palette
    #
    # Parameters:  Ncolors : int
    #                        Number of colors.
    #              cmap : string
    #                     Colormap from the library pylab.
    # Returns:  array
    #           Array of RGBA tuples.
    #
    # Note:  a warning will be raised unless the following lines are
    #         included in the main
    #        >>> from matplotlib.axes._axes import _log as matplotlib_axes_logger
    #        >>> matplotlib_axes_logger.setLevel('ERROR')
    #----------------------------------------------------------------------

    import pylab
    cm = pylab.get_cmap(cmap)
    col=[cm(1.*i/Ncolors) for i in range(Ncolors)] # color will now be an RGBA tuple

    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')
    return col