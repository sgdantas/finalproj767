

def plot_graph(x_value, y_values, legend_names, x_axis_title, y_axis_title, filename_to_save, plot_points = None, saveFigures=True,plot_graph_=False):
    import matplotlib.pyplot as plt 
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)

    if(plot_points):
        for i in range(len(y_values)):
            ax.plot(x_value,
                    y_values[i],
                    plot_points[i],
                    label = legend_names[i])
    else:
        for i in range(len(y_values)):
            ax.plot(x_value,
                    y_values[i],
                    label = legend_names[i])
    ax.set_xlabel (x_axis_title)
    ax.set_ylabel (y_axis_title)

    plt.legend(loc = 'upper left')
    if(saveFigures):
        plt.savefig(filename_to_save)
    if (plot_graph_):
        plt.show()    
    plt.clf()
    
    return

def plot_graph2(x_value, y_values, legend_names, x_axis_title, y_axis_title, filename_to_save, saveFigures=True,plot_graph_=False):
    import matplotlib.pyplot as plt 
    import numpy as np

    y_std = np.std(y_values, axis=1)
    y_mean = np.mean(y_values, axis=1)
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
   
   
    #for i in range(len(y_values)):
    ax.plot(x_value,
            y_mean,
            label = legend_names)#[:,i],
            #label = legend_names[i])
    ax.fill_between(x_value,
                (y_mean - y_std),
                (y_mean + y_std),
                alpha=0.2)
    ax.set_xlabel (x_axis_title)
    ax.set_ylabel (y_axis_title)

    plt.legend(loc = 'upper left')
    if(saveFigures):
        plt.savefig(filename_to_save)
    if (plot_graph_):
        plt.show()    
    plt.clf()
    
    return


def plot_graph3(x_value, y_values, legend_names, x_axis_title, y_axis_title, filename_to_save, saveFigures=True,plot_graph_=False):
    import matplotlib.pyplot as plt 
    import numpy as np

    y_std = np.std(y_values[0], axis=1)
    y_mean = np.mean(y_values[0], axis=1)
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
   
   
    #for i in range(len(y_values)):
    ax.plot(x_value,
            y_mean,
            label = legend_names[0],
            color='b')#[:,i],
            #label = legend_names[i])
    ax.fill_between(x_value,
                (y_mean - y_std),
                (y_mean + y_std),
                alpha=0.2)
    ax.set_xlabel (x_axis_title)
    ax.set_ylabel (y_axis_title)
    
    ax.plot(x_value,
            y_values[1],
            label = legend_names[1],
            color='r')#[:,i],
            #label = legend_names[i])

    ax.set_ylim([7,22])
    plt.legend(loc = 'upper left')
    if(saveFigures):
        plt.savefig(filename_to_save)
    if (plot_graph_):
        plt.show()    
    plt.clf()
    
    return