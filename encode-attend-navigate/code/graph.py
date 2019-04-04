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