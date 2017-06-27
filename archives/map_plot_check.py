from __future__ import print_function
import numpy as np, avicaching_data as ad
import matplotlib.pyplot as plt

J, T = 116, 173

def find_idx_of_nearest_el(array, value):
    """
    Helper function to plot_predicted_map(). Finds the index of the element in
    array closest to value
    """
    return (np.abs(array - value)).argmin()

def plot_predicted_map(lat_long, point_info, plot_offset=0.05):
    """
    Plots the a scatter plot of point_info on the map specified by the lats
    and longs and saves to a file
    """
    # find the dimensions of the plot
    lati = lat_long[:, 0]
    longi = lat_long[:, 1]
    lo_min, lo_max = min(longi) - plot_offset, max(longi) + plot_offset
    la_min, la_max = min(lati) - plot_offset, max(lati) + plot_offset
    plot_width = max(lo_max - lo_min, la_max - la_min)
    lo_max = lo_min + plot_width
    la_max = la_min + plot_width

    lo_range = np.linspace(lo_min, lo_max, num=J + 10, retstep=True)
    la_range = np.linspace(la_min, la_max, num=J + 10, retstep=True)

    lo, la = np.meshgrid(lo_range[0], la_range[0])

    z = np.zeros([J + 10, J + 10])
    for k in xrange(J):
        # find lati[k] in the mesh, longi[k] in the mesh
        lo_k_mesh = find_idx_of_nearest_el(lo[0], longi[k])
        la_k_mesh = find_idx_of_nearest_el(la[:, 0], lati[k])
        z[lo_k_mesh][la_k_mesh] = point_info[k]

    plt.figure(1)
    plt.pcolormesh(lo, la, z, cmap=plt.cm.get_cmap('Greys'), vmin=0.0, vmax=0.1)
    plt.axis([lo.min(), lo.max(), la.min(), la.max()])
    plt.colorbar()
    plt.show()
    plt.close()

X, Y, R = ad.read_XYR_file("./data/density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", J, T)

Y = np.mean(Y, axis=0)
plot_predicted_map(ad.read_lat_long_from_Ffile(
    "./data/loc_feature_with_avicaching_combined.csv", J),
            Y)