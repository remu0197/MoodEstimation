import os  # for makedirs
import homcloud.interface as hc  # HomCloud 
import homcloud.paraview_interface as pv # HomCloud <-> paraview interface
import numpy as np  # Numerical array library
from tqdm import tqdm_notebook as tqdm  # For progressbar
import matplotlib.pyplot as plt  # Plotting
import sklearn.linear_model as lm  # Machine learning
from sklearn.decomposition import PCA  # for PCA
from sklearn.model_selection import train_test_split
import csv

def main() :

    path = '../data/half/bite.csv'

    labels = []
    data = []

    with open(path) as f :
        reader = csv.reader(f)
        header = next(reader)

        for row in reader :
            labels.append(row[2])
            data.append(row[3:])

    Y = np.array(labels)
    X = np.array(data)

    hc.PDList.from_alpha_filtration(X, save_boundary_map=True, save_to="pd/result.idiagram".format(2))
    pds = hs.PDList("pd/result.idiagram".format(2)).dth_diagram(2)
    pds[0].histogram(x_range=(0, 0.03)).plot(colorbar={"type": "log"})

    print(labels)
    # print(datas)

    return True

if __name__ == "__main__":
    main()
