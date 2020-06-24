import sys
from neural_clustering import NeuralClustering

def main() :
    clustering_index = NeuralClustering(path='../../../feature_value/data/all/bite.csv')
    clustering_index.test()
    return True

if __name__ == "__main__":
    args = sys.argv
    main()