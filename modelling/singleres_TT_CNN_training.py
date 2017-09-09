from data_preprocessing import load_standardized_singleres
from singleres_TT_CNN import *
import sys

tt_input_shape=[10, 12, 12, 10]
tt_output_shape=[10, 12, 12, 10]
tt_ranks=[1, 3, 3, 3, 1]

def main():
    xs, ys = load_standardized_singleres()
    model = singleres_TT_CNN(int(sys.argv[1]),int(sys.argv[2]),
                             tt_input_shape, tt_output_shape,
                             tt_ranks, xs)
    model.fit(xs, ys, epochs=10)

if __name__ == '__main__':
    main()