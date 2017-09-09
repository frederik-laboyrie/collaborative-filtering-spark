from data_preprocessing import load_standardized_multires
from multires_TT_CNN import *
import sys

#for filters=6 and kernel_size = 5
tt_input_shape = [10,26,19,6]
tt_output_shape = [10,26,19,6]
tt_ranks = [1,3,3,3,1]

def main():
    multires_data, labels = load_standardized_multires()
    model = multires_TT_CNN(int(sys.argv[1]), int(sys.argv[2]), 
                            tt_input_shape, tt_output_shape,
                            tt_ranks, multires_data)
    full = multires_data[0]
    med = multires_data[1]
    low = multires_data[2]
    model.fit([full,med,low],labels, epochs = 10)

if __name__ == '__main__':
    main()
