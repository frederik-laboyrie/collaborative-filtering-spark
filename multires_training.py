from data_preprocessing import load_standardized_multires
from multires_CNN import multires_CNN
import sys

def main():
	multires_data, labels = load_standardized_multires()
	model = multires_CNN(int(sys.argv[1]),int(sys.argv[2]),multires_data)
	full = multires_data[0]
	med = multires_data[1]
	low = multires_data[2]
	model.fit([full,med,low],labels, epochs = 10)

if __name__ == '__main__':
	main()