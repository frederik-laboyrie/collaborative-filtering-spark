from data_preprocessing import load_standardized_singleres
from vanilla_CNN import *
import sys

def main():
	xs, ys = load_standardized_singleres()
	model = vanilla_CNN(int(sys.argv[1]),int(sys.argv[2]),xs)	
	model.fit(xs,ys)

if __name__ == '__main__':
	main()