from data_preprocessing import load_data
from vanilla_CNN import *
import sys

#subsample remains here during testing

def main():
	x, y = load_data()
	xs, ys = subsample(x,y,500)
	model = vanilla_CNN(int(sys.argv[1]),int(sys.argv[2]),xs)	
	xs = reshape(xs)
	model.fit(xs,ys)

if __name__ == '__main__':
	main()