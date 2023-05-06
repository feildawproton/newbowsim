from .Materials import Hickory

HickoryLayer = {
"Material"	: Hickory,
"Length"	: [ 0.0,  0.142857,  0.2,  1.0],	#frac	
"Height"	: [38.1,      38.1, 12.7, 12.7]	#mm
}

print("don't forget to use layers")

Layers = []

Layers.append(HickoryLayer)
