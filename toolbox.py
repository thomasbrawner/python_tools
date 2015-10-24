
import numpy as np 
from warnings import warn

def n_largest(arr, n, sorted=False):
    '''
    Return the indices of the n largest elements in arr. 
    If sorted, complete sort the array and return the 
    indices for the elements ordered least to greatest. 
    '''
    if sorted:
        return np.argsort(arr)[-5:]
    else:
		return np.argpartition(arr, -n)[-n:]


def round_down_any(arr, base):
    '''
    Round values of arr down to nearest base (integer)
    '''
    if not type(base) == int:
    	warn('base changed to type integer')
    	base = int(base)
    return (arr / base) * base


def scale_convert(arr, new_scale, round=False):
	'''
	Convert the values of arr to the range of values 
	in new_scale (tuple). 
	'''
	old_range = (arr.max() - arr.min())  
	new_range = (new_scale[1] - new_scale[0])  
	new_array = (((arr - arr.min()) * new_range) / old_range) + 1
	if round:
		return np.round(new_array)
	return new_array


if __name__ == '__main__':
	
	arr = np.random.randint(50, size=50)

	# 5 largest elements in arr
	largest_idx = n_largest(arr, n=5)
	largest_idx_sorted = n_largest(arr, n=5, sorted=True)
	print arr[largest_idx]
	print arr[largest_idx_sorted]

	# round down to base 3
	print round_down_any(arr, base=3.0)

	# change scale of arr to (0, 99)
	print scale_convert(arr, (0, 99), round=True)

