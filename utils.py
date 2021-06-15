import matplotlib
import matplotlib.pyplot as plt

def use_backend(backend_name='Agg'):
    matplotlib.use(backend_name)

def create_graph(x_data, y_data,  save_path, fmt='o-k'):
    try:
        plt.figure()
        plt.plot(x_data, y_data, fmt)
        plt.savefig(save_path)
    except Exception as e:
        print(e)

def validate_crop_size(crop_size):
    if len(crop_size) != 3:
        raise ValueError("Crop size must be 3 int values.")
    for c in crop_size:
        if c%32 != 0:
            raise ValueError("Crop size must be a multiple of 32.")