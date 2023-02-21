from PIL import Image

def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    Im = Image.open(filepath)
    Im.show()
    return width, height

print(get_num_pixels("/home/bjornel/market1501/SegmentedMarket1501/1314/uv_maps/1314_c3s3_034228_00_texture.jpg"))