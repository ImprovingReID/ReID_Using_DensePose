import os
from matplotlib import pyplot as plt
import tqdm
import createUV as cuv
import cv2
from pathlib import Path

# def _crop(im, bbox):
#     ih, iw, _ = im.shape
#     b = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
#     x, y, w, h = [int(v * r) for v, r in zip(b, [iw, ih, iw, ih])]
#     return im[y : y + h, x : x + w, :]

# def _stuff(im, idx, output_path):
    
#     if texs == None or len(texs) == 0:
#         print("warning: no detection in crop")
#         return None 
    
#     p = f"{output_path}/uv_maps/{idx:06}.jpg"
#     texs[0]["texture"].save_to_file(p)
#     cv2.imwrite(f"{output_path}/crops/{idx:06}.jpg", im)
#     return p
    
# def parse_train_json(train, data_dir, output_dir):
#     for person in tqdm.tqdm(train):
#         person_id = person["person_id"].replace("/", "-")       
        
#         person_dir = output_dir / person_id
        
#         if not os.path.isdir(person_dir):
#             Path(person_dir + '/' + "crops").mkdir(exist_ok=True, parents=True)  
#             Path(person_dir + '/' + "pkl_files").mkdir(exist_ok=True, parents=True)  
#             Path(person_dir + '/' + "uv_maps").mkdir(exist_ok=True, parents=True)    
        
#         for idx, sample in enumerate(person["samples"]):
#             im = cv2.imread(str(data_dir / sample["image_path"]))
#             bbox = [sample[k] for k in ("xtl", "ytl", "xbr", "ybr")]
#             crop = _crop(im, bbox)          
            
#             p = _stuff(uv_mapper, crop, idx, person_dir)
#             if p is not None:
#                 sample["uv_image_path"] = p
    


def denseSegmentor(dataset_dir,output_dir, crop = False):
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for image in os.listdir(dataset_dir):
        f = os.path.join(dataset_dir, image)
        if os.path.isfile(f):

            person_id = image[0:4]
            person_dir = output_dir + '/' + person_id
            # if os.path.isdir(person_dir):
                # print(f"skipping {person_id} because it already exists")       
                
            # else:
            if not os.path.isdir(person_dir):
                Path(person_dir + '/' + "crops").mkdir(exist_ok=True, parents=True)  
                Path(person_dir + '/' + "pkl_files").mkdir(exist_ok=True, parents=True)  
                Path(person_dir + '/' + "uv_maps").mkdir(exist_ok=True, parents=True)

            # save crop
            crop = cv2.imread(f)
            cv2.imwrite(person_dir + '/' + "crops" + '/' + image, crop)  
            
            #create and save pkl files
            pkl_file , pkl_dir = cuv.create_pkl(image,directory = person_dir + '/' + "crops"  ,savedir = person_dir + '/' + "pkl_files/") 
            
            #create and save uv maps
            cuv.texture(pkl_file , pkl_dir,directory = person_dir + '/' + "crops"  ,savedir = person_dir + '/' + "uv_maps/")


def market1501():
    output_dir= 'data/SegmentedMarket1501'
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    images_dir = 'data/Market-1501-v15.09.15/gt_bbox'
    for image in os.listdir(images_dir):
        f = os.path.join(images_dir, image)
        if os.path.isfile(f):

            person_id = image[0:4]
            person_dir = output_dir + '/' + person_id
            # if os.path.isdir(person_dir):
                # print(f"skipping {person_id} because it already exists")       
                
            # else:
            if not os.path.isdir(person_dir):
                Path(person_dir + '/' + "crops").mkdir(exist_ok=True, parents=True)  
                Path(person_dir + '/' + "pkl_files").mkdir(exist_ok=True, parents=True)  
                Path(person_dir + '/' + "uv_maps").mkdir(exist_ok=True, parents=True)

            # save crop
            crop = cv2.imread(f)
            cv2.imwrite(person_dir + '/' + "crops" + '/' + image, crop)  
            
            #create and save pkl files
            pkl_file , pkl_dir = cuv.create_pkl(image,directory = person_dir + '/' + "crops"  ,savedir = person_dir + '/' + "pkl_files/") 
            
            #create and save uv maps
            cuv.texture(pkl_file , pkl_dir,directory = person_dir + '/' + "crops"  ,savedir = person_dir + '/' + "uv_maps/")

if __name__ == '__main__':    
    market1501()