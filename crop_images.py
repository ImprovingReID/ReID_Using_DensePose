import shutil
import cv2

import tqdm
import createUV

def _crop(im, bbox):
    ih, iw, _ = im.shape
    b = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    x, y, w, h = [int(v * r) for v, r in zip(b, [iw, ih, iw, ih])]
    return im[y : y + h, x : x + w, :]

def _stuff(uv_mapper, im, idx, output_path):
    texs = uvm.get_uv(im)
    if texs == None or len(texs) == 0:
        print("warning: no detection in crop")
        return None 
    
    p = f"{output_path}/uv_maps/{idx:06}.jpg"
    texs[0]["texture"].save_to_file(p)
    cv2.imwrite(f"{output_path}/crops/{idx:06}.jpg", im)
    return p

def _stuff2(uv_mapper, chaxis_root):
    fails = set()   
     
    stuff = ["enter", "exit"]     
     
    for sub in stuff:
        new_sub = chaxis_root / f"{sub}_uv"
        
        for person_id_dir in (chaxis_root / sub).iterdir():
            for im_path in person_id_dir.iterdir():
                im = cv2.imread(str(im_path))
                texs = uvm.get_uv(im)               
                
                if texs == None or len(texs) == 0:
                    fails.add(str(person_id_dir.name))
                    print(f"warning: no detection in crop {im_path}")
                else:
                    out_dir = new_sub / person_id_dir.name
                    out_dir.mkdir(exist_ok=True, parents=True)
                    texs[0]["texture"].save_to_file(str(out_dir / im_path.name))     
                    
    print(fails)
    for person_id in fails:
        for s in stuff:
            p = chaxis_root / f"{s}_uv" / person_id
            if p.exists():
                shutil.rmtree(p)

def parse_train_json(uv_mapper, train, data_dir, output_dir):
    for person in tqdm.tqdm(train):
        person_id = person["person_id"].replace("/", "-")       
        
        person_dir = output_dir / person_id
        if person_dir.exists():
            print(f"skipping {person_id} because it already exists")
            continue         
        
        (person_dir / "uv_maps").mkdir(exist_ok=True, parents=True)
        (person_dir / "crops").mkdir(exist_ok=True, parents=True)       
        for idx, sample in enumerate(person["samples"]):
            im = cv2.imread(str(data_dir / sample["image_path"]))
            bbox = [sample[k] for k in ("xtl", "ytl", "xbr", "ybr")]
            crop = _crop(im, bbox)          
            
            p = _stuff(uv_mapper, crop, idx, person_dir)
            if p is not None:
                sample["uv_image_path"] = p