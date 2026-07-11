from cogworks_data.language import get_data_path

from pathlib import Path
import json
import pickle

class COCOOrganizer:
    
    def __init__(self):
        # load COCO data
        filename = get_data_path("captions_train2014.json")
        with Path(filename).open() as f:
            self.COCO_data = json.load(f)
            
        with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
            self.resnet18_features = pickle.load(f)
        # store all img + caption IDs
        self.image_ids = self.resnet18_features.keys()
        self.caption_ids = []
        # Initialize Various mappings between image/caption IDs, and associating caption-IDs with captions        
        self.image_to_caption_ids = {}
        self.caption_to_image = {}
        self.caption_id_to_caption = {}
        self.caption_id_to_feature = {}
        self.caption_id_to_image_url = {}
        self.image_id_to_url = {img["id"]: img["coco_url"] for img in self.COCO_data["images"]}
        self.organize()
        
    def organize(self):
        #loop thru images
        count = 0
        for image_id in self.image_ids:
            self.image_to_caption_ids[image_id] = [] 
        for annotation in self.COCO_data["annotations"]:
            #get IDs and caption for each annotation
            image_id = annotation['image_id']
            caption_id = annotation['id'] 
            caption = annotation['caption']
#             print(f"Processing annotation with image_id: {image_id}, caption_id: {caption_id}")
            if image_id in self.image_ids:
                self.caption_ids.append(caption_id)
                self.caption_to_image[caption_id] = image_id
                self.caption_id_to_caption[caption_id] = caption
                self.image_to_caption_ids[image_id].append(caption_id)
                self.caption_id_to_feature[caption_id] = self.resnet18_features[image_id]
                self.caption_id_to_image_url[caption_id] = self.image_id_to_url[image_id]
                count+=1
            if count == 10000:
                print("10000 annotations processed")
            if count == 20000:
                print("20000 annotations processed")

    def get_caption_id(self, image_id):
        return self.image_to_caption_ids.get(image_id)
    def get_image_id(self, caption_id):
        return self.caption_to_image.get(caption_id)
    def get_caption(self, caption_id):
        return self.caption_id_to_caption.get(caption_id)
    def get_feature(self, caption_id):
        return self.caption_id_to_feature.get(caption_id)
    def get_image_url(self, caption_id):
        return self.caption_id_to_image_url.get(caption_id)