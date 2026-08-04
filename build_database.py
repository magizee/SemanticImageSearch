import pickle
import warnings

import numpy as np
from mygrad import Tensor

from semantic_search.coco_organizer import COCOOrganizer
from semantic_search.image_model import ImageDescriptors, load_model
from semantic_search.database import Database

# macOS's Accelerate BLAS backend raises spurious divide-by-zero/overflow/
# invalid-value RuntimeWarnings on ordinary, finite matmuls (verified: output
# has no actual NaN/Inf). Harmless false positive, not a real numeric issue.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

CAPTION_DIM = 200
IMAGE_DIM = 512
EMBEDDING_DIM = 128


if __name__ == "__main__":
    coco = COCOOrganizer()

    model = ImageDescriptors(caption_dimension=CAPTION_DIM, image_dimension=IMAGE_DIM, embedding_dimension=EMBEDDING_DIM)
    load_model(model, 'mynn_model_weights_final.pkl')

    image_ids = list(coco.image_ids)
    features = np.stack([coco.resnet18_features[image_id].reshape(-1) for image_id in image_ids])
    embeddings = model.image_embed(Tensor(features)).data

    db = Database()
    db.create_image_database(image_ids, embeddings)
    db.save_image_database(database=db.image_embeddings, filename='data/database.pkl')
    print(f"Saved {len(db.image_embeddings)} image embeddings to data/database.pkl")

    image_metadata = {}
    for image_id in image_ids:
        caption_ids = coco.get_caption_id(image_id)
        caption = coco.get_caption(caption_ids[0]) if caption_ids else None
        image_metadata[image_id] = {
            "url": coco.get_image_url(caption_ids[0]) if caption_ids else None,
            "caption": caption,
        }
    with open('data/image_metadata.pkl', 'wb') as f:
        pickle.dump(image_metadata, f)
    print(f"Saved metadata for {len(image_metadata)} images to data/image_metadata.pkl")
