import pickle
import warnings

import numpy as np
from mygrad import Tensor
from mynn.optimizers.sgd import SGD

# macOS's Accelerate BLAS backend raises spurious divide-by-zero/overflow/
# invalid-value RuntimeWarnings on ordinary, finite matmuls (verified: output
# has no actual NaN/Inf). Harmless false positive, not a real numeric issue.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from semantic_search.coco_organizer import COCOOrganizer
from semantic_search.text_embedding import make_caption_descriptor
from semantic_search.image_model import (
    ImageDescriptors,
    compute_loss_and_accuracy,
    save_model,
    load_model,
)

CAPTION_DIM = 200
IMAGE_DIM = 512


def build_triples(coco, idf_map, seed=None):
    '''Builds (caption-embedding, true-image-feature, confusor-image-feature) triples.

    One row per COCO caption: the caption's GloVe/IDF embedding, its own
    image's resnet18 feature, and a randomly chosen *different* image's
    resnet18 feature as the confusor.

    Parameters
    ----------
    coco : COCOOrganizer
    idf_map : Dict[str, float]
    seed : Optional[int]

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        captions shape=(N, 200), images shape=(N, 512), confusors shape=(N, 512)
    '''
    rng = np.random.default_rng(seed)
    image_id_arr = np.array(list(coco.image_ids))
    true_image_ids = np.array([coco.caption_to_image[cid] for cid in coco.caption_ids])

    # vectorized confusor sampling: pick a random other image per caption,
    # resampling only the (rare) rows that land on the caption's own image
    confusor_idx = rng.integers(0, len(image_id_arr), size=len(true_image_ids))
    confusor_image_ids = image_id_arr[confusor_idx]
    collisions = confusor_image_ids == true_image_ids
    while collisions.any():
        confusor_idx[collisions] = rng.integers(0, len(image_id_arr), size=collisions.sum())
        confusor_image_ids = image_id_arr[confusor_idx]
        collisions = confusor_image_ids == true_image_ids

    n = len(coco.caption_ids)
    captions = np.empty((n, CAPTION_DIM))
    images = np.empty((n, IMAGE_DIM))
    confusors = np.empty((n, IMAGE_DIM))
    for i, caption_id in enumerate(coco.caption_ids):
        captions[i] = make_caption_descriptor(coco.caption_id_to_caption[caption_id], idf_map)
        images[i] = coco.resnet18_features[true_image_ids[i]].reshape(-1)
        confusors[i] = coco.resnet18_features[confusor_image_ids[i]].reshape(-1)

    return captions, images, confusors


def extract_data(caption, image, confusor, validation_split=0.2, seed=None):
    '''Normalizes each feature column, then jointly shuffles and splits the
    three parallel arrays into training and validation sets, keeping
    caption[i]/image[i]/confusor[i] aligned to the same triple throughout.

    Parameters
    ----------
    caption : numpy.ndarray, shape=(N, caption_dim)
    image : numpy.ndarray, shape=(N, image_dim)
    confusor : numpy.ndarray, shape=(N, image_dim)

    Returns
    -------
    Tuple[Tuple[numpy.ndarray, ...], Tuple[numpy.ndarray, ...]]
        (training_set, validation_set), each a (caption, image, confusor) triple.
    '''
    def normalize(data):
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)  # avoid divide-by-zero on constant columns
        return (data - np.mean(data, axis=0)) / std

    caption, image, confusor = normalize(caption), normalize(image), normalize(confusor)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(caption.shape[0])
    caption, image, confusor = caption[indices], image[indices], confusor[indices]

    split_idx = int(caption.shape[0] * (1 - validation_split))
    training_set = (caption[:split_idx], image[:split_idx], confusor[:split_idx])
    validation_set = (caption[split_idx:], image[split_idx:], confusor[split_idx:])
    return training_set, validation_set


if __name__ == "__main__":
    coco = COCOOrganizer()
    with open('data/idf_map.pkl', 'rb') as f:
        idf_map = pickle.load(f)['idf_map']

    captions, images, confusors = build_triples(coco, idf_map)
    training_set, validation_set = extract_data(captions, images, confusors)

    #model and optimizer are initialized
    model = ImageDescriptors(caption_dimension=CAPTION_DIM, image_dimension=IMAGE_DIM, embedding_dimension=128)
    optimizer = SGD(model.parameters, learning_rate=1e-3, momentum=0.9)
    batch_size = 32
    num_epochs = 500

    for epoch in range(num_epochs):
        total_loss, total_accuracy = 0.0, 0.0

        # Shuffle the data at the beginning of each epoch (one shared
        # permutation applied to all three arrays, so triples stay aligned)
        indices = np.arange(len(training_set[0]))
        np.random.shuffle(indices)
        training_set = (training_set[0][indices], training_set[1][indices], training_set[2][indices])

        #process batches of data
        for i in range(0, len(training_set[0]), batch_size):
            batch_captions = Tensor(training_set[0][i : i + batch_size])
            batch_images = Tensor(training_set[1][i : i + batch_size])
            batch_confusors = Tensor(training_set[2][i : i + batch_size])

            #perform forword pass
            # (confusor embedded directly via image_embed, not model(), to
            # avoid recomputing and discarding a redundant caption_emb --
            # that dangling, never-backpropagated graph branch was leaving
            # caption_embed's weight tensor permanently marked read-only)
            caption_emb, image_emb = model(batch_captions, batch_images)
            confusor_emb = model.image_embed(batch_confusors)

            #loss and accuracy computation
            loss, accuracy = compute_loss_and_accuracy(caption_emb, image_emb, confusor_emb)

            #optimize
            loss.backward()
            optimizer.step()
            # mygrad auto-nulls a tensor's gradient once it re-enters a new
            # computational graph, so no explicit zero_grad() call is needed
            # (mynn's SGD doesn't provide one)

            total_loss = total_loss + loss.item() * len(batch_captions)
            total_accuracy = total_accuracy + accuracy * len(batch_captions)

        avg_loss = total_loss / len(training_set[0])
        avg_accuracy = total_accuracy / len(training_set[0])

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    #call save and load functions to determine final model parameters
    save_model(model, 'mynn_model_weights_final.pkl')
    load_model(model, 'mynn_model_weights_final.pkl')
