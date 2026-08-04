import numpy as np
import mygrad as mg
from mynn.layers.dense import dense
from mygrad.nnet.initializers import glorot_normal
from mygrad.nnet.losses import margin_ranking_loss
import pickle

#created myNN model for embedding image descriptors
class ImageDescriptors:
    def __init__(self, caption_dimension, image_dimension, embedding_dimension):
        '''Initializes all of the layers in our model, and sets them
        as attributes of the model.

        Parameters
        ----------
        caption_dimension : int
            The size of the captions.

        image_dimension : int
            The size of the images

        embedding_dimension : int
            The size of the embedded descriptor
        '''
        self.caption_embed = dense(caption_dimension, embedding_dimension, weight_initializer = glorot_normal, bias = True)
        self.image_embed = dense(image_dimension, embedding_dimension, weight_initializer = glorot_normal, bias = True)

    def __call__(self, caption, image):
        '''Passes data as input to our model, forword pass.

        Parameters
        ----------
        caption : Union[numpy.ndarray, mygrad.Tensor], shape=(M, caption_dimension)
            A batch of caption data consisting of M captions,
            each with a dimensionality of caption_dim.

        image : Union[numpy.ndarray, mygrad.Tensor], shape=(M, image_dimension)
            A batch of image data consisting of M images,
            each with a dimensionality of image_dim.

        Returns
        -------
        Tuple[mygrad.Tensor, mygrad.Tensor], each shape=(M, embedding_dim)
        '''
        caption_emb = self.caption_embed(caption)
        image_emb = self.image_embed(image)
        return caption_emb, image_emb
    @property
    def parameters(self):
        '''A convenience function for getting all the parameters of our model.

        This can be accessed as an attribute, via `model.parameters`

        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model
        '''
        return self.caption_embed.parameters + self.image_embed.parameters

#function to compute loss (mygrad's margin ranking loss) and accuracy (fraction of dot product pairs satisfy image_true * caption > image_confusor * caption)
def compute_loss_and_accuracy(caption_emb, image_emb, confusor_emb):
    '''Compute margin ranking loss and accuracy.

    Parameters
    ----------
    caption_emb : mygrad.Tensor, shape=(M, embedding_dim)
        Embeddings for the captions.

    image_emb : mygrad.Tensor, shape=(M, embedding_dim)
        Embeddings for the true images.

    confusor_emb : mygrad.Tensor, shape=(M, embedding_dim)
        Embeddings for the confusor images.

    Returns
    -------
    loss : mygrad.Tensor
        The computed margin ranking loss.

    accuracy : float
        The fraction of correct pairs where the similarity score of the correct image is higher than that of the confusor image.
    '''
    # L2-normalize so similarity is cosine similarity, bounded to [-1, 1].
    # Without this, raw dot-product similarity is unbounded: the model can
    # inflate embedding norms to blow up the loss on remaining hard examples
    # without improving ranking (accuracy is scale-invariant), which makes
    # the fixed numeric margin meaningless and the loss diverge over training.
    caption_emb = caption_emb / mg.sqrt((caption_emb ** 2).sum(axis=1, keepdims=True))
    image_emb = image_emb / mg.sqrt((image_emb ** 2).sum(axis=1, keepdims=True))
    confusor_emb = confusor_emb / mg.sqrt((confusor_emb ** 2).sum(axis=1, keepdims=True))

    # per-example (paired) similarity: caption_emb[i] . image_emb[i], not the
    # full (M, M) cross-batch similarity matrix that "@" would produce, which
    # would compare each caption against every OTHER example's image/confusor
    # too and swamp the real signal with irrelevant pairs
    good_sim = (caption_emb * image_emb).sum(axis=1)
    bad_sim = (caption_emb * confusor_emb).sum(axis=1)

    loss = margin_ranking_loss(x1 = good_sim, x2 = bad_sim, y = 1, margin = 0.25)
    accuracy = np.mean(good_sim.data > bad_sim.data)

    return loss, accuracy

def save_model(model, file_path):
    '''Saves the model weights to a file.

    Parameters
    ----------
    model : ImageDescriptors
        Trained model containing weights

    file_path : str
        Path for the model's weights will be saved
    '''
    with open(file_path, 'wb') as f:
        pickle.dump([p.data for p in model.parameters], f)

def load_model(model, file_path):
    '''Loads the model weights from a file.

    Parameters
    ----------
    model : ImageDescriptors
        Model where weights will load

    file_path : str
       Path where model's weights will be loaded
    '''
    with open(file_path, 'rb') as f:
        weights = pickle.load(f)

    for param, w in zip(model.parameters, weights):
        # np.array(..., copy=True) strips any leftover .base chain from
        # unpickling (observed as a raw `bytes` base on some arrays), which
        # mygrad's internals can't handle when the tensor re-enters a graph
        param.data = np.array(w, copy=True)
