import numpy as np
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
        self.caption_embed = dense(caption_dimension, embedding_dimension, weight_initializer = glorot_normal, bias = False)
        self.image_embed = dense(image_dimension, embedding_dimension, weight_initializer = glorot_normal, bias = False)

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
    good_sim = (caption_emb @ image_emb.T)
    bad_sim = (caption_emb @ confusor_emb.T)

    loss = margin_ranking_loss(x1 = good_sim, x2 = bad_sim, y = 1, margin = 0.25)
    accuracy = np.mean(good_sim > bad_sim)

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
        pickle.dump({k: v.data for k, v in model.parameters.items()}, f)

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

    for k, v in weights.items():
        model.parameters[k].data = v
