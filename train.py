import numpy as np
from mygrad import Tensor
from mynn.optimizers.sgd import SGD

from semantic_search.image_model import (
    ImageDescriptors,
    compute_loss_and_accuracy,
    save_model,
    load_model,
)

#extracting sets of data (caption-ID, image-ID, confusor-image-ID) triples, creating training and validation sets
def shuffle_and_split_data(data, validation_split=0.2):
    '''Shuffles the data and splits it into training and validation sets.

    Parameters
    ----------
    data : numpy.ndarray, shape=(M, N)
        Array of generic data.

    Returns
    -------
    Tuple of numpy.ndarrays
        Training and validation sets for captions, images, and confusors will be split and created.
    '''
    data -= np.mean(data, axis = 0)
    data /= np.std(data, axis = 0)
    N = data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    data = data[indices]
    split_idx = int(N * (1 - validation_split))
    train_data = data[:split_idx]
    validation_data = data[split_idx:]
    return train_data, validation_data

def extract_data(caption, image, confusor):
    '''Extract training and validation sets from three arrays needed to train model

    Parameters
    ----------
    captions : numpy.ndarray, shape=(N, caption_dim)
        Array of caption data.

    images : numpy.ndarray, shape=(N, image_dim)
        Array of image data.

    confusors : numpy.ndarray, shape=(N, image_dim)
        Array of confusor image data.

    Returns
    -------
    Tuple of numpy.ndarrays
        Training and validation sets for captions, images, and confusors.
    '''
    training_set = (shuffle_and_split_data(caption)[0], shuffle_and_split_data(image)[0], shuffle_and_split_data(confusor)[0])
    validation_set = (shuffle_and_split_data(caption)[1], shuffle_and_split_data(image)[1], shuffle_and_split_data(confusor)[1])
    return training_set, validation_set


if __name__ == "__main__":
    #model and optimizer are initialized
    model = ImageDescriptors(caption_dimension = 200, image_dimension = 512, embedding_dimension = 128)
    optimizer = SGD(model.parameters, learning_rate = 1e-3, momentum = 0.9)
    batch_size = 32
    num_epochs = 500

    # TODO: captions, images, confusors still need to come from a triples-extraction
    # step (caption-ID, image-ID, confusor-image-ID) built from semantic_search's
    # COCOOrganizer + text_embedding + resnet18 features. Not implemented yet.
    training_set = extract_data(captions, images, confusors)

    for epoch in range(num_epochs):
        total_loss, total_accuracy = 0.0

        # Shuffle the data at the beginning of each epoch
        indices = np.arange(len(training_set[0]))
        np.random.shuffle(indices)
        training_set = (training_set[0][indices], training_set[1][indices], training_set[2][indices])

        #process batches of data
        for i in range(0, len(training_set[0]), batch_size):
            batch_captions = Tensor(training_set[0][i : i + batch_size])
            batch_images = Tensor(training_set[1][i : i + batch_size])
            batch_confusors = Tensor(training_set[2][i : i + batch_size])

            #perform forword pass
            caption_emb, image_emb = model(batch_captions, batch_images)
            _, confusor_emb = model(batch_captions, batch_confusors)

            #loss and accuracy computation
            loss, accuracy = compute_loss_and_accuracy(caption_emb, image_emb, confusor_emb)

            #optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss = total_loss + loss.item() * len(batch_captions)
            total_accuracy = total_accuracy + accuracy * len(batch_captions)

        avg_loss = total_loss / len(training_set[0])
        avg_accuracy = total_accuracy / len(training_set[0])

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    #call save and load functions to determine final model parameters
    save_model(model, 'mynn_model_weights_final.pkl')
    load_model(model, 'mynn_model_weights_final.pkl')
