import caption_generator

from keras.callbacks import ModelCheckpoint

def train_model(weight = None, batch_size=00, epochs = 25):

    # Total samples : 64146

    cg = caption_generator.CaptionGenerator()
    model = cg.create_model()

    if weight != None:
        model.load_weights(weight)

    counter = 0
    file_name = 'weights-improvement-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit_generator(cg.data_generator(batch_size=batch_size),  steps_per_epoch=300, epochs=epochs, verbose=2, callbacks=callbacks_list)
    try:
        model.save('C:/Users/pc/Desktop/automatic_image_caption_generation/Models/WholeModel.h5', overwrite=True)
        model.save_weights('C:/Users/pc/Desktop/automatic_image_caption_generation/Models/Weights.h5',overwrite=True)
    except:
        print "Error in saving model."
    print "Training complete...\n"

if __name__ == '__main__':
    train_model()