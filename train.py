from tensorflow.keras.optimizers import Adam

# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stan.models import STAN
from stan.utils.metrics import dice_coef
from stan.utils.losses import focal_tversky_loss

from tools.helpers import get_callbacks, get_generators


def train(
    train_dir, val_dir,
    n_class, lr,
    input_shape, input_channel,
    decode_mode,
    snapshot_dir, tensorboard, tensorboard_dir,
    model_name
):
    input_shape = (input_shape, input_shape)
    epochs=3
    batch_size=8

    optimizer = Adam(lr=lr)
    criterion = focal_tversky_loss(gamma=0.75)

    callbacks = get_callbacks(snapshot_dir, model_name, 
                              tensorboard, tensorboard_dir, batch_size)

    train_gen, val_gen = get_generators(train_dir, val_dir,
                                        input_shape, input_channel,
                                        horizontal_flip=True)

    model = STAN(
        n_class,
        input_shape=(input_shape[0], input_shape[1], input_channel),
        decode_mode=decode_mode,
        output_activation='sigmoid'
    )
    model.compile(optimizer=optimizer, loss=criterion, metrics=[dice_coef])
    model.fit(train_gen, batch_size=batch_size, callbacks=callbacks,
              epochs=epochs, steps_per_epoch=len(train_gen),
              validation_data=val_gen, validation_steps=1,verbose="1")


if __name__ == "__main__":

    train_dir = r"C:\Users\Salvatore\Desktop\Computer_Vision\computer_vision\Dataset_BUSI_with_GT\malignant"
    val_dir = ""
    n_class = 1
    epochs=2
    batch_size=8
    snapshot_dir = "."
    lr=1e-4
    input_shape=256
    input_channel= 3
    decode_mode ='transpose'
    tensorboard = False
    tensorboard_dir = "./log"
    model_name = "Test"

    train(train_dir,val_dir,n_class,lr,input_shape,input_channel,decode_mode,snapshot_dir,tensorboard,tensorboard_dir,model_name)
