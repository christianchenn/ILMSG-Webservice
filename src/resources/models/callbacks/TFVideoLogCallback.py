import wandb
import tensorflow as tf

class TFVideoLogCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, log_steps):
        super().__init__()
        self.log_steps = log_steps
        self.batch_count = 0
        
    def on_train_batch_begin(self, batch, logs=None):
        print(batch)
        # self.batch_count += 1
        # if self.batch_count % self.log_steps == 0:
        #     # log custom metrics to WandB
        #     wandb.log({'my_custom_metric': logs['my_custom_metric']})

    