from easydict import EasyDict as edict
import datetime

cfg = edict()

cfg.num_classes = 3
cfg.epochs = 30
cfg.batch_size = 256
cfg.save_mode_path = "soner_classify.h5"
cfg.log_dir = ".\\logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# training options
cfg.train = edict()
cfg.train.num_samples = 157470
cfg.train.learning_rate = 1e-4
cfg.train.dataset = "./tfrecords/train.tfrecords"
cfg.val = edict()
# cfg.val.num_samples = 42608
cfg.val.num_samples = 10651
cfg.val.dataset = "./tfrecords/val.tfrecords"

cfg.num_samples = 0
cfg.num_samples_file_name = "../tfrecords/num_samples_data.npy"
