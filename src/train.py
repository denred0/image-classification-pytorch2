from src.datamodule import ICPDataModule
from src.model import ICPModel
from src.inference import ICPInference

from pathlib import Path

# lightning related imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import timm

import datetime

project_name = 'covid'
data_dir = 'data'
images_ext = 'jpg'
augment_p = 0.7
init_lr = 0.0003
early_stop_patience = 6
max_epochs = 2
progress_bar_refresh_rate = 10

senet154 = {'model_type': 'senet154', 'im_size': None, 'im_size_test': None, 'batch_size': 16}

models = [senet154]

models_for_training = []

for m in models:
    model_data = {'model': m}

    mod = timm.create_model(model_data['model']['model_type'], pretrained=False)
    model_mean = list(mod.default_cfg['mean'])
    model_std = list(mod.default_cfg['std'])

    # get input size
    im_size = 0
    im_size_test = 0

    print(model_data['model']['model_type'] + ' input size is ' + str(mod.default_cfg['input_size']))
    if model_data['model']['im_size']:
        im_size = model_data['model']['im_size']
    else:
        im_size = mod.default_cfg['input_size'][1]

    if model_data['model']['im_size_test']:
        im_size_test = model_data['model']['im_size']
    else:
        im_size_test = mod.default_cfg['input_size'][1]

    dm = ICPDataModule(data_dir=data_dir,
                       augment_p=augment_p,
                       images_ext=images_ext,
                       model_type=model_data['model']['model_type'],
                       batch_size=model_data['model']['batch_size'],
                       input_resize=im_size,
                       input_resize_test=im_size_test,
                       mean=model_mean,
                       std=model_std)

    # To access the x_dataloader we need to call prepare_data and setup.
    # dm.prepare_data()
    dm.setup()

    # Init our model
    model = ICPModel(model_type=model_data['model']['model_type'],
                     num_classes=dm.num_classes,
                     classes_weights=None,
                     learning_rate=init_lr)

    # Initialize a trainer
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=early_stop_patience,
        verbose=True,
        mode='min'
    )

    # logs for tensorboard
    experiment_name = model_data['model']['model_type']
    logger = TensorBoardLogger('tb_logs/' + project_name + '/', name=experiment_name)

    checkpoint_name = experiment_name + '_' + '_{epoch}_{val_loss:.3f}_{val_acc:.3f}_{val_f1_epoch:.3f}'

    checkpoint_callback_loss = ModelCheckpoint(monitor='val_loss',
                                               mode='min',
                                               filename=checkpoint_name,
                                               verbose=True,
                                               save_top_k=1,
                                               save_last=False)

    checkpoint_callback_acc = ModelCheckpoint(monitor='val_acc',
                                              mode='max',
                                              filename=checkpoint_name,
                                              verbose=True,
                                              save_top_k=1,
                                              save_last=False)

    checkpoints = [checkpoint_callback_acc, checkpoint_callback_loss, early_stop_callback]
    callbacks = checkpoints

    trainer = pl.Trainer(max_epochs=max_epochs,
                         progress_bar_refresh_rate=progress_bar_refresh_rate,
                         gpus=1,
                         logger=logger,
                         callbacks=callbacks)

    model_data['icp_datamodule'] = dm
    model_data['icp_model'] = model
    model_data['icp_trainer'] = trainer

    models_for_training.append(model_data)

for model in models_for_training:
    print('##################### START Training ' + model['model']['model_type'] + '... #####################')

    # Train the model ⚡g🚅⚡
    model['icp_trainer'].fit(model['icp_model'], model['icp_datamodule'])

    # Evaluate the model on the held out test set ⚡⚡
    results = model['icp_trainer'].test()[0]

    # save test results
    best_checkpoint = 'best_checkpoint: ' + model['icp_trainer'].checkpoint_callback.best_model_path
    results['best_checkpoint'] = best_checkpoint

    filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '__test_acc_' + str(
        round(results.get('test_acc'), 4)) + '.txt'

    path = 'test_logs/' + project_name + '/' + model['model']['model_type']
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(path + '/' + filename, 'w+') as f:
        print(results, file=f)

    print('##################### END Training ' + model['model']['model_type'] + '... #####################')

# def inference():
#     ICPInference(data_dir='inference',
#                  img_size=380,
#                  show_accuracy=True,
#                  checkpoint='tb_logs/tf_efficientnet_b4_ns/version_4/checkpoints/tf_efficientnet_b4_ns__epoch=2_val_loss=0.922_val_acc=0.830_val_f1_epoch=0.000.ckpt',
#                  std=[0.229, 0.224, 0.225],
#                  mean=[0.485, 0.456, 0.406],
#                  confidence_threshold=1).predict()
#
#
# if __name__ == '__main__':
#     main()
#     # inference()
