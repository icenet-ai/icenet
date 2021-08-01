def train_model(
        seed=42,
        learning_rate=5e-4,
        filter_size=3,
        n_filters_factor=2.,
        n_forecast_days=1,
        weight_decay=0.,
        batch_size=4,
        dropout_rate=0.5):
    np.random.seed(seed)
    tf.random.set_seed = seed
    config = dict(
        seed=args.seed,
        lr=args.lr,
        filter_size=args.filter_size,
        n_filters_factor=args.n_filters_factor,
        lr_10e_decay_fac=args.lr_10e_decay_fac,
        lr_decay_start=args.lr_decay_start,
        lr_decay_end=args.lr_decay_end,
    )
    train_ds, test_ds, val_ds, counts = load_dataset(args.input, batch_size=args.batch)
    train(config, train_ds, val_ds, counts, batch_size=args.batch, epochs=args.epoch)

    icenet2_name = 'unet_batchnorm'
    dataloader_name = '2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month'

    pre_load_network = False
    pre_load_network_fname = 'network_{}.h5'.format(seed)
    custom_objects = {
        'weighted_MAE_corrected': metrics.weighted_MAE_corrected,
        'weighted_RMSE_corrected': metrics.weighted_RMSE_corrected,
        'weighted_MSE_corrected': losses.weighted_MSE_corrected,
    }
    # prev_best = 11.05  # Baseline 'monitor' value for EarlyStopping callback
    prev_best = None

    n_epochs = 35

    # early_stopping_patience = 5
    early_stopping_patience = n_epochs  # No of epochs without improvement before training is aborted

    # checkpoint_monitor = 'val_weighted_RMSE'
    checkpoint_monitor = 'val_weighted_MAE_corrected'
    checkpoint_mode = 'min'

    max_queue_size = 3
    workers = 5
    use_multiprocessing = True

    training_verbosity = 2

    # Data loaders; set up paths
    ####################################################################

    network_folder = os.path.join(config.folders['results'], dataloader_name,
                                  icenet2_name, 'networks')
    if not os.path.exists(network_folder):
        os.makedirs(network_folder)
    network_fpath = os.path.join(network_folder,
                                 'network_{}.h5'.format(wandb.config.seed))
    network_path_preload = os.path.join(network_folder, pre_load_network_fname)

    dataloader_config_fpath = os.path.join('dataloader_configs',
                                           dataloader_name + '.json')

    dataloader = utils.IceNet2DataLoader(dataloader_config_fpath,
                                         wandb.config.seed)

    val_dataloader = utils.IceNet2DataLoader(dataloader_config_fpath)
    val_dataloader.convert_to_validation_data_loader()

    input_shape = (
    *dataloader.config['raw_data_shape'], dataloader.tot_num_channels)

    print(
        '\n\nNUM TRAINING SAMPLES: {}'.format(len(dataloader.all_forecast_IDs)))
    print('NUM VALIDATION SAMPLES: {}\n\n'.format(
        len(val_dataloader.all_forecast_IDs)))
    print('NUM INPUT CHANNELS: {}\n\n'.format(dataloader.tot_num_channels))

    #### Loss, metrics, and callbacks
    ####################################################################

    # Loss
    loss = losses.weighted_MSE_corrected
    # loss = losses.weighted_MSE

    # Metrics
    metrics_list = [
        # metrics.weighted_MAE,
        metrics.weighted_MAE_corrected,
        metrics.weighted_RMSE_corrected,
        losses.weighted_MSE_corrected
    ]

    # Callbacks
    callbacks_list = []

    # Checkpoint the model weights when a validation metric is improved
    callbacks_list.append(
        ModelCheckpoint(
            filepath=network_fpath,
            monitor=checkpoint_monitor,
            verbose=1,
            mode=checkpoint_mode,
            save_best_only=True
        ))

    # Abort training when validation performance stops improving
    callbacks_list.append(
        EarlyStopping(
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            verbose=1,
            patience=early_stopping_patience,
            baseline=prev_best
        ))

    np.random.seed(wandb.config.seed)
    tf.random.set_seed = wandb.config.seed
    dataloader.set_seed(wandb.config.seed)
    dataloader.on_epoch_end()  # Randomly shuffle training samples

    strategy = tf.distribute.experimental.CentralStorageStrategy()

    with strategy.scope():
        if pre_load_network:
            print("\nLoading network from {}... ".format(network_path_preload),
                  end='', flush=True)
            network = load_model(network_path_preload,
                                 custom_objects=custom_objects)
            print('Done.\n')

        else:

            network = models.unet_batchnorm(
                input_shape=input_shape,
                loss=loss,
                metrics=metrics_list,
                learning_rate=learning_rate,
                filter_size=filter_size,
                n_filters_factor=n_filters_factor,
                n_forecast_days=dataloader.config['n_forecast_days'],
            )

        network = unet_batchnorm(input_shape=[432, 432, 57],
                                 filter_size=cfg['filter_size'],
                                 n_filters_factor=cfg['n_filters_factor'],
                                 n_forecast_months=cfg['n_forecast_months'])

        network.summary()

        logging.info("Compiling network")

        # TODO: Custom training for distributing loss calculations
        # TODO: Recode categorical_focal_loss(gamma=2.)
        # TODO: Recode construct_custom_categorical_accuracy(use_all_forecast_months=True|False. single_forecast_leadtime_idx=0)
        network.compile(
            optimizer=tf.keras.optimizers.Adam(lr=cfg['learning_rate']),
            loss='MeanSquaredError',
            metrics=[])

        model_history = network.fit(train,
                                    epochs=epochs,
                                    steps_per_epoch=counts[
                                                        'train'] / batch_size,
                                    validation_steps=counts[
                                                         'val'] / batch_size,
                                    validation_data=val, )

        print('\n\nTraining IceNet2:\n')
        history = network.fit(
            dataloader,
            epochs=n_epochs,
            verbose=training_verbosity,
            callbacks=callbacks_list,
            validation_data=val_dataloader,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing
        )

