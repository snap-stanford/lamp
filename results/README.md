# Results

This results/ folder stores the files containing the experiment results and model checkpoints. The files (with suffix of ".p") are stored under "results/{--exp_id}_{--date_time}/", where --exp_id and --date_time are specified in the command, indicating the experiment id and date for this batch of experiments. The ".p" file is a dictionary containing all necessary information. It can be loaded via: `data_record = pickle.load(open({FILENAME.p}, "rb"))`. The `data_record` dictionary contains the keys of (e.g.):

* "model_dict": a list of model checkpoints, saved every --inspect_interval epochs. The model can be loaded via the command: `model = load_model(data_record["model_dict"][id])`, where id indicates which checkpoint you want to load from.
* "epoch": a list of integers indicating the corresponding epoch number when the model_dict is saved.
* "train_loss" (or "val_loss", "test_loss"): a list containing the loss for training (validation/test) at the corresponding epoch.
* "last_model_dict": the last model_dict.
* "last_optimizer_dict": optimizer state which can be used for resuming the training.
