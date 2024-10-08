import json
import torch
import torch.nn as nn
import numpy as np
from args import get_parser
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
    normalize = args.normalize
    save_path = f'output/{dataset}/{args.id}'

    # Load the data
    (x_test, y_test) = get_data(dataset, normalize=normalize, train=False)

    # Convert data to tensors
    x_test = torch.from_numpy(x_test).float()
    y_test = np.array(y_test)  # Ensure y_test is a numpy array for metric calculation
    n_features = x_test.shape[1]

    # Set up the model
    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
    elif isinstance(target_dims, int):
        out_dim = 1
    else:
        out_dim = len(target_dims)

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )

    # Set up Trainer (optimizer and criteria are not needed for testing)
    trainer = Trainer(
        model,
        None,  # optimizer not needed for testing
        window_size,
        n_features,
        target_dims,
        None,  # n_epochs not needed for testing
        None,  # batch_size not needed for testing
        None,  # init_lr not needed for testing
        None,  # forecast_criterion not needed for testing
        None,  # recon_criterion not needed for testing
        args.use_cuda,
        save_path,
        None,  # log_dir not needed for testing
        args.print_every,
        args.log_tensorboard,
        str(args.__dict__)
    )

    # Some suggestions for POT args
    level_q_dict = {
        "SWAT": (0.90, 0.005),
        "MACHINE" : (0.90, 0.005)
    }
    key = args.dataset
    level, q = level_q_dict[key]
    
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SWAT": 1,
                      "MACHINE" : 1}
    reg_level = reg_level_dict[key]

    # Load the trained model
    trainer.load(f"{save_path}/model.pt")

    # Set up the Predictor
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }

    predictor = Predictor(
        model,
        window_size,
        n_features,
        prediction_args
    )

    # Run prediction
    outputs = predictor.predict_anomalies(
        test=x_test,
        true_anomalies=y_test[window_size:]
    )
