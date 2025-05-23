import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_4 import MSRVTT_DataLoader

def dataloader_msrvtt_val(args, subset="val"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_msrvtt_train(args, subset="train"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.train_csv,
        features_path=args.features_path,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_train,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"val":dataloader_msrvtt_val, "test":dataloader_msrvtt_val,"train":dataloader_msrvtt_train}
