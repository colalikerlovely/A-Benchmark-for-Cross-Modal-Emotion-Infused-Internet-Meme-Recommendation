import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_Douban_MEMERS_retrieval import Douban_MEMERS_DataLoader
from dataloaders.dataloader_Douban_MEMERS_retrieval import Douban_MEMERS_TrainDataLoader

def dataloader_Douban_MEMERS_train(args, tokenizer, load_image, caption_aug, generate_text_backbone_clip_bert=0,original_text_backbone_clip_bert=0):
    Douban_MEMERS_dataset = Douban_MEMERS_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        strategy=args.strategy,
        generate_text_backbone_clip_bert=generate_text_backbone_clip_bert,
        original_text_backbone_clip_bert=original_text_backbone_clip_bert,
        num_class_names=12, 
        generate_images = args.generate_images,
        load_image = load_image,
        caption_aug= caption_aug,
        train_emotion=args.train_emotion,
        test_emotion=args.test_emotion
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(Douban_MEMERS_dataset)
    dataloader = DataLoader(
        Douban_MEMERS_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(Douban_MEMERS_dataset), train_sampler

def dataloader_Douban_MEMERS_test(args, tokenizer,load_image, subset="test", generate_text_backbone_clip_bert=0,original_text_backbone_clip_bert=0):
    Douban_MEMERS_testset = Douban_MEMERS_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        generate_text_backbone_clip_bert=generate_text_backbone_clip_bert,
        original_text_backbone_clip_bert=original_text_backbone_clip_bert,
        num_class_names=12,
        generate_images=args.generate_images,
        load_image = load_image,
        train_emotion=args.train_emotion,
        test_emotion=args.test_emotion
    )
    dataloader_Douban_MEMERS = DataLoader(
        Douban_MEMERS_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_Douban_MEMERS, len(Douban_MEMERS_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["Douban_MEMERS"] = {"train":dataloader_train, "val":dataloader_test, "test":None}
