import torch
from torch.utils.data import DataLoader
from dataset import MajoranaDataset, DatasetConstants, majorana_collate_fn

def get_dataloaders(train_root='data/waveform_npz/MJD_Train_0',
                    test_root='data/waveform_npz/MJD_Test_0',
                    window_size=1,
                    batch_size=32,
                    num_workers=4,
                    normalize_energy=False,
                    use_stats=True):
    
    constants = DatasetConstants() if use_stats else None
    
    train_dataset = MajoranaDataset(
        root=train_root,
        window_size=window_size,
        constants=constants,
        to_torch=True,
        test=False,
        normalize_energy=normalize_energy
    )
    
    test_dataset = MajoranaDataset(
        root=test_root,
        window_size=window_size,
        constants=constants,
        to_torch=True,
        test=True,
        normalize_energy=normalize_energy
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=majorana_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=majorana_collate_fn
    )
    
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders(
        train_root='data/waveform_npz',
        test_root='data/waveform_npz/MJD_Test_0',
        window_size=1,
        batch_size=4,
        num_workers=0,
        normalize_energy=True
    )
    
    print("Dataset loaders created successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    batch = next(iter(train_loader))
    print("\nSample batch:")
    print(f"  Waveform shape: {batch['waveform'].shape}")
    print(f"  Gradient shape: {batch['gradient'].shape}")
    
    if 'energy_label' in batch:
        print(f"  Energy label shape: {batch['energy_label'].shape}")
        print(f"  Energy label sample: {batch['energy_label'][:3]}")
    
    print(f"\n  Available keys: {list(batch.keys())}")
    
    for key in ['detector', 'psd_label_dcr', 'psd_label_high_avse']:
        if key in batch:
            print(f"  {key} sample: {batch[key][:3]}")
