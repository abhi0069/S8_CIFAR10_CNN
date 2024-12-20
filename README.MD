# CIFAR-10 Image Classification with Dilated Convolutions
This project implements a CNN model for CIFAR-10 image classification using dilated convolutions and advanced data augmentation techniques.

## Features
- **Advanced Model Architecture**
  - Depthwise Separable Convolutions
  - Dilated Convolutions with progressive dilation rates
  - Global Average Pooling
  - Batch Normalization
  - Dropout for regularization

- **Data Augmentation (Albumentations)**
  - Horizontal Flips
  - ShiftScaleRotate
  - CoarseDropout (Cutout)
  - Normalization

## Key components:
 - Input → 3 channels
 - C1 Block: Conv (dilation=1) → 24 channels
 - C2 Block: Depthwise Separable Conv → 32 channels
 - C3 Block: Conv (dilation=4) → 48 channels
 - C4 Block: Conv (dilation=8) → 64 channels
 - GAP → Dropout → FC → 10 classes


## Requirements
- bash
- torch
- torchvision
- albumentations
- numpy
- matplotlib
- tqdm


## Results

The model achieves:
- Best Training Accuracy: 84.97%
- Best Test Accuracy: 88.93%

## Logs

EPOCH: 0
Epoch: 0 Loss=1.5457 Batch_id=781 Accuracy=41.11: 100%|█| 782/782 [00:46<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 1.1954, Accuracy: 5642/10000 (56.42%)

EPOCH: 1
Epoch: 1 Loss=1.2102 Batch_id=781 Accuracy=55.51: 100%|█| 782/782 [00:56<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.9612, Accuracy: 6588/10000 (65.88%)

EPOCH: 2
Epoch: 2 Loss=1.4485 Batch_id=781 Accuracy=61.79: 100%|█| 782/782 [00:57<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.9320, Accuracy: 6740/10000 (67.40%)

EPOCH: 3
Epoch: 3 Loss=0.7010 Batch_id=781 Accuracy=65.15: 100%|█| 782/782 [00:57<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.8113, Accuracy: 7184/10000 (71.84%)

EPOCH: 4
Epoch: 4 Loss=1.1807 Batch_id=781 Accuracy=66.68: 100%|█| 782/782 [00:49<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.8311, Accuracy: 7183/10000 (71.83%)

EPOCH: 5
Epoch: 5 Loss=1.1475 Batch_id=781 Accuracy=67.99: 100%|█| 782/782 [00:44<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.7454, Accuracy: 7414/10000 (74.14%)

EPOCH: 6
Epoch: 6 Loss=1.1900 Batch_id=781 Accuracy=69.26: 100%|█| 782/782 [00:44<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.8131, Accuracy: 7219/10000 (72.19%)

EPOCH: 7
Epoch: 7 Loss=0.7969 Batch_id=781 Accuracy=69.67: 100%|█| 782/782 [00:43<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.9427, Accuracy: 6920/10000 (69.20%)

EPOCH: 8
Epoch: 8 Loss=1.1288 Batch_id=781 Accuracy=70.29: 100%|█| 782/782 [00:45<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.7867, Accuracy: 7355/10000 (73.55%)

EPOCH: 9
Epoch: 9 Loss=1.0401 Batch_id=781 Accuracy=71.13: 100%|█| 782/782 [00:45<00:
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.6736, Accuracy: 7710/10000 (77.10%)

EPOCH: 10
Epoch: 10 Loss=0.6777 Batch_id=781 Accuracy=71.86: 100%|█| 782/782 [00:45<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.8049, Accuracy: 7351/10000 (73.51%)

EPOCH: 11
Epoch: 11 Loss=0.5156 Batch_id=781 Accuracy=71.82: 100%|█| 782/782 [00:52<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.7076, Accuracy: 7562/10000 (75.62%)

EPOCH: 12
Epoch: 12 Loss=0.7372 Batch_id=781 Accuracy=72.31: 100%|█| 782/782 [00:44<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.6673, Accuracy: 7718/10000 (77.18%)

EPOCH: 13
Epoch: 13 Loss=1.1543 Batch_id=781 Accuracy=72.96: 100%|█| 782/782 [00:43<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.6305, Accuracy: 7806/10000 (78.06%)

EPOCH: 14
Epoch: 14 Loss=0.8262 Batch_id=781 Accuracy=72.92: 100%|█| 782/782 [00:43<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.6762, Accuracy: 7745/10000 (77.45%)

EPOCH: 15
Epoch: 15 Loss=0.5842 Batch_id=781 Accuracy=73.56: 100%|█| 782/782 [00:43<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.6713, Accuracy: 7732/10000 (77.32%)

EPOCH: 16
Epoch: 16 Loss=0.6336 Batch_id=781 Accuracy=73.97: 100%|█| 782/782 [00:43<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.6097, Accuracy: 7913/10000 (79.13%)

EPOCH: 17
Epoch: 17 Loss=0.5807 Batch_id=781 Accuracy=74.41: 100%|█| 782/782 [00:43<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.6593, Accuracy: 7783/10000 (77.83%)

EPOCH: 18
Epoch: 18 Loss=0.4172 Batch_id=781 Accuracy=74.78: 100%|█| 782/782 [00:43<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.5904, Accuracy: 8017/10000 (80.17%)

EPOCH: 19
Epoch: 19 Loss=0.6671 Batch_id=781 Accuracy=75.43: 100%|█| 782/782 [00:43<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.5139, Accuracy: 8227/10000 (82.27%)

EPOCH: 20
Epoch: 20 Loss=0.5331 Batch_id=781 Accuracy=76.26: 100%|█| 782/782 [00:43<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.5554, Accuracy: 8093/10000 (80.93%)

EPOCH: 21
Epoch: 21 Loss=0.8564 Batch_id=781 Accuracy=76.79: 100%|█| 782/782 [00:43<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.5839, Accuracy: 8093/10000 (80.93%)

EPOCH: 22
Epoch: 22 Loss=0.6310 Batch_id=781 Accuracy=77.58: 100%|█| 782/782 [00:44<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.5128, Accuracy: 8236/10000 (82.36%)

EPOCH: 23
Epoch: 23 Loss=0.4980 Batch_id=781 Accuracy=78.42: 100%|█| 782/782 [00:44<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.4509, Accuracy: 8438/10000 (84.38%)

EPOCH: 24
Epoch: 24 Loss=0.8387 Batch_id=781 Accuracy=79.84: 100%|█| 782/782 [00:46<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.4312, Accuracy: 8498/10000 (84.98%)

EPOCH: 25
Epoch: 25 Loss=0.5512 Batch_id=781 Accuracy=81.08: 100%|█| 782/782 [00:45<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.3965, Accuracy: 8653/10000 (86.53%)

EPOCH: 26
Epoch: 26 Loss=1.2048 Batch_id=781 Accuracy=81.93: 100%|█| 782/782 [00:47<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.3629, Accuracy: 8748/10000 (87.48%)

EPOCH: 27
Epoch: 27 Loss=0.4187 Batch_id=781 Accuracy=83.77: 100%|█| 782/782 [00:47<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.3442, Accuracy: 8829/10000 (88.29%)

EPOCH: 28
Epoch: 28 Loss=0.4783 Batch_id=781 Accuracy=84.24: 100%|█| 782/782 [00:50<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.3303, Accuracy: 8893/10000 (88.93%)

EPOCH: 29
Epoch: 29 Loss=0.4269 Batch_id=781 Accuracy=84.97: 100%|█| 782/782 [00:49<00
Original image type: <class 'PIL.Image.Image'>

Test set: Average loss: 0.3320, Accuracy: 8881/10000 (88.81%)

## Model Performance

The model uses several techniques to achieve good performance:
- OneCycleLR learning rate scheduling
- Progressive receptive field growth using dilated convolutions
- Efficient parameter usage with depthwise separable convolutions
- Effective regularization through data augmentation

## License

MIT License
