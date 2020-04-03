# paddle_seven_days - day4：口罩识别

## 结果展示



| model | data aug | acc |
| -------- | -------- | -------- |
| vgg16     | None     |  1.0    |
| resnet18     | None     |  0.875    |
| vgg16     | RICAP     |   0.9375   |
| resnet18     | RICAP     |  0.875    |
| vgg16     | MIXUP     |   1.0   |
| resnet18     | MIXUP     |  0.875    |

> resnet没调好

### 1. vgg16
![](./vgg16.png)

### 2. vgg16 + ricap
![](./vgg16_ricap.png)

### 3.vgg16 + mixup
![](./vgg16_mixup.png)