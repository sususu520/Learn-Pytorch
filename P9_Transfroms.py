from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -》 tensor数据类型
# 通过 transforms.TOTensor去看两个问题

img_path = "data_set/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")


# 1， transforms该如何被使用(python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()