from src.utils.video import stretch_contrast
from torchvision import transforms


class VideoTransformsV6:
    def __init__(self, p1=2, p2=98, normalize=True):
        self.p1 = p1
        self.p2 = p2
        self.normalize = normalize

    def transforms(self):
        transforms_list = [
            # transforms.Lambda(lambda x: stretch_contrast(x, (self.p1, self.p2))),
            transforms.Lambda(lambda x: x.unsqueeze(0) / 255),
            # transforms.Resize((132, 132))
        ]
        if self.normalize:
            transforms_list.append(transforms.Normalize([0.5], [0.5]))
        return transforms.Compose(transforms_list)
