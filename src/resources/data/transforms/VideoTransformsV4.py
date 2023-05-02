from src.utils.video import stretch_contrast
from torchvision import transforms


class VideoTransformsV4:
    def __init__(self, p1=2, p2=98, normalize=True, color=False):
        self.p1 = p1
        self.p2 = p2
        self.normalize = normalize
        self.color = color

    def transforms(self):
        def norm(x, color):
            x = x/255
            if color:
                return x.permute(2, 0, 1)
            return x.unsqueeze(0)
        size = (128, 128)
        transforms_list = [
            # transforms.Lambda(lambda x: stretch_contrast(x, (self.p1, self.p2))),
            transforms.Lambda(lambda x: norm(x, self.color)),
            transforms.Resize((size))
        ]
        if self.normalize:
            transforms_list.append(transforms.Normalize([0.5], [0.5]))
        return transforms.Compose(transforms_list)
