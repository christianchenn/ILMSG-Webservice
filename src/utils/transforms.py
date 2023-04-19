from src.data.transforms.VideoTransformsV1 import VideoTransformsV1
from src.data.transforms.VideoTransformsV2 import VideoTransformsV2
from src.data.transforms.VideoTransformsV3 import VideoTransformsV3
from src.data.transforms.VideoTransformsV4 import VideoTransformsV4
# from src.data.transforms.VideoTransformsV5 import VideoTransformsV5
from src.data.transforms.VideoTransformsV6 import VideoTransformsV6
from src.data.transforms.VideoTransformsV7 import VideoTransformsV7


def get_video_transforms(version, color=False):
    if version == 1 or version == "VideoTransformsV1":
        return VideoTransformsV1().transforms()
    elif version == 2 or version == "VideoTransformsV2":
        return VideoTransformsV2().transforms()
    elif version == 3 or version == "VideoTransformsV3":
        return VideoTransformsV3().transforms()
    elif version == 4 or version == "VideoTransformsV4":
        return VideoTransformsV4(color=color).transforms()
    # elif version == 5 or version == "VideoTransformsV5":
    #     return VideoTransformsV5().transforms()
    elif version == 6 or version == "VideoTransformsV6":
        return VideoTransformsV6().transforms()
    elif version == 7 or version == "VideoTransformsV7":
        return VideoTransformsV7().transforms()