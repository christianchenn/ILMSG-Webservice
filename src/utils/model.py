import pickle
from tqdm import tqdm
import os
import re

from src.models.audio.v1.AudioAEV34 import AudioAEV34
from src.models.audio.v1.AudioAEV10 import AudioAEV10
from src.models.audio.v1.AudioAEV11 import AudioAEV11
from src.models.audio.v1.AudioAEV12 import AudioAEV12
from src.models.audio.v1.AudioAEV13 import AudioAEV13
from src.models.audio.v1.AudioAEV14 import AudioAEV14
from src.models.audio.v1.AudioAEV15 import AudioAEV15
from src.models.audio.v1.AudioAEV16 import AudioAEV16
from src.models.audio.v1.AudioAEV17 import AudioAEV17
from src.models.audio.v1.AudioAEV18 import AudioAEV18
from src.models.audio.v1.AudioAEV19 import AudioAEV19
from src.models.audio.v1.AudioAEV20 import AudioAEV20
from src.models.audio.v1.AudioAEV21 import AudioAEV21
from src.models.audio.v1.AudioAEV22 import AudioAEV22
from src.models.audio.v1.AudioAEV23 import AudioAEV23
from src.models.audio.v1.AudioAEV24 import AudioAEV24
from src.models.audio.v1.AudioAEV25 import AudioAEV25
from src.models.audio.v1.AudioAEV26 import AudioAEV26
from src.models.audio.v1.AudioAEV27 import AudioAEV27
from src.models.audio.v1.AudioAEV28 import AudioAEV28
from src.models.audio.v1.AudioAEV29 import AudioAEV29
from src.models.audio.v1.AudioAEV3 import AudioAEV3
from src.models.audio.v1.AudioAEV30 import AudioAEV30
from src.models.audio.v1.AudioAEV31 import AudioAEV31
from src.models.audio.v1.AudioAEV32 import AudioAEV32
from src.models.audio.v1.AudioAEV33 import AudioAEV33
from src.models.audio.v1.AudioAEV35 import AudioAEV35
from src.models.audio.v1.AudioAEV36 import AudioAEV36
from src.models.audio.v2.AudioAEV37 import AudioAEV37
from src.models.audio.v2.AudioAEV38 import AudioAEV38
from src.models.audio.v2.AudioAEV39 import AudioAEV39
from src.models.audio.v1.AudioAEV4 import AudioAEV4
from src.models.audio.v2.AudioAEV40 import AudioAEV40
from src.models.audio.v2.AudioAEV41 import AudioAEV41
from src.models.audio.v2.AudioAEV42 import AudioAEV42
from src.models.audio.v1.AudioAEV5 import AudioAEV5
from src.models.audio.v1.AudioAEV6 import AudioAEV6
from src.models.audio.v1.AudioAEV7 import AudioAEV7
from src.models.audio.v1.AudioAEV8 import AudioAEV8
from src.models.audio.v1.AudioAEV2 import AudioAEV2
from src.models.audio.v1.AudioAEV1 import AudioAEV1
from src.models.audio.v1.AudioAEV9 import AudioAEV9
from src.models.audio.v2.AudioAEV43 import AudioAEV43
from src.models.audio.v2.AudioAEV44 import AudioAEV44
from src.models.audio.v2.AudioAEV45 import AudioAEV45
from src.models.audio.v2.AudioAEV46 import AudioAEV46
from src.models.audiov2.AudioV2AEV1 import AudioV2AEV1
from src.models.audiov2.AudioV2AEV2 import AudioV2AEV2


def get_audio_model(version, lr=None, model_name=None, yaml_file=None):
    if version == 1 or version == "AudioAEV1":
        return AudioAEV1(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 2 or version == "AudioAEV2":
        return AudioAEV2(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 3 or version == "AudioAEV3":
        return AudioAEV3(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 4 or version == "AudioAEV4":
        return AudioAEV4(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 5 or version == "AudioAEV5":
        return AudioAEV5(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 6 or version == "AudioAEV6":
        return AudioAEV6(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 7 or version == "AudioAEV7":
        return AudioAEV7(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 8 or version == "AudioAEV8":
        return AudioAEV8(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 9 or version == "AudioAEV9":
        return AudioAEV9(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 10 or version == "AudioAEV10":
        return AudioAEV10(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 11 or version == "AudioAEV11":
        return AudioAEV11(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 12 or version == "AudioAEV12":
        return AudioAEV12(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 13 or version == "AudioAEV13":
        return AudioAEV13(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 14 or version == "AudioAEV14":
        return AudioAEV14(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 15 or version == "AudioAEV15":
        return AudioAEV15(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 16 or version == "AudioAEV16":
        return AudioAEV16(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 17 or version == "AudioAEV17":
        return AudioAEV17(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 18 or version == "AudioAEV18":
        return AudioAEV18(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 19 or version == "AudioAEV19":
        return AudioAEV19(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 20 or version == "AudioAEV20":
        return AudioAEV20(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 21 or version == "AudioAEV21":
        return AudioAEV21(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 22 or version == "AudioAEV22":
        return AudioAEV22(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 23 or version == "AudioAEV23":
        return AudioAEV23(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 24 or version == "AudioAEV24":
        return AudioAEV24(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 25 or version == "AudioAEV25":
        return AudioAEV25(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 26 or version == "AudioAEV26":
        return AudioAEV26(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 27 or version == "AudioAEV27":
        return AudioAEV27(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 28 or version == "AudioAEV28":
        return AudioAEV28(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 29 or version == "AudioAEV29":
        return AudioAEV29(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 30 or version == "AudioAEV30":
        return AudioAEV30(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 31 or version == "AudioAEV31":
        return AudioAEV31(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 32 or version == "AudioAEV32":
        return AudioAEV32(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 33 or version == "AudioAEV33":
        return AudioAEV33(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 34 or version == "AudioAEV34":
        return AudioAEV34(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 35 or version == "AudioAEV35":
        return AudioAEV35(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 36 or version == "AudioAEV36":
        return AudioAEV36(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 37 or version == "AudioAEV37":
        return AudioAEV37(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 38 or version == "AudioAEV38":
        return AudioAEV38(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 39 or version == "AudioAEV39":
        return AudioAEV39(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 40 or version == "AudioAEV40":
        return AudioAEV40(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 41 or version == "AudioAEV41":
        return AudioAEV41(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 42 or version == "AudioAEV42":
        return AudioAEV42(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 43 or version == "AudioAEV43":
        return AudioAEV43(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 44 or version == "AudioAEV44":
        return AudioAEV44(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 45 or version == "AudioAEV45":
        return AudioAEV45(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 46 or version == "AudioAEV46":
        return AudioAEV46(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )


def get_audio_v2_model(version, lr=None, model_name=None, yaml_file=None):
    if version == 1 or version == "AudioV2AEV1":
        return AudioV2AEV1(
            learning_rate=lr,
            run_name=model_name
        )
    elif version == 2 or version == "AudioV2AEV2":
        return AudioV2AEV2(
            learning_rate=lr,
            run_name=model_name
        )
    # elif version == 3 or version == "AudioV2AEV3":
    #     return AudioV2AEV3(
    #         learning_rate=lr,
    #         run_name=model_name
    #     )


def get_visual_model(version, lr=None, model_name=None, yaml_file=None):
    from src.models.video.v1.Vid2SpeechV1 import Vid2SpeechV1
    from src.models.video.v1.Vid2SpeechV2 import Vid2SpeechV2
    from src.models.video.v1.Vid2SpeechV3 import Vid2SpeechV3
    from src.models.video.v1.Vid2SpeechV4 import Vid2SpeechV4
    from src.models.video.v1.Vid2SpeechV5 import Vid2SpeechV5
    from src.models.video.v1.Vid2SpeechV6 import Vid2SpeechV6
    from src.models.video.v1.Vid2SpeechV7 import Vid2SpeechV7
    from src.models.video.v1.Vid2SpeechV8 import Vid2SpeechV8
    from src.models.video.v1.Vid2SpeechV9 import Vid2SpeechV9
    from src.models.video.v1.Vid2SpeechV10 import Vid2SpeechV10
    from src.models.video.v1.Vid2SpeechV11 import Vid2SpeechV11
    from src.models.video.v1.Vid2SpeechV12 import Vid2SpeechV12
    from src.models.video.v1.Vid2SpeechV13 import Vid2SpeechV13
    from src.models.video.v1.Vid2SpeechV14 import Vid2SpeechV14
    from src.models.video.v1.Vid2SpeechV15 import Vid2SpeechV15
    from src.models.video.v1.Vid2SpeechV16 import Vid2SpeechV16
    from src.models.video.v1.Vid2SpeechV17 import Vid2SpeechV17
    from src.models.video.v1.Vid2SpeechV18 import Vid2SpeechV18
    from src.models.video.v1.Vid2SpeechV19 import Vid2SpeechV19
    from src.models.video.v1.Vid2SpeechV20 import Vid2SpeechV20
    from src.models.video.v2.Vid2SpeechV21 import Vid2SpeechV21
    from src.models.video.v2.Vid2SpeechV22 import Vid2SpeechV22
    from src.models.video.v2.Vid2SpeechV23 import Vid2SpeechV23
    from src.models.video.v2.Vid2SpeechV24 import Vid2SpeechV24
    from src.models.video.v2.Vid2SpeechV25 import Vid2SpeechV25
    from src.models.video.v2.Vid2SpeechV26 import Vid2SpeechV26
    from src.models.video.v2.Vid2SpeechV27 import Vid2SpeechV27
    from src.models.video.v2.Vid2SpeechV28 import Vid2SpeechV28
    from src.models.video.v2.Vid2SpeechV29 import Vid2SpeechV29
    from src.models.video.v2.Vid2SpeechV30 import Vid2SpeechV30
    from src.models.video.v2.Vid2SpeechV31 import Vid2SpeechV31
    from src.models.video.v2.Vid2SpeechV32 import Vid2SpeechV32
    from src.models.video.v2.Vid2SpeechV33 import Vid2SpeechV33
    from src.models.video.v2.Vid2SpeechV34 import Vid2SpeechV34
    from src.models.video.v2.Vid2SpeechV35 import Vid2SpeechV35
    from src.models.video.v2.Vid2SpeechV36 import Vid2SpeechV36
    from src.models.video.v2.Vid2SpeechV37 import Vid2SpeechV37
    from src.models.video.v2.Vid2SpeechV38 import Vid2SpeechV38
    from src.models.video.v2.Vid2SpeechV39 import Vid2SpeechV39
    from src.models.video.v2.Vid2SpeechV40 import Vid2SpeechV40
    from src.models.video.v2.Vid2SpeechV41 import Vid2SpeechV41
    from src.models.video.v2.Vid2SpeechV42 import Vid2SpeechV42
    from src.models.video.v2.Vid2SpeechV43 import Vid2SpeechV43
    from src.models.video.v2.Vid2SpeechV44 import Vid2SpeechV44
    from src.models.video.v2.Vid2SpeechV45 import Vid2SpeechV45
    from src.models.video.v2.Vid2SpeechV46 import Vid2SpeechV46
    from src.models.video.v2.Vid2SpeechV47 import Vid2SpeechV47
    from src.models.video.v2.Vid2SpeechV48 import Vid2SpeechV48
    from src.models.video.v2.Vid2SpeechV49 import Vid2SpeechV49
    from src.models.video.v2.Vid2SpeechV50 import Vid2SpeechV50

    if version == 1 or version == "Vid2SpeechV1":
        return Vid2SpeechV1(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 2 or version == "Vid2SpeechV2":
        return Vid2SpeechV2(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 3 or version == "Vid2SpeechV3":
        return Vid2SpeechV3(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 4 or version == "Vid2SpeechV4":
        return Vid2SpeechV4(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 5 or version == "Vid2SpeechV5":
        return Vid2SpeechV5(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 6 or version == "Vid2SpeechV6":
        return Vid2SpeechV6(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 7 or version == "Vid2SpeechV7":
        return Vid2SpeechV7(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 8 or version == "Vid2SpeechV8":
        return Vid2SpeechV8(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 9 or version == "Vid2SpeechV9":
        return Vid2SpeechV9(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 10 or version == "Vid2SpeechV10":
        return Vid2SpeechV10(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 11 or version == "Vid2SpeechV11":
        return Vid2SpeechV11(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 12 or version == "Vid2SpeechV12":
        return Vid2SpeechV12(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 13 or version == "Vid2SpeechV13":
        return Vid2SpeechV13(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 14 or version == "Vid2SpeechV14":
        return Vid2SpeechV14(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 15 or version == "Vid2SpeechV15":
        return Vid2SpeechV15(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 16 or version == "Vid2SpeechV16":
        return Vid2SpeechV16(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 17 or version == "Vid2SpeechV17":
        return Vid2SpeechV17(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 18 or version == "Vid2SpeechV18":
        return Vid2SpeechV18(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 19 or version == "Vid2SpeechV19":
        return Vid2SpeechV19(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 20 or version == "Vid2SpeechV20":
        return Vid2SpeechV20(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 21 or version == "Vid2SpeechV21":
        return Vid2SpeechV21(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 22 or version == "Vid2SpeechV22":
        return Vid2SpeechV22(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 23 or version == "Vid2SpeechV23":
        return Vid2SpeechV23(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 24 or version == "Vid2SpeechV24":
        return Vid2SpeechV24(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 25 or version == "Vid2SpeechV25":
        return Vid2SpeechV25(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 26 or version == "Vid2SpeechV26":
        return Vid2SpeechV26(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 27 or version == "Vid2SpeechV27":
        return Vid2SpeechV27(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 28 or version == "Vid2SpeechV28":
        return Vid2SpeechV28(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 29 or version == "Vid2SpeechV29":
        return Vid2SpeechV29(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 30 or version == "Vid2SpeechV30":
        return Vid2SpeechV30(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 31 or version == "Vid2SpeechV31":
        return Vid2SpeechV31(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 32 or version == "Vid2SpeechV32":
        return Vid2SpeechV32(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 33 or version == "Vid2SpeechV33":
        return Vid2SpeechV33(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 34 or version == "Vid2SpeechV34":
        return Vid2SpeechV34(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 35 or version == "Vid2SpeechV35":
        return Vid2SpeechV35(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 36 or version == "Vid2SpeechV36":
        return Vid2SpeechV36(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 37 or version == "Vid2SpeechV37":
        return Vid2SpeechV37(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 38 or version == "Vid2SpeechV38":
        return Vid2SpeechV38(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 39 or version == "Vid2SpeechV39":
        return Vid2SpeechV39(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 40 or version == "Vid2SpeechV40":
        return Vid2SpeechV40(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 41 or version == "Vid2SpeechV41":
        return Vid2SpeechV41(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 42 or version == "Vid2SpeechV42":
        return Vid2SpeechV42(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 43 or version == "Vid2SpeechV43":
        return Vid2SpeechV43(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 44 or version == "Vid2SpeechV44":
        return Vid2SpeechV44(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 45 or version == "Vid2SpeechV45":
        return Vid2SpeechV45(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 46 or version == "Vid2SpeechV46":
        return Vid2SpeechV46(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 47 or version == "Vid2SpeechV47":
        return Vid2SpeechV47(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 48 or version == "Vid2SpeechV48":
        return Vid2SpeechV48(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 49 or version == "Vid2SpeechV49":
        return Vid2SpeechV49(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )
    elif version == 50 or version == "Vid2SpeechV50":
        return Vid2SpeechV50(
            learning_rate=lr,
            run_name=model_name,
            yaml_file=yaml_file
        )


def find_ckpt(dir, key='val_loss', order="lowest", idx=-1):
    # pattern = re.compile(fr"{key}=([/d.]+)")
    pattern = re.compile(fr"{key}=([\d.]+)")

    def extract(filename):
        match = pattern.search(filename)
        end = float("inf") if order == "lowest" else float("-inf")
        return float(match.group(1).rstrip(".")) if match else end

    files = os.listdir(dir)

    sorted_files = sorted(files, key=extract)

    if idx <= -1:
        return sorted_files[0]
    return sorted_files[idx]


def predict_audio_labels(dataset, output_dir, frame_length, model_version, model_name, model_ckpt=None):
    cwd = os.getcwd()
    ckpt_path = f"{cwd}/models/ilmsg-audio/f{frame_length}/{model_name}".replace("\\", "/")
    ckpt_file = find_ckpt(ckpt_path, "val_loss", "lowest")
    print(ckpt_file)
    model = get_audio_model(int(model_version)).load_from_checkpoint(f"{ckpt_path}/{ckpt_file}")
    # print(model_name)
    # model = get_audio_model(model_name).load_from_checkpoint(
    #     f"D:/College/Tugas_Akhir/ILMSG/data/temp/ilmsg-audioae=epoch=205-step=14214-val_loss=0.5634.ckpt")
    model = model.cuda()

    encoder = model.encoder
    encoder.eval()
    for i in tqdm(range(len(dataset))):
        data, filepath = dataset.item(i)
        basename = filepath.split("/")[-1:][0]
        arr = basename.split("_")
        _range = arr[2].split("-")
        _range[0] = str(int(_range[0]) // 640)
        _range[1] = str(int(_range[1].split(".")[0]) // 640)
        arr[2] = "-".join(_range)
        new_basename = "_".join(arr)
        latent = encoder(data.unsqueeze(1)).squeeze()
        output_dir = output_dir.replace('\\', '/')
        _path = f"{output_dir}/{new_basename}.pkl"
        # print(_path)
        with open(_path, 'wb') as f:
            pickle.dump(latent.detach().cpu().numpy(), f)


def extract_model(model, start_layer, end_layer, freeze_until):
    from torch import nn
    # Create a new model with only the specified layers
    new_model = nn.Sequential()

    # Find the start and end layers in the model
    valid = False
    for name, module in model.named_modules():
        print("NAME:", name)
        print("start_layer", start_layer)
        if start_layer == name:
            valid = True
        if end_layer == name:
            break
        if start_layer == name or valid:
            new_model.add_module(name.replace(".", "-"), module)

    for name, param in new_model.named_parameters():
        if freeze_until == name:
            break
        param.requires_grad = False
    return new_model


def extract_layers(model, start_layer_idx, end_layer_idx):
    from torch import nn
    # Create a new model with only the desired layers
    new_model = nn.Sequential(*list(model.children())[start_layer_idx:end_layer_idx])
    return new_model


def freeze_layers(model, freeze_until=None):
    if freeze_until is not None:
        for param in model.parameters():
            param.requires_grad = True
        if not None:
            for name, param in model.named_parameters():
                print(name)
                if name == freeze_until:
                    break
                param.requires_grad = False
    return model
