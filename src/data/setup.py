import gdown


def download_dataset(url, output):
    url = "https://drive.google.com/file/d/1-6-bYH5y9WO1AUyNNDI0rVNydrkeV7Qb/view?usp=share_link"
    output = "/home/mlpc1/Chen/TA/ILMSG/data/results-wanita.zip"
    return gdown.download(url=url, output=output, quiet=False, fuzzy=True)
