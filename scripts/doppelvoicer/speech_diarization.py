from pathlib import Path

import boto3
import botocore
import pandas as pd

if __name__ == '__main__':
    bucket_name = 'doppelvoicer'
    raw_data_root_dir = \
        Path('~/git/swiss-ml-lib/data/doppelvoicer/raw').expanduser()
    s3_info_file = Path('~/Documents/rootKey.csv').expanduser()
    s3_info = pd.read_csv(s3_info_file)

    access_key, secret_key = s3_info.values[0, 0], s3_info.values[1, 0]

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket_prefix = "historiepodden/raw"
    for obj in bucket.objects.all():
        if 'historiepodden/raw/raw_' in obj.key:
            print(obj)
