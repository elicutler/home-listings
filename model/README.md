### Downloading data from Datafiniti

From `home-listings/model/` run `data_getter.py`. Supports the following options:
```
usage: data_getter.py [-h] [--num_records NUM_RECORDS] [--query_today_updates_only]
                      [--get_timeout_secs GET_TIMEOUT_SECS]
                      [--s3_subfolder S3_SUBFOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --num_records NUM_RECORDS, -n NUM_RECORDS
                        number of records to download from Datafiniti
  --query_today_updates_only, -q
                        only query listings updated today
  --get_timeout_secs GET_TIMEOUT_SECS, -g GET_TIMEOUT_SECS
                        maximum number of seconds to allow download attempt before
                        timing out
  --s3_subfolder S3_SUBFOLDER, -s S3_SUBFOLDER
                        s3 subdirectory within home-listings to download data to.
                        Typically "train", "val", or "test".
```

### Training a model

From `home-listings/model/` open `fit_and_deploy.py`. Open an associated Jupyter console and run the program. Within the program, `hypers` passes command-line arguments to `train.py`.
