# Seed Detector

Software for detecting/measuring and statistics generation for seeding distribution.

To install the python dependencies use:

```shell
$ pip install -r requirements.txt
```

#### Usage:

```console
$ ./seed_tracker [-f | --file=] [-k | --kernel=] [-s | --scale=] [-p | --pixel=] [-c] [-h] [-d | --dir=]
```

#### Flags & key-binds:

* `[-f | --file=]` - The file name to be used as the video source. Use `-d` to use a hardware device.
* `[-k | --kernel=]` - The size of the kernel to be used to blur the image, this number must be odd.
* `[-s | --scale=]` - How much to scale the image down, in 1 to 100% scale 100 being max.
* `[-p | --pixel=]` - How many pixels are equal to 1cm. Defaults to 10.
* `[-c]` - Will start the program in calibration mode.
* `[-h]` - Use a hardware device indicated by `--file=`.
* `[-d | --dir=]` - The direction of the videos. Can be: up, down, left or right.

| key-bind | function                                                                 |
|:--------:|:-------------------------------------------------------------------------|
| q        | Exits the program.                                                       |
| p        | Pauses the video.                                                        |
| r        | Displays the results, this will happen automatically if the videos ends. |
| s        | Starts the program (if in calibration mode).                              |
