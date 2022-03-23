# Purpose
This script only purpose is to categorise the images according to their type provided in the dataset. Separate scripts are available for web (data) and surveillance (sv_data).

## Instructions 1 (data)
1. Place compcars_categorise.py in data folder 
```
.
├── compcars_categorise.py
├── image
├── label
├── misc
├── part
└── train_test_split
```

2. Install necessary packages (use pip if not sure)
    * conda: ``` conda install -c conda-forge scipy pandas numpy opencv tqdm ```
    * pip: ``` pip install scipy pandas numpy opencv-python tqdm ```

3. Open a terminal/cmd in the folder containing the files and run the following command
``` 
python compcars_categorise.py 
```

4. A new folder named **cropped** should appear and contain all the categorised images
```
.
├── compcars_categorise.py
├── cropped
├── image
├── label
├── misc
├── part
└── train_test_split
```

## Instructions 2 (sv_data)
1. Place compcars_categorise.py in data folder 
```
.
├── color_list.mat
├── compcars_sv_categorise.py
├── image
├── model_type.pkl              #this should be created automatically if you ran the script for data first else you can download it
├── README
├── sv_make_model_name.mat
├── test_surveillance.txt
└── train_surveillance.txt

```

2. Install necessary packages (use pip if not sure)
    * conda: ``` conda install -c conda-forge scipy pandas numpy opencv tqdm ```
    * pip: ``` pip install scipy pandas numpy opencv-python tqdm ```

3. Open a terminal/cmd in the folder containing the files and run the following command
``` 
python compcars_sv_categorise.py 
```

4. A new folder named **categorised** should appear and contain all the categorised images
```
.
├── categorised
├── color_list.mat
├── compcars_sv_categorise.py
├── image
├── model_type.pkl
├── README
├── sv_make_model_name.mat
├── test_surveillance.txt
└── train_surveillance.txt
```