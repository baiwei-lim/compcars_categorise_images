import scipy.io as sio
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def mk_df_series(image_parent_path, label_parent_path, attributes_path, car_type_path):
    img_props_df = pd.DataFrame({
        'img_path' : sorted(image_parent_path.rglob('*.*')), #always sort glob
        'lab_path' : sorted(label_parent_path.rglob('*.*'))
        })
    
    attributes_df = pd.read_csv(attributes_path, sep=' ', index_col=('model_id'))
    
    #load mat as dict
    car_type_mat = sio.loadmat(car_type_path, simplify_cells=True)
    car_type_ser = pd.Series(np.concatenate(([np.nan], car_type_mat['types']))) #add nan for unknown types

    return img_props_df, attributes_df, car_type_ser

def add_model_type(img_props_df, attributes_df, car_type_ser):
    #get model id from path
    img_props_df['model_id'] = img_props_df['img_path'].apply(lambda fpath: fpath.parts[2]).astype(int)

    #use model id to get car type **ID** from attributes df
    img_props_df['type_id'] = img_props_df['model_id'].apply(lambda model_id: attributes_df.loc[model_id, 'type'])

    #use type id to get type name
    img_props_df['car_type'] =  img_props_df['type_id'].apply(lambda type_id: car_type_ser.loc[type_id])

def pickle_model_type(img_props_df):
    #save model id with type for use in surveillance dataset
    sv_data_path = Path.cwd().parent.joinpath('sv_data')
    img_props_df.loc[:,['model_id','car_type']].drop_duplicates().set_index('model_id').to_pickle(sv_data_path.joinpath('model_type.pkl'))

def df_dropna(img_props_df, car_type_ser):
    #imgs w/ missing type: 38884, of those, corresponding models w/ missing type: 748
    #drop missing vals in df and rmv 'nan' type from ser
    img_props_df.dropna(inplace=True)
    car_type_ser.drop(0, inplace=True)

def mk_output_subdirs(output_parent, car_type_ser):
    #mk output dir
    if output_parent.exists() != True:
        output_parent.mkdir()
        
    #mk dict containing path to each subdir
    output_subdirs = dict(zip(car_type_ser, [output_parent.joinpath(car_type) for car_type in car_type_ser]))
    for subdir in output_subdirs.values():
        if subdir.exists() != True:
            subdir.mkdir()
    
    return output_subdirs

#func to get coords to crop images
def get_coords(label):
    coords = next(line for pos, line in enumerate(label) if pos == 2)
    return [int(n) for n in coords.split()] #conv str to int

#func to crop img
def crop_write_img(img_path, lab_path, car_type, output_subdirs):
    img = cv2.imread(str(img_path))
    with lab_path.open() as label:
        xmin, ymin, xmax, ymax = get_coords(label)
    cropped = img[ymin:ymax, xmin:xmax]
    
    subdir = output_subdirs[car_type]
    if subdir.joinpath(f'{img_path.name}').exists():
        pass
    else:    
        cv2.imwrite(str(subdir.joinpath(f'{img_path.name}')), cropped)

def main():
    #file paths
    image_parent_path = Path('./image')
    label_parent_path = Path('./label')
    attributes_path = Path('./misc/attributes.txt')
    car_type_path = Path('./misc/car_type.mat')
    output_parent = Path('./cropped')
    
    #preparation
    img_props_df, attributes_df, car_type_ser = mk_df_series(
        image_parent_path, label_parent_path, attributes_path, car_type_path
        )
    add_model_type(img_props_df, attributes_df, car_type_ser)
    pickle_model_type(img_props_df)
    df_dropna(img_props_df, car_type_ser)
    output_subdirs = mk_output_subdirs(output_parent, car_type_ser)
            
    #extract imgs
    tqdm.pandas(desc='Extracting')
    img_props_df.progress_apply(lambda row: crop_write_img(
        row['img_path'], row['lab_path'], row['car_type'], output_subdirs
        ), axis=1)

if __name__ == '__main__':
    main()

