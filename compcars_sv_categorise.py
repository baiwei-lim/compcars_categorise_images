import scipy.io as sio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil

def mk_df(imgs_path, props_path, model_type_pkl):
    #mk df of img paths
    sv_img_props_df = pd.DataFrame({'img_path': sorted(imgs_path.rglob('*.*'))})
    #add sv car id to df (described as surveillance_model_id in README)
    #changed the name as it is confusing to have surveillance_model_id 
    #in the name when the properties also contain model_id
    sv_img_props_df['sv_car_id'] = sv_img_props_df['img_path'].apply(lambda img_path: img_path.parts[1]).astype(int) #convert to int frm str

    #load mat
    props_mat = sio.loadmat(props_path, simplify_cells=True)
    #mk props df
    props_df = pd.DataFrame(props_mat['sv_make_model_name'], columns=['make','model','model_id'])
    props_df.index += 1
    props_df.index.name = 'sv_car_id' #each index refers to sv_car_id in sv_img_props_df therefore renamed index

    #add car model id into df
    sv_img_props_df['model_id'] = sv_img_props_df['sv_car_id'].apply(lambda sv_car_id: props_df.loc[sv_car_id, 'model_id']) #use loc as index changed

    #read in model_type pickle 
    model_type_df =  pd.read_pickle(model_type_pkl)

    #use model_type_df to assign type to each image for categorising
    sv_img_props_df['car_type'] = sv_img_props_df['model_id'].apply(lambda model_id: model_type_df.loc[model_id]) #use loc as index is non-zero
    return sv_img_props_df

def df_drop(sv_img_props_df):
    #nan: 2843
    #drop nan types
    sv_img_props_df.dropna(inplace=True)

def mk_output_subdirs(sv_img_props_df, output_parent_path):
    #mk output dir
    if output_parent_path.exists() != True:
        output_parent_path.mkdir()

    #list of subdirs
    car_types = sv_img_props_df['car_type'].unique()
    #mk subdirs
    output_subdirs = dict(zip(car_types, [output_parent_path.joinpath(car_type) for car_type in car_types]))
    for subdir in output_subdirs.values():
        if subdir.exists() != True:
            subdir.mkdir()

    return output_subdirs

#func to categorize images
def cp_imgs(img_path, car_type, output_subdirs):
    shutil.copy2(img_path, output_subdirs[car_type])

def main():
    #file paths
    imgs_path = Path('./image')
    props_path = Path('./sv_make_model_name.mat')
    model_type_pkl = Path('./model_type.pkl')
    output_parent_path = Path('./categorised')
    
    #preparation
    sv_img_props_df = mk_df(imgs_path, props_path, model_type_pkl)
    df_drop(sv_img_props_df)
    output_subdirs = mk_output_subdirs(sv_img_props_df, output_parent_path)
    
    #copy images
    tqdm.pandas(desc='Catergorising')
    sv_img_props_df.progress_apply(lambda row: cp_imgs(
        row['img_path'], row['car_type'], output_subdirs
        ), axis=1)
    
if __name__ == '__main__':
    main()