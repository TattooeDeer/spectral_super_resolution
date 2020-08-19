#!/usr/bin/env python
# coding: utf-8
import sys
import glob
import re
import pandas as pd
import numpy as np
import copy
import calendar

def create_dataset(save_dataset = False, return_dataset = True, output_path_filename = 'img_metadata.csv'):
   # Variables
    data_types = {
        'ACQUISITION_DATE': pd.DatetimeTZDtype(tz = 'UTC'),
        'WRS_PATH': 'int32',
        'WRS_ROW': 'int32'}

    files_txt = glob.glob(r'images/[0-9]*/*/*[.txt|.TXT]')
    mtl_regex = re.compile('(.*/.*MTL.*)')
    files_metadata = []

    corners = ['UL', 'UR', 'LL', 'LR']
    product_corners_eo = ['PRODUCT_{}_CORNER_LAT'.format(corner) for corner in corners] +\
                        ['PRODUCT_{}_CORNER_LON'.format(corner) for corner in corners]
    product_corners_landsat = ['CORNER_{}_LAT_PRODUCT'.format(corner) for corner in corners] +\
                        ['CORNER_{}_LON_PRODUCT'.format(corner) for corner in corners]

    # This dict is needed since the syntax for EO-1 products and Landsat 8 is different for the corners coords
    corners_dict = dict(zip(product_corners_eo, product_corners_landsat)) 

    meta_cols = ['SceneNumber', 'SPACECRAFT_ID', 'SENSOR_ID', 'ACQUISITION_DATE', 'PRODUCT_ID', 'WRS_PATH', 'WRS_ROW','PRODUCT_TYPE',
                'DATUM', 'MAP_PROJECTION'] +  product_corners_eo
    nScene_regex = re.compile('images/([0-9]*)/.*')


    df_metadata = pd.DataFrame(columns = meta_cols)
    
    # File reading
    ## get metadata filenames
    for file_path in files_txt:
        matched = mtl_regex.match(file_path)
        if matched is not None:
            files_metadata.append(matched.group())
    
    ## Read the metadata files of each image
    for file in files_metadata:
        #print('Reading {} ...\n'.format(file))
        f = open(file, "r")
        file_str = f.read()
        file_metadata = {}
        file_metadata['SceneNumber'] = int(nScene_regex.match(file).groups()[0])

        for col in meta_cols:
            pattern = re.compile('.*'+col+'.* = (.*)\\n.*')
            matched = pattern.findall(file_str)
            if len(matched) == 0:
                if col == 'WRS_PATH':
                    matched = re.findall('.*/EO1\w(\d{3})\d{3}.*', file)
                    file_metadata[col] = matched[0]
                if col == 'WRS_ROW':
                    matched = re.findall('.*/EO1\w\d{3}(\d{3}).*', file)
                    file_metadata[col] = matched[0]
                elif col == 'PRODUCT_TYPE':
                    pattern = re.compile('.*DATA_TYPE = (.*)\\n.*')
                    matched = pattern.findall(file_str)
                    file_metadata[col] = matched[0].strip('"')
                elif col == 'DATUM':
                    pattern = re.compile('.*REFERENCE_DATUM = (.*)\\n.*')
                    matched = pattern.findall(file_str)
                    file_metadata[col] = matched[0]
                elif col == 'PRODUCT_ID':
                    matched = re.findall('(EO[A-Z0-9]*)_.*', file)
                    file_metadata[col] = matched[0]
                elif 'CORNER' in col:
                    file_metadata[col] = re.findall('.*{} = (.*)\\n.*'.format(corners_dict[col]), file_str)[0]
                elif 'DATE' in col:
                    file_metadata[col] = re.findall('.*DATE_ACQUIRED = (.*)\\n.*', file_str)[0]
            else:
                file_metadata[col] = matched[0].strip('"')
        df_metadata = df_metadata.append(file_metadata, ignore_index = True)
        f.close()

    # Preprocessing code
    #df = create_dataset(files_metadata)
    col_corners = df_metadata.filter(regex = r'CORNER').columns

    data_types.update(dict(zip(col_corners, ['float64']*len(col_corners))))

    # Center of the image
    df_metadata = df_metadata.astype(data_types)
    df_metadata['CENTER_LAT'] = df_metadata.apply(lambda row: np.round(row.filter(regex = 'LAT').values.mean(), 4), axis = 1)
    df_metadata['CENTER_LON'] = df_metadata.apply(lambda row: np.round(row.filter(regex = 'LON').values.mean(), 4), axis = 1)

    '''
    In the atmospheric correction of FLAASH in ENVI, its required the MODTRAN model. On the FLAASH user guide various methods are showed for
    determining the corresponding atmospheric modelo to use. This script will calculate the model based in the latitude of the center point
    of the image and the month of the year.
    '''

    months_dict = dict((k,v) for k,v in enumerate(calendar.month_name))
    modtran_atm = pd.read_csv('MODTRAN atmospheres.csv', index_col = 'Latitude(°N)')

    df_metadata['MODTRAN_ATMOSPHERE'] = df_metadata.apply(lambda row: modtran_atm.loc[
                                                                np.round((row.CENTER_LAT),-1).astype(int),
                                                                months_dict[row.ACQUISITION_DATE.month]],
                                                                axis = 1)
    df_metadata.sort_values(by = ['SceneNumber'], inplace = True)
    if save_dataset is True:
        print('save')
        df_metadata.to_csv(output_path_filename, index = None)
    if return_dataset is True:
        return df_metadata



if __name__ == '__main__':
    create_dataset(save_dataset=True)