import pandas as pd
import numpy as np
import math
from tqdm import tqdm


tqdm.pandas()


# Convert Accelerometer reading to g units, 8bit data with a range of +-2g #
def convert_acc(x):
    x = int(x,16)
    if x>127:
        x = x-256
    return np.float64(x*2/128)


# Convert Magnetometer reading to uT units, 16bit data with a range of +-1200uT #
def convert_mag(x):
    x = int(x,16)
    if x>32767:
        x = x-65536
    return np.float64(x*0.0366)


# Convert a single row of Accelerometer and Magnetometer data to engineering units and return a df #
def convert_acc_mag_row(row):
    # Initially the data was gathered without magnetometer, so check length for identification #
    data_dict = {'x': [], 'y': [], 'z': []}
    if len(row) == 162:
        data_dict['mx'] = convert_mag(row[:4])
        data_dict['my'] = convert_mag(row[4:8])
        data_dict['mz'] = convert_mag(row[8:12])
        row = row[12:]
        for i in range(0, len(row), 6):
            data_dict['x'].append(convert_acc(row[i:i+2]))
            data_dict['y'].append(convert_acc(row[i+2:i+4]))
            data_dict['z'].append(convert_acc(row[i+4:i+6]))
    return data_dict


def calcualte_rotation_factor(idle_df):
    # We need data for the device when the vehicle is static
    [mx,my,mz] = idle_df[['mx', 'my', 'mz']].mean().tolist()
    [x,y,z] = idle_df[['x', 'y', 'z']].iloc[0].apply(np.mean).tolist()

    phi = math.atan(y/z)
    theta = math.atan(-x/(y*math.sin(phi)+z*math.cos(phi)))
    psi = math.atan((mz*math.sin(phi) - my*math.cos(phi))/(mx*math.cos(theta) + my*math.sin(theta)*math.sin(phi) + mz*math.sin(theta)*math.cos(phi)))

    Rx = np.array([[1,0,0], [0,math.cos(phi),math.sin(phi)], [0,-math.sin(phi),math.cos(phi)]])
    Ry = np.array([[math.cos(theta),0,-math.sin(theta)], [0,1,0], [math.sin(theta),0,math.cos(theta)]])
    Rz = np.array([[math.cos(phi),math.sin(phi),0], [-math.sin(phi),math.cos(phi),0], [0,0,1]])

    R = Rx.dot(Ry).dot(Rz)
    R = np.linalg.inv(R)
    
    return R


def decode_acc_hex(acc_hex, speed):
    """ Author: Qirun Chen
    """
    
    # Decode acc hex. It returns a series with a dictionary on each index.
    # apply(pd.Series) to unpact the dictionaries and convert the series to a dataframe
    print("Decoding each accelerometer hex code.")
    decoded_acc_df = acc_hex.apply(convert_acc_mag_row)\
                            .progress_apply(pd.Series)
    
    # Calculate the rotation factor using idling acc
    idle_index = (speed <= 1)
    idle_df = decoded_acc_df[idle_index]
    R = calcualte_rotation_factor(idle_df)
    
#     return decoded_acc_df[['x', 'y', 'z']]
    
    # Each row is a timestamp. Since the accelerometer sensor is at 25Hz, it means 1 timestamp has 25 mini steps.
    rotated_acc_list = list()
    for x_s, y_s, z_s in tqdm(decoded_acc_df[['x', 'y', 'z']].values, 
                              desc="Rotating accelerometer signals."):
        
        # x_s, y_s, and z_s are a list of 25 mini steps.  
        rotated_acc_mini_step_list = list()
        
        # We zip them together so that we can rotate each mini step.
        for x, y, z in zip(x_s, y_s, z_s):
            # Apply rotation matrix and store the values
            A = np.array([[x], [y], [z]])
            B = np.dot(R,A)
            rotated_acc_mini_step_list.append(B[:, 0])
        
        rotated_acc = np.asarray(rotated_acc_mini_step_list)#.mean(axis=0)
        rotated_acc_list.append({'acc_x':rotated_acc[:, 0], 'acc_y':rotated_acc[:, 1], 'acc_z':rotated_acc[:, 2]})     
    
    return pd.DataFrame(rotated_acc_list).apply(pd.Series)

    