# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:02:52 2020

@author: Faroud
"""

"""
This class read and stock he content 
of a hand pose inside a variable 
"""

# %% Imports

import glob
import numpy as np
import pandas as pd


#%%
HAND_POSES_DIRECTORY = '../../PositionHandJoints-20201021/PositionHandJoints/'
#%%

class HandsData():
    def __init__(self, left_hand_data_, right_hand_data_, classes_dict_):
        self.left_hand_data =  left_hand_data_
        self.right_hand_data = right_hand_data_
        self.classes_dict = classes_dict_
        
    def fuze_all_hands_data(self):
        return pd.concat([self.left_hand_data, self.right_hand_data], ignore_index=True)
    
# %%
class Parser():
    def __init__(self, hand_poses_directory_=HAND_POSES_DIRECTORY):
        self.hand_poses_directory = hand_poses_directory_
        self.HEADER_SPE_CHAR = '_'
        # the header in csv file of a left hand data is different from 
        # the one in the right hand data file. Here the idea is to define 
        # a normalized hader name for all the data weither it comes from the left or the right
        #self.hader_names = 
        

    def _load_a_hand_poses_file_path(self, hande_code: str ):
        """
        Load paths of the files containing all the specified hand 
        poses.

        Parameters
        ----------
        hande_code : str
            String targeting a hand: 
                "LH" for left hand 
                "RH" for right hand

        Returns
        -------
        List
            List of path.
        """        
        return glob.glob(self.hand_poses_directory+hande_code+'*')
    
    
    def _load_all_left_hand_poses_file_path(self):
        """
        Load file paths of left hand poses
        
        Returns
        -------
        List
            List of str representing the list of path.
        """
        return self._load_a_hand_poses_file_path('LH')
    
    
    def _load_all_right_hand_poses_file_path(self):
        """
        Load file paths of left hand poses
        
        Returns
        -------
        List
            List of str representing the list of path.
        """
        return self.__load_a_hand_poses_file_path('RH')
    
    
    def _normalize_header(self, header):
        return [a_column_name.split(self.HEADER_SPE_CHAR, 1)[1] for a_column_name in header ]

    
    def _load_all_data_by_hand(self, hand_code: str):
        if (hand_code not in ['L', 'R']):
            print('Bad hand code, should be L or R')
        else:
            a_hand_files = self._load_a_hand_poses_file_path(hand_code)
            print(f'First {hand_code} hand file: {a_hand_files[0]}')
            print(f'Last {hand_code} hand file: {a_hand_files[-1]}')
            #
            def inside_rec(index):
                if(index == -1):
                    return 
                else:
                    df = pd.read_csv(
                        a_hand_files[index], sep=' ',
                        na_values=' ',
                        dtype='float64'
                    )
                    #add a column for the class based on the file name structure
                    the_class = a_hand_files[index].split('_')[-1].split('.')[0]
                    df['_classe'] = the_class
                    return df.append(inside_rec(index-1), ignore_index=True).dropna(axis='columns')
           
            the_hand_data = inside_rec(len(a_hand_files)-1) 
            # standardizing hearder name
            the_hand_data.columns = self._normalize_header(the_hand_data.columns)
            return the_hand_data
    
    def parse(self):
        # create an instance of HandsData
        left_hand_df = self._load_all_data_by_hand('L')
        right_hand_df = self._load_all_data_by_hand('R')
        
        # all different available hand config
        target_names = np.unique(
        np.append(left_hand_df["classe"].unique(), 
                  right_hand_df["classe"].unique())
        )
        #
        target_index = np.arange(len(target_names))
        classes_dict = {target_names[i]: i for i in target_index}
        
        hands_data = HandsData(left_hand_df, right_hand_df, classes_dict)
        
        return hands_data


# %% main function, just for test purposes

def main():
    #parser = Parser(HAND_POSES_DIRECTORY)    
    parser = Parser()    
    hands_data = parser.parse()
    
    # print(hands_data.left_hand_data.iloc[[0, -1]])
    # print(hands_data.left_hand_data.iloc[[130, 150, 200]])
    
    # print(hands_data.right_hand_data.iloc[[0, -1]])
    # print(hands_data.right_hand_data.iloc[[130, 150, 200]])
    
    
        
if __name__ == "__main__":
    main()
    
