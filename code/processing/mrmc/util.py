import subprocess
import numpy as np
import json
import string
import random
import pandas as pd
import numpy as np
import os

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
    
def get_imrmc_mean_std_metric(l_frames_by_result):
    # generate random string to ensure correctness
    random_id = id_generator()
    
    if len(l_frames_by_result)==3:
        result = pd.merge(l_frames_by_result[0], l_frames_by_result[1], on="individual_image_name")
        result3 = pd.merge(result, l_frames_by_result[2], on="individual_image_name")
        
        # check mask names are the same
        assert(result3['individual_mask_name_x'].equals(result3['individual_mask_name_y']))
        assert(result3['individual_mask_name_x'].equals(result3['individual_mask_name']))
        
        # drop duplicate columns
        result3 = result3.drop(columns=['individual_mask_name_x', 'individual_mask_name_y'])
        
        # rename and reorder columns
        result3 = result3.rename(columns={"model_name_x": "model_name_reader0", 
                           "model_name_y": "model_name_reader1",
                           "model_name": "model_name_reader2",
                           "metric_x": "reader0",
                           "metric_y": "reader1",
                           "metric": "reader2"})
        
        result3 = result3[['individual_image_name', 
                           'individual_mask_name', 
                           'model_name_reader0', 
                           'model_name_reader1', 
                           'model_name_reader2', 
                           'reader0',
                           'reader1',
                           'reader2']]
        
        # check concatenation was correct
        img_name = result3['individual_image_name'][0]
        
        r0_new = result3[result3['individual_image_name']==img_name]['reader0'].item()
        r0_old = l_frames_by_result[0][l_frames_by_result[0]['individual_image_name']==img_name]['metric'].item()
        assert(r0_new==r0_old)
        r1_new = result3[result3['individual_image_name']==img_name]['reader1'].item()
        r1_old = l_frames_by_result[1][l_frames_by_result[1]['individual_image_name']==img_name]['metric'].item()
        assert(r1_new==r1_old)
        
        r2_new = result3[result3['individual_image_name']==img_name]['reader2'].item()
        r2_old = l_frames_by_result[2][l_frames_by_result[2]['individual_image_name']==img_name]['metric'].item()
        assert(r2_new==r2_old)
        result_pd = result3
        outName = 'result_' + random_id + '.csv' 
        result_pd[['reader0',
                   'reader1',
                   'reader2',
                   'reader3',
                   'reader4']].to_csv(outName, index=False) 

    elif len(l_frames_by_result)==5:
        
        result5 = pd.merge(l_frames_by_result[0], l_frames_by_result[1], on="individual_image_name", suffixes=('_1', '_2'))
        result5 = result5.merge(l_frames_by_result[2], on="individual_image_name")
        result5 = result5.rename(columns={"model_name": "model_name_3", 
                           "individual_mask_name": "individual_mask_name_3",
                           "metric": "metric_3"})
        result5 = pd.merge(result5, l_frames_by_result[3], on="individual_image_name")
        result5 = result5.rename(columns={"model_name": "model_name_4", 
                           "individual_mask_name": "individual_mask_name_4",
                           "metric": "metric_4"})
        result5 = pd.merge(result5, l_frames_by_result[4], on="individual_image_name")
        result5 = result5.rename(columns={"model_name": "model_name_5", 
                           "individual_mask_name": "individual_mask_name_5",
                           "metric": "metric_5"})
        
        # # check mask names are the same
        assert(result5['individual_mask_name_1'].equals(result5['individual_mask_name_2']))
        assert(result5['individual_mask_name_1'].equals(result5['individual_mask_name_3']))
        assert(result5['individual_mask_name_1'].equals(result5['individual_mask_name_4']))
        assert(result5['individual_mask_name_1'].equals(result5['individual_mask_name_5']))
        
        # drop duplicate columns
        result5 = result5.drop(columns=['individual_mask_name_2', 
                                        'individual_mask_name_3',
                                        'individual_mask_name_4',
                                        'individual_mask_name_5'])
        
        # rename and reorder columns
        result5 = result5.rename(columns={"model_name_1": "model_name_reader0", 
                                          "model_name_2": "model_name_reader1",
                                          "model_name_3": "model_name_reader2",
                                          "model_name_4": "model_name_reader3",
                                          "model_name_5": "model_name_reader4",
                                          "individual_mask_name_1": "individual_mask_name",
                                          "metric_1": "reader0",
                                          "metric_2": "reader1",
                                          "metric_3": "reader2",
                                          "metric_4": "reader3",
                                          "metric_5": "reader4"})
        
        result5 = result5[['individual_image_name', 
                           'individual_mask_name', 
                           'model_name_reader0', 
                           'model_name_reader1', 
                           'model_name_reader2', 
                           'model_name_reader3', 
                           'model_name_reader3', 
                           'reader0',
                           'reader1',
                           'reader2',
                           'reader3',
                           'reader4']]
        
        # # check concatenation was correct
        img_name = result5['individual_image_name'][0]
        
        r0_new = result5[result5['individual_image_name']==img_name]['reader0'].item()
        r0_old = l_frames_by_result[0][l_frames_by_result[0]['individual_image_name']==img_name]['metric'].item()
        assert(r0_new==r0_old)
        r1_new = result5[result5['individual_image_name']==img_name]['reader1'].item()
        r1_old = l_frames_by_result[1][l_frames_by_result[1]['individual_image_name']==img_name]['metric'].item()
        assert(r1_new==r1_old)
        
        r2_new = result5[result5['individual_image_name']==img_name]['reader2'].item()
        r2_old = l_frames_by_result[2][l_frames_by_result[2]['individual_image_name']==img_name]['metric'].item()
        assert(r2_new==r2_old)
        result_pd = result5

        outName = 'result_' + random_id + '.csv' 
        result_pd[['reader0',
                   'reader1',
                   'reader2',
                   'reader3',
                   'reader4']].to_csv(outName, index=False) 

    else:
        raise NotImplementedError     
        
    saveName = 'result_imrmc_' + random_id + '.json' # output
    out = subprocess.check_output("Rscript run_doIMRMCestOR.R " + outName + " " + saveName, shell=True)
    with open(saveName) as f:
        l = json.load(f)
    
    mean = l['estimation'][0]['readerAveragedPerformance']
    var = l['estimation'][0]['variance']
    std = np.sqrt(var)
    
    #delete intermediate files
    os.remove(outName)
    os.remove(saveName)

    return mean, std