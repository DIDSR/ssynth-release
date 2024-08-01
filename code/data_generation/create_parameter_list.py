import os
import random

import pandas as pd

import config
import util


def get_blood_variation_df(num):
    l_model = util.get_l_model()
    l_hairModel = util.get_l_hairModel()
    l_lesion = util.get_l_lesion()
    l_times = util.get_l_times()
    l_lesionMat = util.get_l_lesionMat()
    l_melanosomes = util.get_l_melanosomes()
    l_light = util.get_l_light()
    l_hairAlbedoIndex = util.get_l_hairAlbedoIndex()

    l_param_list = []
    while len(l_param_list) != num:
        id_model = random.choice(l_model)
        id_hairModel = random.choice(l_hairModel)
        id_lesion = random.choice(l_lesion)
        id_timePoint = random.choice(l_times)
        id_lesionMat = random.choice(l_lesionMat)
        id_mel = random.choice(l_melanosomes)
        id_light = 0
        id_hairAlbedo = random.choice(l_hairAlbedoIndex)

        param_combo = [id_model, id_hairModel, id_lesion, id_timePoint, id_lesionMat, [], id_mel, id_light,
                       id_hairAlbedo]

        if param_combo in l_param_list:
            continue

        else:
            l_param_list.append(param_combo)

    print(len(l_param_list))
    print('param list generated')

    # create and save df with lesion params
    df = pd.DataFrame(l_param_list,
                      columns=['id_model', 'id_hairModel', 'id_lesion', 'id_timePoint', 'id_lesionMat', 'id_fracBlood',
                               'id_mel', 'id_light', 'id_hairAlbedo'])
    print(df.head())
    print(len(df))
    return df


if __name__ == "__main__":

    df = get_blood_variation_df(num=1815)

    # generate text files
    l_fractionBlood = util.get_l_fractionBlood()

    df_blood_all = pd.DataFrame(columns=['id_model', 'id_hairModel', 'id_lesion', 'id_timePoint', 'id_lesionMat',
                                         'id_fracBlood', 'id_mel', 'id_light', 'id_hairAlbedo'])
    for blood in l_fractionBlood:
        df_blood = df.copy()
        df_blood['id_fracBlood'] = blood

        save_path = config.output_dir + 'param_lists/'
        save_name = (save_path + 'blood_variation_blood' + str(blood) + '_light0.csv')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_blood.to_csv(save_name, index=False)
        df_blood_all = pd.concat([df_blood_all, df_blood], ignore_index=True)
    save_name_A = (save_path + 'blood_variation_light0_all_examples.csv')
    print(len(df_blood_all))
    df_blood_all.to_csv(save_name_A, index=False)
