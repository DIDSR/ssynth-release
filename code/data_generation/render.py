import sys

sys.path.append('/projects01/VICTRE/elena.sizikova/code/mitsuba_setup/mitsuba3/build/python')
import mitsuba as mi

mi.set_variant('scalar_spectral')

import argparse
import os
import pandas as pd
import time
import config
import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--saveDir', type=str, help='directory to save outputs', required='True')
    parser.add_argument('--variation', type=str, help='type of variation to generate', default='None')
    parser.add_argument('--regLesions', help='whether to use regular lesions', action='store_true')
    parser.add_argument('--res', help='image resolution', type=int, default=128)
    parser.add_argument('--noHair', help='whether to remove hair', action='store_true')
    parser.add_argument('--row_id', help='row id to run', type=int, default=0)
    args = parser.parse_args()

    print("saveDir " + str(args.saveDir))
    print("args.variation " + str(args.variation))
    print("args.regLesions " + str(args.regLesions))
    print("args.res " + str(args.res))
    print("args.noHair " + str(args.noHair))

    if args.regLesions:
        lesion_directory = config.sDir_lesion_ver0
    else:
        lesion_directory = config.sDir_lesion_ver1

    l_params_to_render = []
    assert (args.variation != 'None')
    if args.variation == 'mel':
        csv_save_name = (config.param_dir + 'params_lists/' + 'mel_variation_light0_release.csv')
    elif args.variation == 'blood':
        csv_save_name = (config.param_dir + 'params_lists/' + 'blood_variation_light0_release.csv')
    elif args.variation == 'hair':
        csv_save_name = (config.param_dir + 'params_lists/' + 'lesion_regularity_light0_release.csv')
    elif args.variation == 'reg':
        csv_save_name = (config.param_dir + 'params_lists/' + 'lesion_regularity_light0_release.csv')
    elif args.variation == '10k':
        csv_save_name = (config.param_dir + 'params_lists/' + '/10k_dataset_release.csv')
    else:
        csv_save_name = ''
    print('csv_save_name ' + str(csv_save_name))
    data_csv = pd.read_csv(csv_save_name)

    for row_id in range(args.row_id, args.row_id + 1):
        print('running ', str(row_id))
        # get all data to render
        params = data_csv.iloc[row_id]
        print('params ' + str(params))
        id_model = int(params['id_model'])
        if args.noHair:
            id_hairModel = -1
        else:
            id_hairModel = int(params['id_hairModel'])

        id_lesion = int(params['id_lesion'])
        id_timePoint = int(params['id_timePoint'])
        id_lesionMat = int(params['id_lesionMat'])
        id_fracBlood = float(params['id_fracBlood'])
        id_mel = float(params['id_mel'])
        id_light = int(params['id_light'])
        id_hairAlbedo = int(params['id_hairAlbedo'])
        if 'mi_variant' in params.keys():
            id_miVariant = str(params['mi_variant'])
        else:
            id_miVariant = mi.variant()
        id_lesionScale = float(params['lesion_scale'])
        id_origin_y = float(params['origin_y'])
        offset = float(params['offset'])

        # get material names
        sel_lesionMat, sel_lightName, sel_hair_albedo = util.get_materials_names(id_lesionMat, id_light, id_hairAlbedo)

        # get render camera
        cam_top = util.get_sensor(id_origin_y=id_origin_y)

        # get folder to save output
        save_folder = util.get_save_folder(args.saveDir, id_model, id_hairModel,
                                           id_mel, id_fracBlood, id_lesion, id_timePoint,
                                           sel_lesionMat, sel_hair_albedo, sel_lightName,
                                           id_miVariant, id_lesionScale,
                                           id_origin_y=id_origin_y)
        print('save_folder ' + str(save_folder))

        if os.path.isfile(save_folder + "/image.png") and os.path.isfile(save_folder + "/mask.png"):
            print('files exist; finishing')
        else:
            print('\nrendering mask..')
            start_time = time.time()
            scene_ref = util.render_image(id_model, id_hairModel, id_lesion, sel_lesionMat, id_fracBlood, id_mel,
                                          id_timePoint, sel_lightName, sel_hair_albedo,
                                          IMAGE=False,
                                          lesion_directory=lesion_directory,
                                          lesionScale=id_lesionScale,
                                          yOffset_lesion=offset)
            ref_image = mi.render(scene_ref, sensor=cam_top, spp=32)
            mi.util.write_bitmap(save_folder + "/mask.png", ref_image)

            total_time = time.time() - start_time
            print('render mask time ' + str(total_time))

            print('\nrendering image..')
            start_time = time.time()
            scene_ref = util.render_image(id_model, id_hairModel, id_lesion, sel_lesionMat, id_fracBlood, id_mel,
                                          id_timePoint, sel_lightName, sel_hair_albedo,
                                          IMAGE=True,
                                          lesion_directory=lesion_directory,
                                          lesionScale=id_lesionScale,
                                          yOffset_lesion=offset)
            ref_image = mi.render(scene_ref, sensor=cam_top, spp=args.res)
            mi.util.write_bitmap(save_folder + "/image.png", ref_image)
            total_time = time.time() - start_time
            print('render image time ' + str(total_time))
