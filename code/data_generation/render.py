import sys

sys.path.append('/projects01/VICTRE/elena.sizikova/code/mitsuba_setup/mitsuba3/build/python')
import mitsuba as mi

mi.set_variant('scalar_spectral')

import argparse
import ast
import os
import pandas as pd
import time
import config
import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- I/O and scheduling ---
    parser.add_argument('--saveDir',       type=str, help='directory to save outputs', required=True)
    parser.add_argument('--sch',           type=str, help='type of scheduler (sge | slurm); if omitted, --row_id is used', default=None)
    parser.add_argument('--row_id',        type=int, help='row id to run (used when --sch is not set)', default=0)
    parser.add_argument('--numRun',        type=int, help='number of consecutive rows to run', default=1)

    # --- Dataset selection ---
    # parser.add_argument('--oasis',         help='use OASIS parameters / CSV', action='store_true')
    parser.add_argument('--variation',     type=str, help='variation type when not using --oasis', default='None')
    parser.add_argument('--regLesions',    help='use regular lesions (non-oasis only)', action='store_true')

    # --- Render settings ---
    parser.add_argument('--res',           type=int, help='image resolution (spp for final render)', default=128)

    # --- Artifact flags ---
    parser.add_argument('--noHair',        help='remove hair', action='store_true')
    parser.add_argument('--hairDense',     help='use high-density hair layers', action='store_true')
    parser.add_argument('--bloodVessel',   help='use blood-vessel skin layers', action='store_true')
    parser.add_argument('--noVasculature', help='disable vasculature', action='store_true')
    parser.add_argument('--frame',         help='include black frame', action='store_true')
    parser.add_argument('--CalChart',      help='include calibration chart', action='store_true')
    parser.add_argument('--ruler',         help='include ruler', action='store_true')
    parser.add_argument('--render_all_arts', help='read per-row artifact flags from CSV', action='store_true')

    args = parser.parse_args()

    # --- Echo arguments ---
    print("saveDir:          " + str(args.saveDir))
    print("sch:              " + str(args.sch))
    print("row_id:           " + str(args.row_id))
    print("numRun:           " + str(args.numRun))
    # print("oasis:            " + str(args.oasis))
    print("variation:        " + str(args.variation))
    print("regLesions:       " + str(args.regLesions))
    print("res:              " + str(args.res))
    print("noHair:           " + str(args.noHair))
    print("frame:            " + str(args.frame))
    print("CalChart:         " + str(args.CalChart))
    print("ruler:            " + str(args.ruler))
    print("hairDense:        " + str(args.hairDense))
    print("bloodVessel:      " + str(args.bloodVessel))
    print("noVasculature:    " + str(args.noVasculature))
    print("render_all_arts:  " + str(args.render_all_arts))

    # ------------------------------------------------------------------ #
    # Lesion directory
    # ------------------------------------------------------------------ #
    # if args.oasis:
    lesion_directory = config.sDir_lesion_ver3
    # else:
    #     if args.regLesions:
    #         lesion_directory = config.sDir_lesion_ver0
    #     else:
    #         lesion_directory = config.sDir_lesion_ver1

    # ------------------------------------------------------------------ #
    # CSV selection
    # ------------------------------------------------------------------ #
    # if args.oasis:
    if args.bloodVessel:
        csv_save_name = config.param_dir + 'oasis_all_examples_bloodVessel_corrected.csv'
    elif args.render_all_arts:
        csv_save_name = config.param_dir + 'oasis_all_examples_allArtifacts.csv'
    else:
        csv_save_name = config.param_dir + 'oasis_all_examples.csv'
    # else:
    #     assert args.variation != 'None', "--variation must be set when --oasis is not used"
    #     variation_map = {
    #         'mel':   'mel_variation_light0_release.csv',
    #         'blood': 'blood_variation_light0_release.csv',
    #         'hair':  'lesion_regularity_light0_release.csv',
    #         'reg':   'lesion_regularity_light0_release.csv',
    #         '10k':   '10k_dataset_release.csv',
    #     }
    #     assert args.variation in variation_map, f"Unknown variation: {args.variation}"
    #     csv_save_name = config.param_dir + 'params_lists/' + variation_map[args.variation]

    print('csv_save_name: ' + str(csv_save_name))
    data_csv = pd.read_csv(csv_save_name)

    # Parse list-valued columns only when they are present (OASIS CSVs)
    # if args.oasis:
    data_csv['ruler_params']    = data_csv['ruler_params'].apply(ast.literal_eval)
    data_csv['calChart_params'] = data_csv['calChart_params'].apply(ast.literal_eval)

    # ------------------------------------------------------------------ #
    # Determine starting row
    # ------------------------------------------------------------------ #
    if args.sch == 'slurm':
        start_row = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1  # tasks start from 1
        print('ROW (slurm): ' + str(start_row))
    elif args.sch == 'sge':
        start_row = int(os.environ['SGE_TASK_ID']) - 1          # tasks start from 1
        print('ROW (sge): ' + str(start_row))
    else:
        start_row = args.row_id
        print('ROW (row_id): ' + str(start_row))

    # ------------------------------------------------------------------ #
    # Main render loop
    # ------------------------------------------------------------------ #
    for row_id in range(start_row, start_row + args.numRun):
        print('\nrunning row: ' + str(row_id))
        params = data_csv.iloc[row_id]
        print('params: ' + str(params))

        # --- Per-row artifact overrides (OASIS only) ---
        # if args.oasis and args.render_all_arts:
        # no_calChart = bool(params['no_calChart'])
        # no_ruler    = bool(params['no_ruler'])
        # no_frame    = bool(params['no_frame'])
        # no_hair     = bool(params['no_hair'])
        # else:
        #     no_calChart = False
        #     no_ruler    = False
        #     no_frame    = False
        #     no_hair     = False
        # print(f'artifact overrides — no_calChart:{no_calChart}  no_ruler:{no_ruler}  no_frame:{no_frame}  no_hair:{no_hair}')

        # --- Core parameters ---
        id_model      = int(params['id_model'])
        id_lesion     = int(params['id_lesion'])
        id_timePoint  = int(params['id_timePoint'])
        id_lesionMat  = int(params['id_lesionMat'])
        id_fracBlood  = float(params['id_fracBlood'])
        id_mel        = float(params['id_mel'])
        id_light      = int(params['id_light'])
        id_hairAlbedo = int(params['id_hairAlbedo'])
        id_origin_y   = float(params['origin_y'])

        # Mitsuba variant — fall back to currently active variant if not in CSV
        if 'mi_variant' in params.keys():
            id_miVariant = str(params['mi_variant'])
        else:
            id_miVariant = mi.variant()
        print('setting mitsuba variant to ' + id_miVariant)
        mi.set_variant(id_miVariant)

        # Lesion scale — OASIS uses a fixed 1.5; non-OASIS reads from CSV
        # if args.oasis:
        id_lesionScale = 1.5
        # else:
        #     id_lesionScale = float(params['lesion_scale'])

        # Offset — only present in non-OASIS CSVs
        offset = float(params['offset']) if 'offset' in params.keys() else 0.0

        # Ruler / calChart params — only present in OASIS CSVs
        ruler_params    = params['ruler_params']    #if args.oasis else None
        calChart_params = params['calChart_params'] #if args.oasis else None

        # --- Hair model ---
        if args.noHair:# or no_hair:
            id_hairModel = -1
        else:
            id_hairModel = int(params['id_hairModel'])

        # --- Frame flag ---
        if args.frame:# and not no_frame:
            id_frame = 1 if id_timePoint < 30 else -1
        else:
            id_frame = -1

        # --- Calibration chart flag ---
        if args.CalChart:# and not no_calChart:
            id_calChart = 1 if id_timePoint < 30 else -1
        else:
            id_calChart = -1

        # --- Ruler flag ---
        if args.ruler:#  and not no_ruler:
            id_ruler = 1 if id_timePoint < 30 else -1
        else:
            id_ruler = -1

        # --- Skin layers directory ---
        if args.hairDense:# and not no_hair:
            skin_layers_directory = config.sDir_layers_hair
        elif args.bloodVessel:
            skin_layers_directory = config.sDir_layers_bloodVessel
        else:
            skin_layers_directory = config.sDir_layers_orig
        print('loading skin layers from: ' + str(skin_layers_directory))

        # --- Derived names ---
        sel_lesionMat, sel_lightName, sel_hair_albedo = util.get_materials_names(
            id_lesionMat, id_light, id_hairAlbedo
        )

        # --- Camera ---
        cam_top = util.get_sensor(id_origin_y=id_origin_y)

        # --- Save folder ---
        save_folder = util.get_save_folder(
            args.saveDir, id_model, id_hairModel,
            id_mel, id_fracBlood, id_lesion, id_timePoint,
            sel_lesionMat, sel_hair_albedo, sel_lightName,
            id_miVariant, id_lesionScale,
            id_frame, id_calChart, id_ruler, id_origin_y
        )
        print('save_folder: ' + str(save_folder))

        if os.path.isfile(save_folder + "/image.png") and os.path.isfile(save_folder + "/mask.png"):
            print('files exist; skipping')
            continue

        # --- Render mask ---
        print('\nrendering mask..')
        start_time = time.time()
        scene_ref = util.render_image(
            id_model, id_hairModel, id_lesion,
            sel_lesionMat, sel_lightName, sel_hair_albedo,
            id_fracBlood, id_mel,
            id_timePoint, id_origin_y,
            id_frame, id_calChart, calChart_params, id_ruler, ruler_params,
            IMAGE=False,
            lesion_directory=lesion_directory,
            skin_layers_directory=skin_layers_directory,
            pre_processed_lesion=False
            # lesionScale=id_lesionScale,
            # yOffset_lesion=offset
        )

        ref_image = mi.render(scene_ref, sensor=cam_top, spp=32)
        mi.util.write_bitmap(save_folder + "/mask.png", ref_image)
        print('render mask time: ' + str(time.time() - start_time))

        # --- Render image ---
        print('\nrendering image..')
        start_time = time.time()
        scene_ref = util.render_image(
            id_model, id_hairModel, id_lesion,
            sel_lesionMat, sel_lightName, sel_hair_albedo,
            id_fracBlood, id_mel,
            id_timePoint, id_origin_y,
            id_frame, id_calChart, calChart_params, id_ruler, ruler_params,
            IMAGE=True,
            lesion_directory=lesion_directory,
            skin_layers_directory=skin_layers_directory,
            # lesionScale=id_lesionScale,
            # yOffset_lesion=offset,
            pre_processed_lesion=False,
            no_vasculature=args.noVasculature
        )

        ref_image = mi.render(scene_ref, sensor=cam_top, spp=args.res)
        mi.util.write_bitmap(save_folder + "/image.png", ref_image)
        print('render image time: ' + str(time.time() - start_time))
