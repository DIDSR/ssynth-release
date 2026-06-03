import os
import random
import mitsuba as mi
import config
import trimesh
import math
import json
         
def get_frame(id_origin_y):
    radius = id_origin_y - 2
    tube_radius = 2.0 
    major_sections = 256
    minor_sections = 256
    mesh = trimesh.creation.torus(radius, tube_radius, major_sections, minor_sections)
    angle = math.pi / 2
    direction = [1, 0, 0]
    center = [0, 0, 0]
    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
    mesh.apply_transform(rot_matrix)
    print('Frame created ...')
    os.makedirs(config.sDir_materials + 'objects', exist_ok=True)
    mesh.export(config.sDir_materials + 'objects/frame.obj')
    return
    
def render_image(id_model, id_hairModel, id_lesion,
                 sel_lesionMat, sel_lightName, sel_hair_albedo, 
                 id_fracBlood, id_mel,
                 id_timePoint, id_origin_y,
                 id_frame=-1, id_calChart=-1, calChart_params=[], id_ruler=-1, ruler_params=[],
                 saveDir='', IMAGE=True,
                 lesion_directory='', skin_layers_directory=config.sDir_layers_orig,
                pre_processed_lesion=True,  verbose=True, no_vasculature=False):
    
    uniformScale = 1  # uniformly scale the models
    yOffset = -1.5 # this is to counter the y offset of the models in houdini which is not centered at 0.
    roomSize = 20  # current skin size is 20x20x5mm, so the room should be large enough to fit the skin
    xtScale = 0.1  # extinction scale. 1 unit in mitsuba/houdini = 1mm, and optical coefficients are in inverse cm
    lesionScale=1.5
    
    if verbose:
        print("Model id = " + str(id_model))
        print("Lesion id = lesion" + str(id_lesion) + "_T" + f'{id_timePoint:03d}')
        print("Lesion material = " + str(sel_lesionMat))
        print("Blood fraction of Dermis material = " + str(id_fracBlood))
        print("Melanosome fraction of Epidermis material = " + str(id_mel))
        print("mitsuba variant " + str(mi.variant()))
        print("lesionScale " + str(lesionScale))
        print("sel_lightName " + str(sel_lightName))
        print("Selected artifacts --> " + "Frame:" + str(id_frame) + ", calChart:" + str(id_calChart) + ", ruler:" + str(id_ruler) )
        print()

    # testing material for hair
    muaHair_black = 28.5
    musHair_black = 37.5
    extHair_black = muaHair_black + musHair_black

    # refractive index for epidermis between 1.42-1.44
    iorEpi = 1.43

    # refractive index for blood = 1.36 for 680-930nm
    iorBlood = 1.36

    # refractive index for hypodermis
    iorHypo = 1.44

    # refractive index for dermis is wavelength-dependent, but cannot input spectrum for ior in bsdf
    # therefore, will normalize to lambda = 500nm
    A = 1.3696
    B = 3916.8
    C = 2558.8
    iorDerm = A + (B / (500 ** 2)) + (C / (500 ** 4))

    #frame = get_frame(id_origin_y) #generate black frame around the image

    scene = {'type': 'scene',
             'integrator': {'type': 'volpathmis',
                            'max_depth': 1000}}
            
    if pre_processed_lesion == False:
        print('load lesion from ' + lesion_directory)
    
        myMesh = trimesh.load(lesion_directory + '/lesion' + str(
                                       id_lesion) + '_T' + f'{id_timePoint:03d}' + '.obj')
        myMesh.apply_scale(uniformScale * lesionScale)
        myMesh.apply_translation([-myMesh.bounds[0, 0] - myMesh.extents[0] / 2.0,
                                  -myMesh.bounds[0, 1] - myMesh.extents[1] / 2.0,
                                  -myMesh.bounds[0, 2]])
        myMesh.fix_normals()
        
        filename = str(id_lesion) + '_T' + f'{id_timePoint:03d}'

        print('save lesion mesh file to' + lesion_directory)
        os.makedirs(config.sDir_materials +'/objects/preprocessed_lesions/', exist_ok=True)
        with open(config.sDir_materials +'/objects/preprocessed_lesions' + '/mesh_' + filename + '.obj', 'w') as f:
            f.write(trimesh.exchange.export.export_obj(myMesh, include_normals=True)) 
        f.close()
        
        myMesh_meta = {
            "extents": myMesh.extents.tolist(),
            "bounds": myMesh.bounds.tolist()
        }
        with open(config.sDir_materials +'/objects/preprocessed_lesions' + '/mesh_' + filename + '_meta.json', 'w') as f:
            json.dump(myMesh_meta, f)
        f.close()

    # if pre_processed_lesion == True: #loading the pre-processed lesion mesh object 
    print('load lesion from ' +  config.sDir_materials +'/objects/preprocessed_lesions')
    filename = str(id_lesion) + '_T' + f'{id_timePoint:03d}'
    
    with open(config.sDir_materials +'/objects/preprocessed_lesions' + '/mesh_' + filename + '_meta.json', 'r') as f:
        myMesh = json.load(f)
    f.close()
    
    tr=[0,2.0,0]
    lesion_translation = [tr[0], tr[1]-0.5*myMesh["extents"][1], 0.5*(myMesh["bounds"][0][2]-myMesh["bounds"][1][2])]
    
    scene['lesion'] = {'type': 'obj',
                       'filename': config.sDir_materials +'/objects/preprocessed_lesions' + '/mesh_' + filename + '.obj',
                       'face_normals': True,
                       'to_world': mi.ScalarTransform4f.scale(uniformScale * lesionScale).translate(lesion_translation).rotate(
                           [0, 0, 0], 0),
                       } 

    if IMAGE:
        scene['lesion']['bsdf'] = {'type': 'roughdielectric',
                                   'alpha': 0.01,
                                   'int_ior': 1.32988 - (-3.97577e7) * 0.95902 ** 500,
                                   'ext_ior': 1.000277}
        scene['lesion']['interior'] = {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir_materials + 'opticalMaterials/' + str(sel_lesionMat) + '_alb.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir_materials + 'opticalMaterials/' + str(sel_lesionMat) + '_ext.spd'
            },
            'scale': xtScale
        }
    else:
        scene['lesion']['bsdf'] = {'type': 'twosided',
                                   'material': {
                                       'type': 'diffuse',
                                       'reflectance': {
                                           'type': 'rgb',
                                           'value': [1.0, 1.0, 1.0]
                                       }
                                   }
                                   }

    if IMAGE:
        if id_hairModel == -1:
            if verbose:
                print('not using hair') 
        else:
            scene['hair'] = {
                'type': 'obj',
                'filename': skin_layers_directory + 'hair_' + f'{id_hairModel:03d}' + '.obj',
                'to_world': mi.ScalarTransform4f.scale(uniformScale).translate([0, yOffset, 0]).rotate([0, 0, 0], 0),
                'bsdf': {'type': 'roughdielectric',
                         'alpha': 0.01,
                         'int_ior': 1.55,
                         'ext_ior': 1.000277},
                'interior': {
                    'type': 'homogeneous',
                    'sigma_t': extHair_black,
                    'albedo': {
                        'type': 'rgb',
                        'value': sel_hair_albedo
                    },
                    'scale': 3
                },
            }

        scene['epidermis'] = {
            'type': 'obj',
            'filename': skin_layers_directory + 'epidermis_' + f'{id_model:03d}' + '.obj',
            'to_world': mi.ScalarTransform4f.scale(uniformScale).translate([0, yOffset, 0]).rotate([0, 0, 0], 0),
            'bsdf': {'type': 'roughdielectric',
                     'alpha': 0.01,
                     'int_ior': iorEpi,
                     'ext_ior': 1.000277},
            'interior': {
                'type': 'homogeneous',
                'albedo': {
                    'type': 'spectrum',
                    'filename': config.sDir_materials + 'opticalMaterials/epidermis_alb_mel' + str(id_mel) + '.spd'
                },
                'sigma_t': {
                    'type': 'spectrum',
                    'filename': config.sDir_materials + 'opticalMaterials/epidermis_ext_mel' + str(id_mel) + '.spd'
                },
                'scale': xtScale
            }
        }
        if no_vasculature:
            print('not adding vasculature')
        else:
            scene['vascular'] = {
                'type': 'obj',
                'filename': skin_layers_directory + 'vascular_' + f'{id_model:03d}' + '.obj',
                'to_world': mi.ScalarTransform4f.scale(uniformScale).translate([0, yOffset, 0]).rotate([0, 0, 0], 0),
                'bsdf': {'type': 'roughdielectric',
                         'alpha': 0.01,
                         'int_ior': iorBlood,
                         'ext_ior': 1.000277},
                'interior': {
                    'type': 'homogeneous',
                    'albedo': {
                        'type': 'spectrum',
                        'filename': config.sDir_materials + 'opticalMaterials/blood_HbO2_alb' + '.spd'
                    },
                    'sigma_t': {
                        'type': 'spectrum',
                        'filename': config.sDir_materials + 'opticalMaterials/blood_HbO2_ext' + '.spd'
                    },
                    'scale': xtScale
                }
            }
        scene['dermis'] = {
            'type': 'obj',
            'filename': skin_layers_directory + 'dermis_' + f'{id_model:03d}' + '.obj',
            'to_world': mi.ScalarTransform4f.scale(uniformScale).translate([0, yOffset, 0]).rotate([0, 0, 0], 0),
            'bsdf': {'type': 'roughdielectric',
                     'alpha': 0.01,
                     'int_ior': iorDerm,
                     'ext_ior': 1.000277},
            'interior': {
                'type': 'homogeneous',
                'albedo': {
                    'type': 'spectrum',
                    'filename': config.sDir_materials + 'opticalMaterials/dermis_alb_fB' + str(id_fracBlood) + '.spd'
                },
                'sigma_t': {
                    'type': 'spectrum',
                    'filename': config.sDir_materials + 'opticalMaterials/dermis_ext_fB' + str(id_fracBlood) + '.spd'
                },
                'scale': xtScale
            }
        }

        scene['subcutfat'] = {
            'type': 'obj',
            'filename': skin_layers_directory + 'hypodermis_' + f'{id_model:03d}' + '.obj',
            'to_world': mi.ScalarTransform4f.scale(uniformScale).translate([0, yOffset, 0]).rotate([0, 0, 0], 0),
            'bsdf': {'type': 'roughdielectric',
                     'alpha': 0.01,
                     'int_ior': iorHypo,
                     'ext_ior': 1.000277},
            'interior': {
                'type': 'homogeneous',
                'albedo': {
                    'type': 'spectrum',
                    'filename': config.sDir_materials + 'opticalMaterials/hypo_alb' + '.spd'
                },
                'sigma_t': {
                    'type': 'spectrum',
                    'filename': config.sDir_materials + 'opticalMaterials/hypo_ext' + '.spd'
                },
                'scale': xtScale
            }
        }
        if sel_lightName == 'diffuse':
            scene['env_light'] = {
                'type': 'constant',
                      'radiance': {
                          'type': 'rgb',
                          'value': 2.0
                      }
                }
        else:
            scene['env_light'] = {
                'type': 'envmap',
                'filename': config.sDir_hdri + sel_lightName + '.exr',
                'scale': 3
            }

        scene['wall_floor'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([roomSize, 1, roomSize]).translate([0, -roomSize, 0]).rotate(
                [1, 0, 0], -90),
            'bsdf': {
                'type': 'twosided',
                'material': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': 0.5
                    }
                }
            }
        }

        if id_frame == -1:
            if verbose:
                print('not using frame')
        else:
            scene['frame'] = {
                        'type': 'obj',
                        'filename': config.sDir_materials + 'objects/frame.obj',
                        'to_world': mi.ScalarTransform4f.scale(uniformScale).translate([0, (yOffset+4)/uniformScale, 0]).rotate([0, 0, 0], 0),
                        'material': {
                            'type': 'diffuse',
                            'reflectance': {
                                'type': 'rgb',
                                'value': [0.0, 0.0, 0.0]
                            }
                        }
                        }
            
        if id_calChart == -1:
            if verbose:
                print('not using calibration chart')
        else:
            #calChart_params = [int(x)  for x in calChart_params.strip("").strip("[").strip("]").split(",")]
            calChart_x = calChart_params [0]
            calChart_z = calChart_params [1]
            calChart_radius = calChart_params [2]
            calChart_color = calChart_params [3]
            if calChart_color == 0:
                value = [250/255, 95/255, 45/255] #orange
            else:
                value = [22/255, 115/255, 225/255] #blue
            scene['calibration_chart'] = {
                'type': 'sphere',
                'to_world': mi.ScalarTransform4f.scale(uniformScale).translate([0, 0, 0]).rotate([0, 0, 0], 0),
                'radius': calChart_radius/20*id_origin_y,
                'center': [(calChart_x-(id_origin_y-11)/2)/uniformScale, (yOffset+5)/uniformScale, calChart_z/uniformScale],
                'material': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': value#[250/255, 95/255, 45/255] #[234, 135, 45] orange #[22, 145, 200] blue
                    }
                }
                }

        if id_ruler == -1:
            if verbose:
                print('not using ruler')
        else:
            #ruler_params = [int(x) for x in ruler_params.strip("").strip("[").strip("]").split(",")]
            ruler_x = ruler_params[0]
            ruler_z = ruler_params[1]
            idx = 0
            num_columns = 15
            for col in list(range(num_columns)):
                name = 'line' + str(idx)
                scene[name] = {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f.scale([uniformScale/20, uniformScale/2, uniformScale/4]).translate([(0.5*20)*col+(ruler_x*20),
                            yOffset+(4*2)/uniformScale, ruler_z*4]).rotate([1, 0, 0], 90), #default cube size is 2 mm
                'bsdf': {
                        'type': 'twosided',
                        'material': {
                        'type': 'diffuse',
                        'reflectance': {
                            'type': 'rgb',
                            'value': [0.0, 0.0, 0.0] #[0, 0, 0]
                        }
                    },
                },
                }
                idx += 1
                            
            scene['line_h'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([uniformScale*4.5, uniformScale, uniformScale/20]).translate([(ruler_x+3)/4.5, yOffset+4/uniformScale,
                      (ruler_z)*20]).rotate([1,0, 0], 89), #default cube size is 2 mm
            # 'material': {
            #     'type': 'diffuse',
            #     'reflectance': {
            #         'type': 'rgb',
            #         'value': [0.5, 0.0, 0.0]  #[0, 0, 0]
            #     }
            # },
                'bsdf': {
                    'type': 'twosided',
                    'material': {
                        'type': 'diffuse',
                        'reflectance': {
                            'type': 'rgb',
                            'value': [0.0, 0.0, 0.0]  # [0, 0, 0]
                        }
                    },
                },

                # flip_normals: True,
            }

    else:

        scene['epidermis'] = {
            'type': 'obj',
            'filename': skin_layers_directory + 'epidermis_' + f'{id_model:03d}' + '.obj',
            'to_world': mi.ScalarTransform4f.scale(uniformScale).translate([0, yOffset, 0]).rotate([0, 0, 0], 0),
            'bsdf': {'type': 'diffuse',
                     'reflectance': {
                         'type': 'rgb',
                         'value': [0.0, 0.0, 0.0]
                     }
                     }
        }
        scene['shape_light'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([roomSize, 1, roomSize]).translate([0, roomSize, 0]).rotate(
                [1, 0, 0], 90),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'd65',
                    'scale': 3
                }
            }
        }
        scene['wall_floor'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([roomSize, 1, roomSize]).translate([0, -roomSize, 0]).rotate(
                [1, 0, 0], -90),
            'bsdf': {
                'type': 'twosided',
                'material': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': [0.0, 0.0, 0.0]
                    }
                }
            }
        }

    scene_ref = mi.load_dict(scene)
    return scene_ref

def get_sensor(id_origin_y=15):
    # Creating a single sensor from top
    cam_top = mi.load_dict({
        'type': 'perspective',
        'srf': {
            'type': 'uniform',
            'value': 1.0
        },
        'to_world': mi.scalar_spectral.Transform4f.look_at(
            target=[0, 0, 0],
            origin=[0, id_origin_y, 0],
            up=[0, 0, 1]),
        'fov': 75,
        'film': {
            'type': 'hdrfilm',
            'width': 1024, 'height': 1024,
        }
    })
    return cam_top

def get_l_model():
    l_model = list(range(100))
    for x in [2, 14, 32, 54, 59, 61]:  # remove non-working models
        l_model.remove(x)
    return l_model

def get_l_hairModel():
    l_hairModel = list(range(100))
    for x in [2, 14, 32, 54, 59, 61]:  # remove non-working models
        l_hairModel.remove(x)
    return l_hairModel

def get_l_lesion():
    l_lesion = list(range(1, 21))
    return l_lesion

def get_l_origin_y():
    l_origin_y = [13, 14, 15]
    return l_origin_y
    
def get_l_times():
    l_times = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    return l_times

def get_l_lesionMat():
    l_lesionMat = list(range(1, 19))
    return l_lesionMat

def get_l_fractionBlood():
    l_fractionBlood = [0.002, 0.005, 0.02, 0.05]
    return l_fractionBlood

def get_l_melanosomes():
    l_melanosomes = [float(x) / 100 for x in range(1, 51)]  # range between 0.01 and 1.0
    return l_melanosomes

def get_l_light():
    l_light = list(range(19))  # [1.0,2.0,3.0, 4.0]
    return l_light

def get_l_hairAlbedoIndex():
    l_hairAlbedoIndex = [0, 1, 2]
    return l_hairAlbedoIndex

def get_calChart_params():
    calChart_params = {
        "z": list(range(-8,-4)) + list(range(5, 9)), #center of the calibration chart in the z direction
        "x": list(range(-7,-5)) + list(range(6, 8)), #center of the calibration chart in the x direction
        "radius": list(range(2,5)) #radius of the calibrartion chart
    }
    return calChart_params

def get_ruler_params():
    ruler_params = {
        "z": list(range(-7,-3)) + list(range(4, 7)), 
        "x": list(range(-8,2)),
    }
    return ruler_params

def get_param_combo(light_id=None):
    l_model = get_l_model()
    l_hairModel = get_l_hairModel()
    l_lesion = get_l_lesion()
    l_times = get_l_times()
    l_lesionMat = get_l_lesionMat()
    l_fractionBlood = get_l_fractionBlood()
    l_melanosomes = get_l_melanosomes()
    l_light = get_l_light()
    l_hairAlbedoIndex = get_l_hairAlbedoIndex()

    id_model = random.choice(l_model)
    id_hairModel = random.choice(l_hairModel)
    id_lesion = random.choice(l_lesion)
    id_timePoint = random.choice(l_times)
    id_lesionMat = random.choice(l_lesionMat)
    id_fracBlood = random.choice(l_fractionBlood)
    id_mel = random.choice(l_melanosomes)

    if not light_id:
      id_light = random.choice(l_light)
    else:
      id_light = light_id   
    id_hairAlbedo = random.choice(l_hairAlbedoIndex)

    print('id_model ' + str(id_model))
    print('id_hairModel ' + str(id_hairModel))
    print('id_lesion ' + str(id_lesion))
    print('id_timePoint ' + str(id_timePoint))
    print('id_lesionMat ' + str(id_lesionMat))
    print('id_fracBlood ' + str(id_fracBlood))
    print('id_mel ' + str(id_mel))
    print('id_light ' + str(id_light))
    print('id_hairAlbedo ' + str(id_hairAlbedo))

    sel_lesionMat, sel_lightName, sel_hair_albedo = get_materials_names(id_lesionMat, id_light, id_hairAlbedo)

    return id_model, id_hairModel, id_lesion, sel_lesionMat, id_fracBlood, id_mel, id_timePoint, sel_lightName, sel_hair_albedo

def get_l_lesionMat():
    lesionMat = ["melDermEpi",
                 "HbO2x0.1Epix0.025", "HbO2x0.1Epix0.05", "HbO2x0.1Epix0.1", "HbO2x0.1Epix0.15", "HbO2x0.1Epix0.25",
                 "HbO2x0.1Epix0.4",
                 "HbO2x0.5Epix0.025", "HbO2x0.5Epix0.05", "HbO2x0.5Epix0.1", "HbO2x0.5Epix0.15", "HbO2x0.5Epix0.25",
                 "HbO2x0.5Epix0.4",
                 "HbO2x1.0Epix0.025", "HbO2x1.0Epix0.05", "HbO2x1.0Epix0.1", "HbO2x1.0Epix0.15", "HbO2x1.0Epix0.25",
                 "HbO2x1.0Epix0.4"]
    return lesionMat

def get_light_names():
    exr_files = ['rural_asphalt_road_4k', 'comfy_cafe_4k', 'reading_room_4k', 'school_hall_4k', 'bathroom_4k',
                 'floral_tent_4k',
                 'st_fagans_interior_4k', 'vulture_hide_4k', 'lapa_4k', 'surgery_4k', 'veranda_4k',
                 'vintage_measuring_lab_4k',
                 'yaris_interior_garage_4k', 'hospital_room_4k', 'bush_restaurant_4k', 'lythwood_room_4k',
                 'kiara_interior_4k',
                 'reinforced_concrete_01_4k', 'graffiti_shelter_4k', 'diffuse']
    return exr_files

def get_l_hair_albedo():
    l_hair_albedo = [[0.57, 0.57, 0.57], [0.9, 0.9, 0.9], [0.84, 0.6328, 0.44]]
    return l_hair_albedo

def get_materials_names(id_lesionMat, id_light, id_hairAlbedo):
    lesionMat = get_l_lesionMat()
    exr_files = get_light_names()
    sel_lesionMat = lesionMat[id_lesionMat]
    sel_lightName = exr_files[id_light]
    l_hair_albedo = get_l_hair_albedo()
    sel_hair_albedo = l_hair_albedo[id_hairAlbedo]
    return sel_lesionMat,sel_lightName, sel_hair_albedo

def get_save_folder(saveDir, id_model, id_hairModel, id_mel, id_fracBlood, id_lesion, id_timePoint,
                    sel_lesionMat, sel_hair_albedo, sel_lightName, mi_variant, id_lesionScale, id_frame, id_calChart, id_ruler,
                    id_origin_y=None):
    folder = saveDir + "output/skin_" + f'{id_model:03d}'
    folder += "/hairModel_" + f'{id_hairModel:03d}'
    folder += "/mel_" + str(id_mel)
    folder += "/fB_" + str(id_fracBlood)
    folder += "/lesion_" + str(id_lesion)
    folder += "/T_" + f'{id_timePoint:03d}'
    folder += "/" + str(sel_lesionMat)
    folder += "/hairAlb_" + '-'.join([str(x) for x in sel_hair_albedo])
    folder += "/lesionScale_" + str(id_lesionScale) + "/"
    folder += "/light_" + sel_lightName + "/"
    if id_origin_y:
        folder += "/origin_y_" + str(id_origin_y) + "/"
    folder += "/mi_" + mi_variant + "/"
    folder += "/frame_" + str(id_frame) + "/"
    folder += "/calChart_" + str(id_calChart) + "/"
    folder += "/ruler_" + str(id_ruler) + "/"
    os.makedirs(folder, exist_ok=True)
    return folder