#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:42:00 2022

@author: arat
"""

import numpy as np
import os

import aux_parameters_functions as parf
import lineage_fit_main as lfit


IS_PLOTTED = True
PROC_COUNT = 1
SIMU_COUNT = 1000

IS_SAVED = True
# ..............
if IS_SAVED:
    FIG_DIRECTORY = "figures/manuscript"
# ..............
else:
    FIG_DIRECTORY = None
DIR = FIG_DIRECTORY + '/parameters/'


# POINT_0 = [0.0247947389, 0.440063202, 0.186824276, 0.725200993, 27.0,
#            2.45423414e-06, 0.122028128, 0.0, 40.0, 58.0]


# CMAES fits
# ----------

FITS = {}
PAR_SPACES = {}


# 5 parameter fits
# ................
PAR_SPACES['5'] = {"is_sen_commun_to_types": True,
                   "is_lmin_a_fixed": False,
                   "is_ltrans_fixed": True,
                   "is_l0_fixed": True, "is_l1_fixed": True, "lmin_a": None,
                   "par_l_init": [0, 0, 0]}

# D_{5, 1} new_parameters/bounds_0-2_0-80_0-80/...
FITS['5_1'] = [# ...fit_par5_weight.5-1
               [0.023096746389807365, 0.06583187577578642, 0.02403692154854273,
                0.12503198256667541, 0.0], # 5_1
               [0.025789160404743786, 0.09869194763997158, 0.02758270866064814,
                0.16579507459111156, 0.0], # 5_3 bis
               [0.023420185983363567, 0.07049488191330434, 0.02408546552079708,
                0.12723348101742243, 1.0], # 5_3
               [0.03297689323308528, 0.46589281156079554, 0.04501884174290883,
                0.3353400322554727, 0.0], # 5_4 (s0.7)
               [0.024236194483896162, 0.06670517125590057,
                0.023827595532913155, 0.12866047622162408, 0.0], # 5_6
               [0.04427255723281527, 0.5115323176760375, 0.007058417697529897,
                0.015022250682394364, 26.0], # 5_7
               # ...fit_par5_weight.5-1-wrt-count
               [0.04367222052103024, 0.8108927414361464, 0.022805104035329017,
                0.0661556568786559, 19.0]] # 5_9 new

# D_{5, 2} good_fits_3
FITS['5_2'] = [[0.044623051910801084, 0.7659987487642193, 0.02628774196603994,
                0.0650672533938157, 21.521123077794247],
               [0.03585852571166418, 0.3445600966245351, 0.01175205248660626,
                0.024348334151752377, 20.122040115998285],
               [0.03344657707867535, 0.2135724456778484, 0.007338962418497969,
                0.020938825209574286, 20.048578318706273],
               [0.0473110375313884, 0.7550959690112771, 0.05755424832125892,
                0.8380049092390066, 19.50367308939538],
               [0.032950287727107605, 0.1987760223822911, 0.016221892028331707,
                0.047537438563538065, 19.725107661256878]]

FITS['5'] = np.concatenate(([FITS['5_1'], FITS['5_2']]), 0)


# 6 parameter fits (D_6)
# ......................
PAR_SPACES['6'] = {"is_sen_commun_to_types": True,
                  "is_lmin_a_fixed": False,
                  "is_ltrans_fixed": False,
                  "is_l0_fixed": True, "is_l1_fixed": True, "lmin_a": None,
                  "par_l_init": [None, 0, 0]}

FITS['6'] = [# new_parameters/new/
             [0.0249333485686157, 0.229501229046105, 0.024554428981260305,
              0.20139925064004008, 27.0,  24.0], # 8_9
             [0.024942685674423464, 0.21412115454754932, 0.022860811264906734,
              0.16443838056510174, 27.0, 23.0], # 8_10
             [0.02508881270780235, 0.22140558287212533, 0.021206348556090666,
              0.142783049838495, 29.0, 24.0], # 8_11
             [0.024455832312792105, 0.2426269853962198, 0.024582002974013424,
              0.23926475296920502, 29.0, 30.0], # 8_12
             # new_parameters/new_new/
             [0.0466960844, 0.913188659, 0.0101757547, 0.0196079358, 27.0,
              4.0], # 8_9
             [0.025152923955762675, 0.1972024601390917, 0.02150781630514025,
              0.16558309111017744, 29.0, 25.0]] # 8_10


# 7 parameter fits (D_7)
# ......................
PAR_SPACES['7'] = {"is_sen_commun_to_types": False,
                   "is_lmin_a_fixed": True,
                   "is_ltrans_fixed": True,
                   "is_l0_fixed": True, "is_l1_fixed": True, "lmin_a": 30,
                   "par_l_init": [0, 0, 0]}

FITS['7'] = [# new_parameters/new/
             [0.03809525602850779, 0.6764722910004339, 0.036943798113969264,
              0.042346167124904174, 7.484301204966951e-07, 0.07786415091059952,
              0.0], # 8_5
             [0.04040875386746258, 0.6809189008317653, 0.40235747676055694,
              0.20624895081822117, 3.4933262080606965e-05, 0.10738773054545452,
              24.0], # 8_6
             [0.03703438566796348, 0.6878879548518446, 0.5245246203619681,
              0.56529495818464, 7.776600509500184e-07, 0.07320873305047698,
              0.0], # 8_7
             [0.041066036712026896, 0.9202320898287035, 0.7706768875364964,
              0.8735058558966649, 7.080916875407512e-05, 0.08613168070734047,
              0.0], # 8_8
             # new_parameters/new_new
             [0.0399363414, 0.925618414, 0.613401558, 0.952952601,
              1.79628396e-06, 0.0759136355, 0.0], # 8_5
             [0.03563359434297483, 0.5892402176627066, 0.9961187560927056,
              0.05834534563436748, 4.859076465521392e-07, 0.07908282073699802,
              0.0]] # 8_6


# 8 parameter fits
# ................
PAR_SPACES['8'] = {"is_sen_commun_to_types": False,
                   "is_lmin_a_fixed": False,
                   "is_ltrans_fixed": True,
                   "is_l0_fixed": True, "is_l1_fixed": True, "lmin_a": None,
                   "par_l_init": [0, 0, 0]}

# D_{8, 1} new_parameters/bounds_0-2_0-80_0-80/
FITS['8_1'] = [# ...fit_par8_weight.5-1
               [0.05444814749298596, 0.9773012633070454, 0.708352114321143,
                0.5729491698553342, 42.0, 0.9265217721111532,
                0.6574391326356714, 24.0], # 8_1
               [0.04569543318138426, 0.8789408459663268, 0.008718367444882736,
                0.01759034760095099, 33.0, 0.330796502888141,
                1.9082589716474294, 28.0], # 8_1 bis
               [0.04860489273374364, 0.783045672729122, 0.995345378826493,
                0.07353216275885957, 42.0, 0.8286137171421877,
                1.2210678403186481, 25.0], # 8_2
               [0.04090611802124736, 0.3526537494643214, 0.07445693667087958,
                1.6935702995960527, 12.0, 0.03313629834344905,
                0.4092908565839665, 17.0], # 8_2 bis
               [0.02541930039708978, 0.3161188810068128, 0.8558747036533583,
                0.5987714310976386, 0.0, 2.120541523014397e-05,
                0.08822084358043356, 0.0], # 8_3
               [0.026318335720942127, 0.3072855547448646, 0.9703609355736703,
                1.7463217147589079, 0.0, 1.1947444469585629e-06,
                0.10690422778634473, 0.0], # 8_3 bis
               [0.027920789432492644, 0.3428470240244475, 0.9954146286428511,
                0.03850112309028803, 0.0, 3.2095223852787346e-07,
                0.10892359260131551, 0.0], # 8_4
               [0.03925412888444615, 0.5016433742710762, 0.03214460182169152,
                0.17729940565442176, 0.0, 0.9701133338506821,
                0.05521409283347772, 14.0], # 8_7
               # ...fit_par8_weight.5-1-wrt-count
               [0.025154449389371336, 0.25637137728070253, 0.5588440524832747,
                0.7861336757668571, 0.0, 4.863925108501737e-06,
                0.12580565434369623, 0.0], # 8_9 new
               [0.024979334547893313, 0.252558691217324, 0.8080256452970531,
                0.8255408692012838, 1.0, 9.14033529304837e-06,
                0.12054322647277455, 0.0] # 8_9 new bis
               ]

# D_{8, 2} new_parameters/bounds_0-1_20-80_0-60_weight-sensibility_
#                                 popsize58-64-70_propB.25_sigma.5_tcomputNone
FITS['8_2'] = [[0.0335402759396807, 0.5592098334257327, 0.9550179823818613,
                0.11500711071125647, 20.0, 5.547052522181879e-05,
                0.08703654183657589, 0.0], # 8_1
               [0.04462486098761796, 0.6012088502354092, 0.9339645752793777,
                0.18548456048739237, 43.0, 0.1230221878478705,
                0.3211574581195019, 18.0], # 8_2
               [0.034248485579054036, 0.5639698274948275, 0.9403173131685366,
                0.8218540131283341, 20.0, 0.00011651822879516314,
                0.0895562165072471, 0.0], # 8_3
               [0.033925186243295796, 0.5447444478821817, 0.08999921270583844,
                0.005689709841017046, 20.0, 3.0141026354050095e-05,
                0.11404063115936902, 0.0], # 8_4
               [0.041767748870233865, 0.5562354302574486, 0.6060365149108222,
                0.45356574701044583, 35.0, 9.944462651139446e-05,
                0.05998955413065765, 22.0], # 8_5
               [0.02924923104997743, 0.4698345403550679, 0.1956283058804279,
                0.2888100091615281, 20.0, 1.5889926921625606e-05,
                0.07463674041201719, 0.0], # 8_6
               [0.042216004228343894, 0.9347564797050445, 0.11155772442114213,
                0.3790751031910482, 23.0, 0.039758958453873935,
                0.7654964048894426, 6.0], # 8_7
               [0.03627428153466001, 0.7326115419032979, 0.36807852855181006,
                0.09035373246053419, 20.0, 7.067520714423553e-07,
                0.08634428321190975, 1.0], # 8_8
               [0.031463281600428404, 0.4462292579776469, 0.24088557962811014,
                0.18318726004089136, 20.0, 3.724996895156016e-05,
                0.09086082810039844, 0.0], # 8_1 new
               [0.0469645275, 0.946763088, 0.000258276466, 0.00859915981, 32.0,
                0.75712434, 0.984806376, 25.0], # 8_2 new
               [0.04167311, 0.77978366, 0.30329655, 0.75564247, 31.0,
                0.00641842, 0.11603488, 27.0], # 8_3 new
               [0.03148669647668701, 0.42590160638673835, 0.46808840501929827,
                0.48464513313325713, 20.0, 0.0003879032682806152,
                0.10037006731530733, 0.0], # 8_4 new
               [0.0346671492, 0.583057834, 0.221637096, 0.284716527, 20.0,
                0.00032902313, 0.0925905135, 0.0], # 8_6 new
               [0.02706138811179183, 0.20738590117335168, 0.010039528985475827,
                0.014525433442926361, 20.0, 4.371659520047529e-07,
                0.07634533773721339, 0.0], # 8_7 new
               [0.0357026564, 0.594344344, 0.995423828, 0.626655447, 20.0,
                5.42294461e-06, 0.0881984979, 0.0]] # 8_8 new

# D_{8, 3}
FITS['8_3'] = [# new_parameters/new/
               [0.0363201604920173, 0.6211172811786885, 0.9864774449495055,
                0.16944472757074613, 27.0, 7.424530550205999e-07,
                0.08828183952176077, 0.0], # 8_1
               [0.03841637439912296, 0.7481457814752552, 0.8240846627497458,
                0.3261703068220255, 27.0, 1.283864219031294e-05,
                0.08636724433274705, 0.0], # 8_2
               [0.03724653803074866, 0.6958778428983051, 0.9216423981870685,
                0.6807696641628169, 27.0, 9.031326027170439e-05,
                0.08786764923069355, 0.0], # 8_3
               [0.0375406603060553, 0.7017620348382265, 0.8660465092679146,
                0.3365240206406175, 27.0, 9.058109798633986e-05,
                0.0833715995728713, 0.0], # 8_4
               # new_parameters/new_new
               [0.04040792, 0.61568435, 0.00982868, 0.00921592, 28.0,
                0.02301651, 0.21884089, 19.0], # 8_1
               [0.036495921099194036, 0.6279258979505797, 0.5474392340718554,
                0.688439147758897, 27.0, 2.4747892213276166e-05,
                0.08502674244397229, 0.0], # 8_2
               [0.03965715341417918, 0.8029540833235593, 0.9926935351310178,
                7.343267949289595e-05, 27.0, 3.2413484640568345e-08,
                0.08784157664727238, 0.0], # 8_3
               [0.03331144075157841, 0.4828395798508738, 0.8360098238668126,
                0.952496677266192, 27.0, 3.156898018751744e-05,
                0.08032483212580444, 0.0]] # 8_4

# D_{8, 4} fit/ 8_1_new
FITS['8_4'] = [[0.024113021576096363, 0.5114710804803486, 0.4627320883748722,
                0.7692729061454864, 3.510079199420031e-05, 0.1279767514692167,
                0.7500973343114539, 30.345159170435515]]

FITS['8'] = np.concatenate((FITS['8_1'], FITS['8_2'], FITS['8_3'],
                            FITS['8_4']), 0)


# 9 parameter fits
# ................
PAR_SPACES['9'] = {"is_sen_commun_to_types": False,
                   "is_lmin_a_fixed": False,
                   "is_ltrans_fixed": False,
                   "is_l0_fixed": True, "is_l1_fixed": True, "lmin_a": None,
                   "par_l_init": [None, 0, 0]}

# D_{9_1} fit/
FITS['9_1'] = [[0.022937605161445086, 0.45285929375401757, 0.9981853952033533,
                0.913105213243456, 27.0, 2.41253073806292e-05,
                0.12006445525790242, 0.0, 30.0], # 9_1 outcmaes_1
               [0.022600228658682216, 0.44692840432054726, 0.5264037537081989,
                0.8554713828074565, 28.0, 1.5219703893180832e-06,
                0.11621693497058465, 0.0, 30.0], # 9_2 outcmaes_1
               [0.02289913695825492, 0.4546482376039831, 0.6966767333809484,
                0.9999391747472598, 27.0, 1.679115760992804e-05,
                0.11950532643134654, 0.0, 30.0], # 9_3 outcmaes_1
               [2.34848700e-02, 4.90276758e-01, 2.48240574e-01, 8.81285491e-01,
                3.00407786e+01, 4.54379693e-06, 1.14720383e-01, 1.82232589e+00,
                30], # 9_1 outcmaes_1
               [0.024093671922855033, 0.5066605493532886, 0.48015555183193004,
                0.394686739092988, 26.64568542275716, 2.0816481857595947e-07,
                0.12058125783987778, -0.49886474399478853, 30], # 9_2 cmaes_new
               [2.33212133e-02, 4.48589525e-01, 9.63567774e-01, 9.39539544e-01,
                2.66963691e+01, 5.50737210e-05, 1.25412734e-01, 2.78657325e+00,
                28], # 9_3 outcmaes_1
               [2.29507091e-02, 4.56081665e-01, 9.54441828e-01, 5.58172764e-03,
                2.65734495e+01, 2.00444649e-04, 1.25235248e-01, -3.33154127e-01,
                30], # 9_3 outcmaes_1 incubent,
               # [0.0229, 0.4546, 0.6967, 0.9999, 27, 0, 0.1195, 0, 30], # ''rounded
               [0.02296461985287969, 0.45674637159422016, 0.47129787805292034,
                0.991295906356734, 26.832094987311894, 0.00018816118719657752,
                0.12509223147530557, -0.22981758923623374, 30]] # ''x'

# D_{9_2} fit/
FITS['9_2'] = [[0.024332666314842176, 0.5113255633160987, 0.979038888225393,
                0.000408697057169946, 27.0, 1.2732873989983443e-09,
                0.12060410754347281, 0.0, 29.0], # 9_4 outcmaes_1
               [0.022859077436928287, 0.4521633058547674, 0.994086427464303,
                0.47953885932967055, 21.0, 2.8347331776635323e-08,
                0.11949347893534912, 1.0, 30.0]] # 9_5 outcmaes_1

FITS['9'] = np.concatenate((FITS['9_1'], FITS['9_2']), 0)


# 10 parameter fits
# .................
PAR_SPACES['10'] = {"is_sen_commun_to_types": False,
                    "is_lmin_a_fixed": False,
                    "is_ltrans_fixed": True,
                    "is_l0_fixed": False, "is_l1_fixed": False, "lmin_a": None,
                    "par_l_init": [0, None, None]}

# D_10 fit/fit_
FITS['10'] = [[0.0247947389, 0.440063202, 0.186824276, 0.725200993, 27.0,
               2.45423414e-06, 0.122028128, 0.0, 40.0, 58.0], # last par!
              [3.11977272e-02, 7.08488041e-01, 6.39997524e-01, 9.95725810e-01,
               2.99895644e+01, 6.61497178e-05, 1.10345013e-01, 1.26960113e+00,
               3.03750425e+01, 4.28410175e+01], # 10_4
              [2.59744456e-02, 4.53318593e-01, 9.80110940e-01, 8.27586703e-01,
               2.65208687e+01, 2.24794575e-06, 8.98147045e-02, -4.90469702e-01,
               3.04792422e+01, 6.00879394e+01], # 10_4 bis
              [3.05800239e-02, 6.21520040e-01, 9.80201063e-01, 7.26433310e-01,
               2.66475896e+01, 6.44399176e-06, 1.12620357e-01, -3.67367622e-01,
               2.75092992e+01, 4.05917186e+01], # 10_5 14.08
              [2.85808329e-02, 6.18166313e-01, 9.22107732e-01, 1.31813745e-01,
               2.80694803e+01, 1.71455064e-04, 9.29822205e-02, 2.13558718e+00,
               2.78569000e+01, 5.45032817e+01], # 10_6 13.36
              [3.10880022e-02, 7.00379878e-01, 5.06486973e-01, 4.79592289e-01,
               3.01849515e+01, 5.61878654e-06, 9.30558612e-02, 1.96219450e+00,
               2.77586448e+01, 2.81526632e+01], # 10_7 14.93
              [2.85110589e-02, 5.69001958e-01, 6.64886546e-01, 2.63683068e-03,
               2.66907983e+01, 5.37129663e-05, 9.04086534e-02, 9.68365609e-01,
               2.75351189e+01, 4.83074602e+01], # 10_11 13.22
              [2.91255272e-02, 5.84469259e-01, 1.88953283e-01, 7.99940081e-01,
               2.65253070e+01, 1.10974247e-05, 1.01186643e-01, 1.05439533e+00,
               2.76905490e+01, 5.22594891e+01], # 10_11 13.08
              [0.0257504882705583, 0.4489558810390396, 0.8015293954292777,
               0.9492502399974612, 27.016459402528817, 1.116656258395159e-05,
               0.09017955869240737, 0.17088370369693062, 30.283881776080054,
               60.38838330942985],
              [0.03119772716291226, 0.7084880407827931, 0.6399975244725553,
               0.9957258100484327, 29.989564393431188, 6.614971779055046e-05,
               0.11034501314474954, 1.2696011337228121, 30.37504253344168,
               42.84101750399293], # unknown
              [0.027353248849121672, 0.5292479218935842, 0.6965002616359262,
               0.5784232864134902, 26.917095647285205, 1.6623334349802218e-06,
               0.09263527435395125, 0.12106945032600497, 29.80477859997253,
               50.096960653428795], # 10_5 13.49
              [0.028368873401920246, 0.5803900846685431, 0.9942956343200258,
               0.6690081396086305, 26.926832840515946, 2.9685180009290762e-08,
               0.10022805986221615, 1.7741282358035804, 31.49799909200641,
               60.44570721384862], # 10_7 13.10
              [0.02753905817554722, 0.5426916396139521, 0.9233199690303889,
               0.1994291922110803, 31.287240548856836, 0.00011523854579929581,
               0.10138672762691626, 2.636830088702448, 28.999419407476545,
               60.49075581536706], # 10_10 14.25
              [0.027329149153181267, 0.5506886238498353, 0.5916118649343631,
               0.7803421860870299, 26.509039673295007, 0.00017538056245629935,
               0.11343010464056746, -0.2490060434390846, 35.459860606357154,
               50.45052202294791], # 10_4new2 12.55
              [0.029585924242189237, 0.6255118367218917, 0.1221282374574072,
               0.09477730158711453, 27.147293456418968, 2.9260039506011045e-07,
               0.10209626004421057, 0.4765383470499529, 30.098450341489894,
               30.280576268797667], # 10_4new 13.46
              [0.029585329570609633, 0.62551596001981, 0.1253558109124526,
               0.08412971557124121, 26.95284900681613, 5.870386365626226e-06,
               0.10211582743498271, 0.09580787153586812, 30.258217803098752,
               30.07531123135822], # 10_4new 13.46
              [2.30198266e-02, 4.16860456e-01, 5.73590848e-01, 5.06827807e-01,
               2.68404858e+01, 7.28086973e-10, 8.71135089e-02, 8.85052810e-02,
               3.89259315e+01, 5.36821268e+01], # 10_from_best_1 10.53
              [2.49819049e-02, 4.38510117e-01, 9.42405769e-01, 6.28298641e-01,
               2.66366389e+01, 8.54756010e-07, 1.19365999e-01, -4.97829404e-01,
               4.03729076e+01, 3.56191523e+01],  # 10_from_best_1 11.62
              [2.36940178e-02, 4.00627308e-01, 6.35231288e-01, 7.73880620e-01,
               2.70581734e+01, 3.96476926e-06, 1.15755990e-01, -2.57061527e-01,
               4.01641628e+01, 5.19387073e+01], # 10_from_best_2 11.32
              [ 2.40316320e-02, 4.63689940e-01, 5.71402580e-01, 8.43848006e-01,
               2.69187637e+01, 2.56552076e-04, 9.43694141e-02, -2.38298395e-01,
               4.02448979e+01, 5.37239266e+01], # 10_from_best_2 10.80
              [2.63775524e-02, 4.50198115e-01, 2.01542786e-01, 8.12655215e-01,
               2.65133445e+01, 4.76322590e-05, 1.08859759e-01, -1.95486674e-01,
               3.10598877e+01, 6.04996589e+01], # 10_from_best_3 10.82
              [2.47947389e-02, 4.40063202e-01, 1.86824276e-01, 7.25200993e-01,
               2.65085099e+01, 2.45423414e-06, 1.22028128e-01, 2.20929270e-01,
               4.03317404e+01, 5.76659966e+01] # 10_from_best_3 10.67
              ]


lengths = [len(FITS['5']), len(FITS['6']), len(FITS['7']), len(FITS['8']),
           len(FITS['9']), len(FITS['10'])]

# Parameters for the job array. `idx` should ran from 0 to `job_count -1`.
job_count = sum(lengths)
if __name__ == "__main__":
    print(f"SLURM_ARRAY_TASK_ID should ran from 0 to {job_count -1}")
is_run_in_parallel_from_slurm = "SLURM_ARRAY_TASK_ID" in os.environ.keys()

# If parallel computation run from sbacth command, only one idx computed.
if is_run_in_parallel_from_slurm:
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    idxs = [idx]
# Otherwise computation in serie.
else:
    idxs = np.array([])
    if __name__ == "__main__":
        idxs = np.arange(job_count)



# Iteration on all jobs to run.
for run_idx in idxs:
    print(f'\n Simulation n° {run_idx + 1} / {job_count}')

    # No dilatation of L_INIT, possibly translation
    # ---------------------------------------------
    if run_idx < sum(lengths[:-1]):
        # > Fit on 5 parameters.
        if  run_idx < lengths[0]:
            PAR_SPACE_CHOICE = PAR_SPACES['5']
            FIT = FITS['5'][run_idx]

        # > Fit on 6 parameters.
        elif run_idx < sum(lengths[:2]):
            PAR_SPACE_CHOICE = PAR_SPACES['6']
            FIT = FITS['6'][run_idx - lengths[0]]
            print(run_idx, run_idx - lengths[0])

        # > Fit on 7 parameters.
        elif run_idx < sum(lengths[:3]):
            PAR_SPACE_CHOICE = PAR_SPACES['7']
            FIT = FITS['7'][run_idx - sum(lengths[:2])]
            print(run_idx, run_idx -  sum(lengths[:2]))

        # > Fit on 8 parameters.
        elif run_idx < sum(lengths[:4]):
            PAR_SPACE_CHOICE = PAR_SPACES['8']
            FIT = FITS['8'][run_idx - sum(lengths[:3])]
            print(run_idx, run_idx - sum(lengths[:3]))

        # > Fit on 9 parameters.
        elif run_idx < sum(lengths[:5]):
            PAR_SPACE_CHOICE = PAR_SPACES['9']
            FIT = FITS['9'][run_idx - sum(lengths[:4])]
            print(run_idx, run_idx - sum(lengths[:4]))

    # Dilatation of L_INIT
    # --------------------
    else:
        # > Fit on 10 parameters.
        PAR_SPACE_CHOICE = PAR_SPACES['10']
        FIT = FITS['10'][run_idx - sum(lengths[:5])]

    PARAMETERS = lfit.point_to_cost_fct_parameters(FIT, kwarg=PAR_SPACE_CHOICE)
    print('Parameters NTA: ', PARAMETERS[0])
    print('Parameters SEN: ', PARAMETERS[1])
    print('Parameters L_INIT: ', PARAMETERS[2])
    if IS_PLOTTED:
        parf.plot_laws(PARAMETERS)
    lfit.compute_n_plot_gcurves(FIT, kwarg=PAR_SPACE_CHOICE,
                                is_plotted=IS_PLOTTED, proc_count=PROC_COUNT,
                                simulation_count=SIMU_COUNT)
