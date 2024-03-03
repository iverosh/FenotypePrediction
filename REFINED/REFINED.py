#python REFINED.py data/normalized_padel_feats_NCI60_672_small.csv res 1 0 10
from sys import argv
# [name, dataset_path, saving_path, iters_count]

# if len(argv) < 3:
#     print("Not enought args\nPlease input: saving path, count of hill climb iterations")
#     exit()

# path, iters_count = argv[1:]
# iters_count = int(iters_count)
iters_count = 10
path = "res"
import time
import cv2 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from Toolbox import two_d_eq, Assign_features_to_pixels
from sklearn.metrics.pairwise import euclidean_distances
import math
from itertools import product
import paraHill
from Toolbox import REFINED_Im_Gen
from myMDS import myMDS


start_time = time.time()
X = np.memmap("memmapped.dat", mode = "r", shape = (400, 407), dtype='int64').T

feature_names_list = ['Ca1_101073_A_T', 'Ca1_274437_T_C', 'Ca1_293229_G_A', 'Ca1_317478_C_A', 'Ca1_318266_A_G', 'Ca1_318271_G_A', 'Ca1_323797_G_T', 'Ca1_324527_T_G', 'Ca1_350395_T_A', 'Ca1_350494_A_G', 'Ca1_371165_G_T', 'Ca1_371175_T_A', 'Ca1_371216_A_G', 'Ca1_382119_G_C', 'Ca1_393327_C_G', 'Ca1_396182_C_T', 'Ca1_399402_C_A', 'Ca1_414572_G_A', 'Ca1_420648_T_C', 'Ca1_420734_T_C', 'Ca1_422010_T_C', 'Ca1_422611_G_A', 'Ca1_422617_G_A', 'Ca1_422623_T_G', 'Ca1_430734_T_C', 'Ca1_430735_G_A', 'Ca1_431646_A_G', 'Ca1_442737_C_A', 'Ca1_449010_T_C', 'Ca1_461587_T_A', 'Ca1_468077_A_C', 'Ca1_608516_T_G', 'Ca1_608554_C_A', 'Ca1_608649_T_A', 'Ca1_726015_A_C', 'Ca1_737218_T_G', 'Ca1_752961_A_T', 'Ca1_764277_C_T', 'Ca1_771989_A_G', 'Ca1_806263_A_G', 'Ca1_807787_C_T', 'Ca1_807822_G_A', 'Ca1_838327_T_C', 'Ca1_845841_C_A', 'Ca1_872675_C_T', 'Ca1_872682_C_A', 'Ca1_882266_C_A', 'Ca1_904772_A_G', 'Ca1_915567_A_T', 'Ca1_927678_C_T', 'Ca1_931602_T_A', 'Ca1_931728_T_C', 'Ca1_934186_T_C', 'Ca1_952261_A_G', 'Ca1_952309_C_T', 'Ca1_966729_C_A', 'Ca1_966776_T_C', 'Ca1_966831_C_T', 'Ca1_970249_T_A', 'Ca1_977274_A_G', 'Ca1_977303_A_G', 'Ca1_977337_C_T', 'Ca1_978810_G_A', 'Ca1_978866_G_A', 'Ca1_998157_A_C', 'Ca1_1001936_G_C', 'Ca1_1001945_T_A', 'Ca1_1023011_C_T', 'Ca1_1023887_A_G', 'Ca1_1024612_A_T', 'Ca1_1032992_C_A', 'Ca1_1032994_A_C', 'Ca1_1033034_A_C', 'Ca1_1055001_A_G', 'Ca1_1063583_G_A', 'Ca1_1067983_T_C', 'Ca1_1077356_A_G', 'Ca1_1080806_G_T', 'Ca1_1099901_T_A', 'Ca1_1113422_C_T', 'Ca1_1113541_A_T', 'Ca1_1118313_G_T', 'Ca1_1123177_T_A', 'Ca1_1123223_C_T', 'Ca1_1128406_T_C', 'Ca1_1128433_T_C', 'Ca1_1128455_A_T', 'Ca1_1128499_A_G', 'Ca1_1155142_C_A', 'Ca1_1167943_G_A', 'Ca1_1197395_A_C', 'Ca1_1197472_C_G', 'Ca1_1214874_T_C', 'Ca1_1224263_T_A', 'Ca1_1224311_T_A', 'Ca1_1224345_G_T', 'Ca1_1224365_C_A', 'Ca1_1242015_T_C', 'Ca1_1248550_C_G', 'Ca1_1270990_T_A', 'Ca1_1276531_C_T', 'Ca1_1284410_G_A', 'Ca1_1288271_A_G', 'Ca1_1295110_A_G', 'Ca1_1337817_G_T', 'Ca1_1338395_A_G', 'Ca1_1338422_G_A', 'Ca1_1343655_G_T', 'Ca1_1343729_C_T', 'Ca1_1353790_T_A', 'Ca1_1353798_G_C', 'Ca1_1353876_C_T', 'Ca1_1355093_G_T', 'Ca1_1362732_T_A', 'Ca1_1372278_A_G', 'Ca1_1372301_A_C', 'Ca1_1372317_C_G', 'Ca1_1380295_T_C', 'Ca1_1382930_G_A', 'Ca1_1382969_C_T', 'Ca1_1383024_C_G', 'Ca1_1391129_G_A', 'Ca1_1404820_T_C', 'Ca1_1404848_A_G', 'Ca1_1423164_T_C', 'Ca1_1441722_G_A', 'Ca1_1443612_T_G', 'Ca1_1443614_G_C', 'Ca1_1453072_C_A', 'Ca1_1457542_G_A', 'Ca1_1463497_G_C', 'Ca1_1463579_G_C', 'Ca1_1467331_G_T', 'Ca1_1492296_A_T', 'Ca1_1511501_A_G', 'Ca1_1511523_T_G', 'Ca1_1520328_C_A', 'Ca1_1528052_G_A', 'Ca1_1528162_G_A', 'Ca1_1532918_A_C', 'Ca1_1533118_C_T', 'Ca1_1534227_C_T', 'Ca1_1536432_G_T', 'Ca1_1545721_C_G', 'Ca1_1552406_A_C', 'Ca1_1553267_A_G', 'Ca1_1555825_A_T', 'Ca1_1555831_C_G', 'Ca1_1560796_G_A', 'Ca1_1562218_C_A', 'Ca1_1590875_G_T', 'Ca1_1622307_A_C', 'Ca1_1634236_G_A', 'Ca1_1634261_G_A', 'Ca1_1643959_C_T', 'Ca1_1647611_A_T', 'Ca1_1651662_T_G', 'Ca1_1661336_A_T', 'Ca1_1661337_C_T', 'Ca1_1666655_A_G', 'Ca1_1680426_T_C', 'Ca1_1680454_T_G', 'Ca1_1694313_T_A', 'Ca1_1711149_A_G', 'Ca1_1711166_G_A', 'Ca1_1711285_G_T', 'Ca1_1713338_G_T', 'Ca1_1713341_A_T', 'Ca1_1732351_C_G', 'Ca1_1737275_A_G', 'Ca1_1737311_C_G', 'Ca1_1737321_T_C', 'Ca1_1741298_C_T', 'Ca1_1744087_A_G', 'Ca1_1745468_T_A', 'Ca1_1745498_A_G', 'Ca1_1752930_G_A', 'Ca1_1769208_C_T', 'Ca1_1769313_G_A', 'Ca1_1775071_A_G', 'Ca1_1775117_G_A', 'Ca1_1792406_T_C', 'Ca1_1792412_C_T', 'Ca1_1802612_T_C', 'Ca1_1802655_T_C', 'Ca1_1802686_C_T', 'Ca1_1808643_A_T', 'Ca1_1808654_G_T', 'Ca1_1826905_C_T', 'Ca1_1826915_T_C', 'Ca1_1858664_A_G', 'Ca1_1858717_G_A', 'Ca1_1859502_T_G', 'Ca1_1859532_T_G', 'Ca1_1859541_T_A', 'Ca1_1859659_G_T', 'Ca1_1902178_G_A', 'Ca1_1931844_A_C', 'Ca1_1941178_T_A', 'Ca1_1963329_C_T', 'Ca1_1968309_G_A', 'Ca1_1970785_A_G', 'Ca1_1973210_T_C', 'Ca1_1975152_C_T', 'Ca1_2022225_C_T', 'Ca1_2044937_T_C', 'Ca1_2044940_C_A', 'Ca1_2081846_C_A', 'Ca1_2088933_A_T', 'Ca1_2089358_C_T', 'Ca1_2095150_G_T', 'Ca1_2118239_G_A', 'Ca1_2118249_C_T', 'Ca1_2123393_C_T', 'Ca1_2124581_A_G', 'Ca1_2124830_G_C', 'Ca1_2131479_A_G', 'Ca1_2131625_G_A', 'Ca1_2131662_T_C', 'Ca1_2131671_T_C', 'Ca1_2152476_A_G', 'Ca1_2160865_A_C', 'Ca1_2160933_A_T', 'Ca1_2171127_C_T', 'Ca1_2172623_G_T', 'Ca1_2189525_A_T', 'Ca1_2189645_G_C', 'Ca1_2198562_T_A', 'Ca1_2214749_G_A', 'Ca1_2218420_A_G', 'Ca1_2218626_T_G', 'Ca1_2218700_G_A', 'Ca1_2225735_T_A', 'Ca1_2234134_G_A', 'Ca1_2234145_A_G', 'Ca1_2234274_A_C', 'Ca1_2234278_A_T', 'Ca1_2238867_T_A', 'Ca1_2238888_A_C', 'Ca1_2238891_C_T', 'Ca1_2238958_T_C', 'Ca1_2259186_G_A', 'Ca1_2262576_T_C', 'Ca1_2262591_G_A', 'Ca1_2262633_G_T', 'Ca1_2270600_A_G', 'Ca1_2271409_T_C', 'Ca1_2271421_T_C', 'Ca1_2277599_C_T', 'Ca1_2309179_A_G', 'Ca1_2313076_A_T', 'Ca1_2313615_A_G', 'Ca1_2320673_C_A', 'Ca1_2320691_C_T', 'Ca1_2328979_A_G', 'Ca1_2330274_A_C', 'Ca1_2330287_C_G', 'Ca1_2342137_C_A', 'Ca1_2360375_G_T', 'Ca1_2364872_T_C', 'Ca1_2373466_A_G', 'Ca1_2373480_T_G', 'Ca1_2374345_A_C', 'Ca1_2374520_T_A', 'Ca1_2375085_C_T', 'Ca1_2375294_A_T', 'Ca1_2375928_G_A', 'Ca1_2383974_G_C', 'Ca1_2388102_C_T', 'Ca1_2389626_G_A', 'Ca1_2398988_A_G', 'Ca1_2400346_T_C', 'Ca1_2411993_A_G', 'Ca1_2432376_A_G', 'Ca1_2437208_G_T', 'Ca1_2437317_G_A', 'Ca1_2440912_T_A', 'Ca1_2441018_C_T', 'Ca1_2450200_T_C', 'Ca1_2450209_C_T', 'Ca1_2469345_G_A', 'Ca1_2469867_G_A', 'Ca1_2480310_G_A', 'Ca1_2496727_C_T', 'Ca1_2510183_G_C', 'Ca1_2511223_T_A', 'Ca1_2511238_A_G', 'Ca1_2511250_G_A', 'Ca1_2511253_C_T', 'Ca1_2512573_A_T', 'Ca1_2513046_A_G', 'Ca1_2524005_T_C', 'Ca1_2524021_G_C', 'Ca1_2533229_G_A', 'Ca1_2548183_G_T', 'Ca1_2579883_T_C', 'Ca1_2598554_A_G', 'Ca1_2640175_C_A', 'Ca1_2703473_T_G', 'Ca1_2704240_A_G', 'Ca1_2704247_A_G', 'Ca1_2730646_C_G', 'Ca1_2788788_C_G', 'Ca1_2800975_A_C', 'Ca1_2809226_C_T', 'Ca1_2834171_G_A', 'Ca1_2834216_T_A', 'Ca1_2837698_G_T', 'Ca1_2854316_T_A', 'Ca1_2869033_T_A', 'Ca1_2869467_C_T', 'Ca1_2869532_T_A', 'Ca1_2892256_A_G', 'Ca1_2892263_A_C', 'Ca1_2900696_G_A', 'Ca1_2900699_T_C', 'Ca1_2902723_C_T', 'Ca1_2902744_A_C', 'Ca1_2909737_C_A', 'Ca1_2909739_G_A', 'Ca1_2921443_C_A', 'Ca1_2928372_G_A', 'Ca1_2933533_G_A', 'Ca1_2936230_T_C', 'Ca1_2948112_A_C', 'Ca1_2964451_G_A', 'Ca1_2964512_C_A', 'Ca1_2981187_G_A', 'Ca1_2993011_C_A', 'Ca1_2995302_C_T', 'Ca1_3048432_T_C', 'Ca1_3053548_T_C', 'Ca1_3053615_T_C', 'Ca1_3053662_T_C', 'Ca1_3081553_T_A', 'Ca1_3084224_C_T', 'Ca1_3084234_T_A', 'Ca1_3099941_T_C', 'Ca1_3102387_C_T', 'Ca1_3120137_A_G', 'Ca1_3120143_G_A', 'Ca1_3120163_A_G', 'Ca1_3120165_C_A', 'Ca1_3142432_C_T', 'Ca1_3302504_C_G', 'Ca1_3953643_A_G', 'Ca1_4235091_C_A', 'Ca1_4244360_A_G', 'Ca1_4249113_C_T', 'Ca1_4379745_C_A', 'Ca1_4474445_C_G', 'Ca1_4518653_A_T', 'Ca1_4766059_A_T', 'Ca1_4782895_C_T', 'Ca1_5124322_C_A', 'Ca1_5367087_A_C', 'Ca1_5371236_C_T', 'Ca1_5391354_T_G', 'Ca1_5391371_C_T', 'Ca1_5428038_T_C', 'Ca1_5428080_T_A', 'Ca1_5441052_T_C', 'Ca1_5441097_C_T', 'Ca1_5480709_G_C', 'Ca1_5511919_C_T', 'Ca1_5511968_A_C', 'Ca1_5511996_C_A', 'Ca1_5515026_T_C', 'Ca1_5614996_A_G', 'Ca1_5619968_A_T', 'Ca1_5632070_G_T', 'Ca1_5638251_G_A', 'Ca1_5638273_T_C', 'Ca1_5645640_A_G', 'Ca1_5720006_T_A', 'Ca1_5720035_T_A', 'Ca1_5731384_T_C', 'Ca1_5742731_G_A', 'Ca1_5745277_G_A', 'Ca1_5757141_C_T', 'Ca1_5758008_T_A', 'Ca1_5963423_C_T', 'Ca1_6132150_C_T', 'Ca1_6138577_T_C', 'Ca1_6138657_A_G', 'Ca1_6140949_G_A', 'Ca1_6144083_A_G', 'Ca1_6151476_G_C', 'Ca1_6154711_G_A', 'Ca1_6155943_G_C', 'Ca1_6155945_T_G', 'Ca1_6162693_T_C', 'Ca1_6165883_C_A', 'Ca1_6277507_T_C', 'Ca1_6277549_A_G', 'Ca1_6282159_C_A', 'Ca1_6314770_A_C', 'Ca1_6327680_C_A', 'Ca1_6327709_G_T', 'Ca1_6330555_A_G']

# print(X)
Nn = len(feature_names_list)                                                              # Number of features
nn = math.ceil(np.sqrt(Nn))                                                   			     # Image dimension
print(Nn, nn)

Euc_Dist = euclidean_distances(X.T)
Euc_Dist = np.maximum(Euc_Dist, Euc_Dist.transpose())  
# print(Euc_Dist.dtype)       			                 

mds_xy = myMDS(Euc_Dist)
eq_xy = two_d_eq(mds_xy, Nn)                                        # -> [0,1]
Img = Assign_features_to_pixels(eq_xy,nn,verbose=0)					# Img is the none-overlapping coordinates generated by MDS



# Convert from 'F34' to int 34
init_map = np.char.strip(Img.astype(str),'F').astype(int)
map_in_int = init_map


for iter_num in range(iters_count):
    init_coords = [x for x in product([0,1,2],repeat = 2)]
    for init_coord in init_coords:
        # generate the centroids
        xxx = [init_coord[0]+i*3 for i in range(int(nn/3)+1) if (init_coord[0]+i*3)<nn]
        yyy = [init_coord[1]+i*3 for i in range(int(nn/3)+1) if (init_coord[1]+i*3)<nn]
        centr_list = [x for x in product(xxx,yyy)]
        swap_dict = paraHill.evaluate_centroids_in_list(centr_list,Euc_Dist,map_in_int)
        # print(swap_dict)
        
        map_in_int = paraHill.execute_dict_swap(swap_dict, map_in_int)

        print(">",init_coord,"Corr:",paraHill.universial_corr(Euc_Dist,map_in_int))


coords = np.array([[item[0] for item in np.where(map_in_int == ii)] for ii in range(Nn)])

X_REFINED_MDS = REFINED_Im_Gen(X[:,:],nn, map_in_int, feature_names_list, coords)


X_reshaped = X_REFINED_MDS.reshape(X_REFINED_MDS.shape[0], nn, nn)
if not os.path.exists(path):
    os.mkdir(path)
np.save('allimg.npy', X_reshaped)
for i in range(X_reshaped.shape[0]):
    plt.imsave(f"""{path}/img{i}.png""", X_reshaped[i], cmap='gray')
print(X_reshaped.shape)
end_time = time.time()
execution_time = end_time - start_time

print("Время выполнения кода: ", execution_time, "секунд")



