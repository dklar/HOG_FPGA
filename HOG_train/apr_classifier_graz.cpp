float Intercept = [-0.69022634];
float classifier[] = {
0.39763,-0.44933,-0.28633,-0.26239,-0.28383,-0.12976,-0.03283,-0.40134,0.27085,0.20832,0.14450,-0.38930,-0.25899,-0.09970,-0.10096,-0.39905,0.81675,0.06593,0.00089,-0.14158,-0.15057,
-0.32873,0.60537,-0.17601,0.00990,0.01332,0.12530,-0.00461,-0.11752,-0.55026,-0.26100,0.60739,0.05889,0.20457,0.24473,0.35954,-0.28018,-0.10305,0.02662,0.08754,0.31600,-0.44094,
-0.58437,-0.59817,0.20521,-0.70841,-0.30439,0.45788,0.53605,0.47233,0.70178,-0.35246,-0.72923,-0.46307,-0.12582,0.42329,0.15593,0.14728,-0.34308,-0.28111,-0.89024,-0.25709,-0.21258,
0.08266,1.13686,0.20700,0.38414,-0.81967,0.55472,0.07740,0.04456,0.78639,-0.46303,-0.35618,-0.59660,0.57095,0.30220,-0.02013,0.55612,-0.15772,-0.21670,-0.54779,-0.86266,-0.91145,
0.01242,0.22276,-0.25734,0.09367,-0.23157,-0.49921,-0.11730,0.15642,0.20325,0.16350,0.12391,0.37473,0.88582,0.67078,0.37085,-0.10032,-0.06965,-0.69960,-0.63500,0.04049,-0.00696,
0.56214,0.69472,0.05426,-0.19623,-0.23691,-0.68121,-0.15633,-0.11053,-0.03190,0.21876,-0.13469,-0.26406,0.04780,-0.23018,-0.47920,0.04820,0.10701,-0.21784,-0.37168,-0.08421,0.38297,
0.26061,0.35888,-0.18000,-0.07586,0.54949,-0.07922,0.28594,0.25046,0.65517,-0.08545,-0.09453,-0.69809,0.21312,0.36245,-0.20755,-0.23693,-0.22788,0.64938,0.00020,-0.19019,-0.75816,
-0.30940,-1.07304,0.63727,0.07925,-0.02480,0.17000,-0.13463,-0.29881,0.74537,0.41288,0.29137,-0.09876,0.35897,-0.30465,0.10462,-0.27915,0.21350,-0.49780,-0.71071,-0.21679,-0.10195,
-0.32216,-0.20647,-0.56326,0.18676,0.83858,1.10577,0.66852,-0.02881,-0.06590,0.15658,-0.08252,-0.39193,-0.20748,0.03851,0.29256,1.02968,0.18428,0.24205,0.02227,-0.03207,-0.37908,
-0.44307,0.32214,0.18372,0.58767,-0.36876,0.57940,0.68761,0.27617,-0.43023,-0.45645,-0.00124,0.65779,-0.13133,-0.29768,-0.55256,-0.25842,-0.57426,-0.59694,-0.86778,0.27952,0.77465,
-0.01149,0.10445,0.12451,-0.12891,-0.39833,-0.94112,-0.07615,0.18474,0.37593,0.45797,0.20680,0.28654,0.69274,0.59980,0.59195,0.10212,-0.24926,-0.17133,-0.37939,0.36078,0.12319,
1.06515,-0.30788,-0.34771,-0.25647,-0.00589,-0.02277,-0.23427,-0.07936,0.20369,0.41320,-0.02701,-0.68477,-0.95887,-0.22427,-0.75573,-0.63169,-0.74188,-0.20027,0.79911,0.29798,-1.09691,
0.53019,0.10730,0.25008,-0.36504,-0.19313,-0.06328,-0.08952,-0.16972,0.48145,-0.37737,-0.38446,0.16267,-0.22488,0.05465,-0.03798,-0.24823,-0.10338,0.33804,-0.49927,-0.11317,-0.23068,
-0.13828,0.10356,0.07988,0.77948,0.42482,0.43114,-0.13169,0.28921,-0.05685,-0.14045,-0.26129,-0.17311,-0.21231,-0.31283,-0.19574,-0.17855,0.25479,-0.56838,-0.23381,0.53942,-0.52793,
-0.17128,-0.11239,-0.41009,0.32895,0.53865,0.50715,-0.37396,-0.53212,-0.46185,-0.08157,0.10814,-0.30519,0.53795,0.59710,-0.79960,-0.26648,0.33584,0.17752,0.11142,-0.04889,0.19303,
0.07290,1.11467,-0.17583,-0.49489,-0.35303,0.08540,-0.18765,-0.06704,0.15736,0.19704,0.54610,1.03212,-0.15435,0.01335,-0.03636,-0.04812,-0.10038,0.15692,-0.00037,0.49505,0.39078,
0.36087,0.21178,0.14457,0.33399,0.72101,-0.04883,-0.36372,0.14593,-0.15149,-0.26028,-0.49599,-0.42294,-0.03392,-0.18543,-0.60972,-0.54334,-0.28250,-0.42509,-0.00070,-0.64805,-0.40641,
-0.03474,0.21613,-0.40313,0.51624,0.06791,-0.22699,-0.30346,-0.56701,0.10795,-0.13421,0.94006,0.28243,0.11726,0.32753,-0.63319,-0.23429,-0.23936,-0.08786,0.03601,0.62110,0.24633,
-0.57728,-0.14857,-0.42044,-0.40290,-0.85461,-0.32080,0.13320,-0.06452,0.21777,0.00644,-0.00913,-0.05219,0.02646,-0.52288,-0.17893,0.33522,0.84939,0.45708,-0.30038,0.09822,-0.10947,
-0.08309,0.40008,-0.26096,0.58208,0.51841,0.10371,0.02740,0.47868,0.13600,-0.17864,0.28905,-0.62492,-0.74147,0.25875,0.34557,0.14904,-0.11050,-0.39835,-0.43813,-0.00876,-0.58660,
-0.10098,0.31263,0.36444,-0.33091,-0.29248,-0.26188,0.01648,0.50999,-0.49304,-0.32965,0.01055,0.35890,0.49684,-0.51841,-0.28938,-0.21152,0.29195,-0.40842,-0.45109,0.14989,0.38708,
-0.20260,0.32338,0.42134,-0.29127,-0.32600,0.47903,0.22395,0.02125,-0.23369,0.20943,0.04388,-0.05387,-0.46557,0.08218,-0.22438,-0.61631,0.04918,0.50909,0.56887,1.00728,-0.11759,
-0.76377,-0.64776,-0.26186,0.06546,-0.22542,0.28859,0.10508,1.33002,0.92050,0.19285,-0.19775,0.53375,0.06523,-0.94939,-0.74619,-0.27735,-0.07829,-0.25751,0.20378,-0.19732,0.41655,
0.74191,-0.36344,0.54138,-0.06195,0.49094,0.26628,-0.34892,-0.45116,0.00611,0.74589,0.16610,-0.95465,-0.86176,-0.38666,-0.59257,-0.01065,0.58494,-0.02429,0.21940,-0.57109,-0.88999,
-0.38769,0.47657,0.43762,0.05668,-0.28980,-0.06054,-0.18359,-0.36869,0.15212,-0.28362,0.69906,0.54317,-0.07816,-0.20697,0.14791,-0.48989,1.49603,0.46784,-0.70518,0.03931,-0.34377,
-0.08674,0.17790,-0.25801,-0.28475,-0.70178,-0.65401,0.20904,0.47844,-0.34380,-0.28529,-0.23870,-0.47951,-0.37041,0.26782,0.76115,-0.08016,0.36309,-0.01470,-0.07739,0.01300,-0.17810,
-0.04597,0.84108,0.74492,0.46899,-0.15954,-0.48859,0.12023,0.38831,0.06813,-0.40695,-0.01059,-0.02345,-0.25453,0.23148,0.14150,-0.29614,-0.05851,-0.46203,-0.29227,0.03309,0.22259,
-0.13693,-0.21289,0.19030,-0.03906,0.23126,0.04098,-0.79357,-0.39622,-0.36161,0.28788,-0.04830,-0.19325,-0.12124,-0.00407,-0.09416,0.09860,0.02588,-0.64806,0.64442,0.75421,-0.10802,
0.14525,-0.38486,-0.21198,0.26053,0.10810,0.44022,-0.36979,0.17061,-0.25849,0.04436,0.35879,0.20888,-0.53746,-0.22920,0.23583,0.12115,0.06540,0.03468,-0.34960,0.05660,-0.18445,
-0.31686,-0.77695,0.08514,-0.44119,0.16104,0.13520,0.29577,0.27841,0.04705,0.04495,-0.00086,-0.21555,-0.03845,-0.15338,-0.32499,0.74004,-0.16847,-0.24072,0.35283,-0.04045,-0.26017,
-0.47980,0.24862,-0.38712,-0.40502,0.68280,0.23474,0.09360,0.70486,0.49215,-0.36863,-0.21635,-0.35603,-0.11728,-0.24545,-0.30150,-0.18847,0.15170,-0.40411,-0.36292,0.11825,-0.12943,
-0.40734,-0.28094,0.15050,0.51427,-0.33699,-0.76861,0.17017,0.50997,0.85298,-0.18406,0.09022,-0.20519,0.12360,0.72480,0.52887,0.22245,-0.03557,-0.17709,-0.32524,-0.19164,-0.86282,
-0.25292,-0.56558,-0.04248,0.54749,0.92804,-0.40735,-0.20088,0.14160,-0.26747,-0.65553,0.28623,0.06037,-0.04006,0.30588,0.06165,0.41561,0.40328,-0.14781,-0.10466,-0.00368,0.51663,
0.33905,0.14218,0.33069,0.49113,-0.07257,-0.17933,-0.04012,-0.32829,0.11348,-0.30462,0.03509,-0.25922,-0.19365,0.01107,-0.40320,-0.28748,-0.01856,0.27953,0.29897,-0.29710,-0.49409,
-0.09430,-0.62879,-0.29188,-0.46962,-0.02520,0.24974,-0.18644,-0.37445,0.12470,0.36707,0.24247,0.65575,-0.17030,-0.16864,0.13143,0.04423,0.23507,0.22169,-0.17696,-0.31665,-0.53384,
-0.18418,0.82296,0.80903,-0.67220,0.06094,0.13125,-0.12607,-0.38617,-0.02166,-0.43941,-0.59259,-0.35545,-0.42358,0.41477,-0.16893,0.04128,-0.21193,0.17052,0.39669,0.20123,-0.25187,
0.36510,0.10365,-0.33862,-0.43878,-0.15912,-0.09998,-0.27581,0.11036,0.70625,0.33743,-0.00310,-0.36885,-0.50881,-0.10242,-0.65167,-0.64904,0.59068,0.45699,0.37098,0.00023,-0.14884,
-0.43445,-0.06099,-0.06413,0.06409,0.18904,0.19348,0.52813,0.08611,-0.43532,-0.21822,-0.22710,-0.56257,-0.10328,0.43008,0.62830,0.00466,0.09322,0.13102,-0.37378,0.05199,-0.50276,
-0.24970,0.29624,-0.03585,0.25489,0.12411,-0.66307,0.05712,-0.00316,-0.53406,0.01351,0.55677,-0.18262,0.44181,-0.05765,-0.26040,-0.20573,0.38988,-0.57074,-0.49387,0.10987,0.33751,
0.46494,0.40142,0.28521,-0.17093,-0.56972,-0.27690,0.22331,0.45643,-0.23319,-0.43029,-0.35523,0.28147,0.27896,0.14983,-0.45535,-0.25576,-0.09066,0.02830,-0.64089,0.04441,-0.15435,
-0.10301,0.09266,0.19509,0.37730,0.07959,0.03922,0.23711,-0.07611,-0.14371,-0.02681,0.03828,0.30748,-0.00173,0.55220,0.14183,-0.57547,-0.19832,-0.23511,0.42847,0.71755,0.24521,
-0.39655,0.01172,-0.79757,-0.31283,0.17100,0.14846,0.39334,0.32774,-0.07120,-0.42632,-0.75900,0.08761,-0.09649,0.20784,0.00211,-0.12905,0.25008,0.03761,0.01852,0.54815,0.06268,
-0.44520,0.07263,-0.35415,0.39619,0.77050,-0.01468,-0.10067,-0.17026,-0.24658,-0.15869,-0.27448,0.23192,0.11569,-0.42080,-0.31464,-0.64597,0.30230,0.10643,0.97906,0.27020,0.03406,
-0.55912,0.23885,-0.47240,-0.48304,0.31107,0.22068,0.30958,0.56082,-0.43244,-0.59352,-0.71178,-0.74482,-0.37110,-0.21469,0.12092,0.29659,0.58481,-0.12757,-0.34503,-0.34068,-0.04496,
-0.12669,0.36314,0.41379,0.44749,0.38775,-0.12343,-0.25434,-0.54061,-0.30669,-0.50339,0.36715,0.70010,0.08021,-0.05663,-0.25338,-0.01978,-0.73923,-0.43208,-0.04463,0.23946,-0.09149,
-0.16196,0.26376,0.07982,-0.09964,0.12717,-0.18250,-0.43877,0.62982,0.20335,-0.04559,-0.23058,-0.79306,-0.59769,-0.36436,-0.00251,1.14088,0.29090,0.40759,0.15085,0.32773,0.19491,
-0.42724,0.17460,-0.15507,0.07054,0.21346,-0.14906,-0.07550,-0.32682,-0.22015,-0.13798,0.06803,-0.21501,-0.33745,0.13336,0.06896,0.13493,-0.34127,-0.27572,-0.13825,0.13793,0.33371,
-0.10077,-0.11846,-0.35145,0.03060,-0.08517,-0.38437,0.43723,0.45853,-0.25097,0.01042,0.61875,-0.12067,0.07390,-0.19610,0.05267,-0.09957,0.52975,0.15887,-0.23155,-0.10555,0.17490,
-0.59368,-0.21349,0.25509,-0.06463,-0.47919,0.22417,-0.30430,0.19233,-0.29993,-0.36721,0.10776,0.17681,0.21769,0.02769,0.72445,0.67161,0.40283,-0.02269,-0.33887,-0.64142,-0.18648,
-0.14858,-0.51192,0.39891,-0.09688,-0.23149,0.17447,-0.65468,0.30641,0.10472,0.30885,0.34882,0.15434,0.04020,0.13549,-0.50743,0.35263,0.13704,-0.09925,0.25507,0.09535,0.64500,
0.02146,0.03155,0.50553,0.23292,0.36914,-0.04666,0.07942,1.14862,0.35596,0.00125,0.02845,0.34742,-0.20842,0.08171,-0.02635,-0.34724,-0.07979,-0.11056,-0.31067,-0.54643,-0.86935,
-0.88864,-0.04881,-0.76366,-0.10395,-0.16907,0.12561,-0.13724,-0.10380,-0.64450,0.51675,0.56844,-0.04720,-0.21308,0.01660,-0.12652,-0.14725,-0.13797,-0.25885,0.11920,-0.21175,0.61585,
-0.07469,0.35883,0.11899,0.57553,-0.08623,-0.10367,-0.61308,-0.64458,-0.14307,0.01529,0.66670,-0.58213,-0.00106,-0.21041,-0.31890,-0.28131,-0.50816,0.45248,-0.04970,0.65505,-0.44063,
0.24632,-0.19921,-0.21680,-0.06347,-0.09131,0.14440,0.43152,-0.37241,1.09662,0.09240,0.44065,-0.25866,-0.02409,-0.42056,-0.41087,0.21781,-0.28061,0.32665,-0.03317,-0.27214,0.07443,
0.04713,0.24032,0.29319,0.64866,-0.19594,-0.62921,-0.10743,0.01663,-0.09304,-0.27977,-0.22801,-0.18752,0.18805,-0.29620,-0.34469,0.32308,-0.14451,-0.53818,