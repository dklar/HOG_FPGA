float Intercept = [-0.24986103];
float classifier[] = {
0.18146,-0.30075,-0.31064,-0.35681,0.24027,-0.15647,-0.16166,0.02850,-0.03693,0.14031,-0.06773,-0.39310,-0.47687,0.36718,-0.02041,-0.23138,0.42145,-0.00561,0.30600,-0.33176,-0.00479,
-0.22410,0.57568,0.02488,-0.05041,0.02458,0.17676,-0.17155,-0.15251,-0.36102,0.04555,0.83629,0.17842,0.03825,-0.00452,-0.01644,-0.07613,0.04932,-0.09259,-0.07880,-0.01831,-0.12345,
-0.12814,-0.56986,0.04279,-0.55763,-0.10102,0.33472,0.25485,0.15390,0.24554,-0.06116,-0.59768,-0.36549,0.24877,0.23673,-0.41667,0.05062,-0.06447,0.05116,-0.31718,-0.21661,-0.04768,
0.26935,0.73370,0.09655,0.09539,-0.59630,0.51707,0.42362,0.17699,0.19826,-0.27191,-0.24697,-0.15963,-0.02015,0.11904,-0.00812,0.48254,0.32999,-0.08862,-0.49013,-0.72618,-0.53019,
-0.35605,0.58373,-0.20564,0.06919,-0.48020,-0.72225,-0.06745,0.11524,0.02902,0.11496,-0.09244,0.28395,0.78539,0.75895,0.40673,-0.28046,-0.18688,-0.67714,-0.33789,0.54191,0.20608,
0.47840,0.23627,0.15706,0.06044,0.07714,-0.31658,-0.33407,0.24336,-0.31530,-0.17774,-0.31712,0.29948,0.46837,-0.02031,-0.08289,-0.22779,0.01574,-0.33980,-0.14073,-0.48720,0.30528,
0.46735,0.07028,0.01638,-0.30367,0.02043,-0.22788,0.02745,-0.02803,0.43307,0.24097,0.01571,0.03747,0.08244,0.29544,0.00993,-0.30546,-0.29201,0.47999,0.49173,-0.08788,-0.22681,
-0.09432,-0.56885,0.28283,0.13000,-0.02063,0.24228,-0.05136,-0.20522,0.30940,0.09851,0.21036,-0.05595,-0.04313,-0.08853,-0.07209,0.01954,-0.21968,0.01467,-0.30813,-0.02245,0.05953,
-0.00938,-0.01386,-0.48406,0.24213,0.26931,0.76588,0.31012,-0.09884,-0.49826,0.00203,0.04550,-0.57425,-0.42220,-0.19342,0.62665,0.89788,0.18275,0.68048,-0.00417,0.04247,-0.08629,
-0.38609,0.62026,0.23320,0.37279,-0.22059,0.17416,0.42403,0.49686,-0.32428,-0.54641,-0.03719,0.09799,0.34879,-0.14771,-0.23214,-0.12130,-0.43928,-0.43912,-0.54678,-0.05278,0.46681,
0.14720,-0.56898,-0.28281,0.02903,-0.33999,-0.69965,-0.20975,0.11373,0.65121,0.60553,-0.11596,0.46315,0.72246,0.52278,0.55241,-0.23410,-0.17107,0.12980,-0.35799,-0.01789,0.61444,
1.11987,-0.25340,-0.50181,-0.25020,-0.19158,0.05078,-0.32014,-0.75998,-0.07321,0.52498,0.10899,-0.32362,-1.01384,-0.57881,-0.47230,-0.15638,-0.55436,0.00703,0.79132,0.20732,-0.87925,
0.05002,0.29755,0.05297,-0.49125,0.13115,-0.22080,0.26694,-0.14799,0.22555,0.06306,0.10402,0.12177,-0.02404,0.07064,-0.04308,-0.45957,-0.39955,0.09363,-0.40836,-0.16243,-0.09740,
0.02411,-0.21748,-0.15007,1.22465,0.49533,0.20820,-0.11199,-0.09954,-0.29103,0.03147,-0.14216,-0.17401,0.12725,-0.12739,-0.07002,-0.17008,0.16890,-0.15440,-0.25203,0.44224,-0.41268,
-0.02042,-0.13386,-0.48833,0.18092,0.77053,0.57976,-0.02160,-0.53006,-0.35855,-0.28093,-0.33842,-0.21971,0.12041,0.56507,-0.33285,-0.10409,-0.13703,0.15590,0.15235,-0.11326,0.08473,
-0.08957,1.03254,0.12187,-0.33753,-0.41735,0.29151,-0.04804,-0.10855,0.15272,0.36843,0.44345,0.63555,-0.13387,-0.10902,0.01302,-0.00824,-0.03308,0.27833,-0.40869,0.49651,0.29837,
0.06678,0.12991,-0.20167,0.30712,0.39313,-0.52110,-0.18732,0.20301,0.01862,-0.11578,-0.43160,-0.32530,-0.07685,-0.10284,-0.40260,-0.36976,0.06009,-0.00329,-0.37293,-0.40181,-0.13766,
0.05997,0.23411,0.08713,0.10818,0.14681,0.19362,-0.14906,-0.67542,-0.19081,0.22107,0.37743,-0.44157,0.28276,0.56174,-0.11702,-0.33536,-0.27653,0.07033,0.04596,0.40999,0.06217,
-0.43223,0.08468,0.17706,-0.46613,-0.55314,-0.25003,0.26531,-0.19524,-0.67830,-0.21649,0.17949,0.15191,-0.19085,-0.34038,-0.09755,0.55755,0.98474,0.47546,0.06318,-0.07853,-0.25014,
-0.09723,0.04327,0.27920,0.29005,0.41446,0.13321,0.04592,-0.18631,0.05496,-0.26228,0.25844,-0.15468,0.24072,0.38468,-0.24783,0.19043,0.00175,-0.07422,-0.09043,0.00430,-0.37620,
0.03271,-0.11969,0.16264,-0.31815,-0.00159,-0.26099,-0.25208,0.33156,-0.20323,-0.05903,-0.13116,-0.01754,0.07230,-0.38240,-0.23104,0.03022,0.31769,-0.15918,0.16653,0.03772,0.21436,
-0.60444,0.24031,0.67637,-0.19085,0.05012,0.20928,-0.22088,-0.29525,0.37763,0.59020,0.28096,0.05190,-0.39322,-0.11972,-0.21119,-0.13603,0.00200,0.00823,-0.00756,0.73891,0.17083,
-0.35483,-0.41426,-0.36337,-0.04557,-0.45483,0.09921,-0.02124,0.52843,0.79501,0.32447,0.16489,0.21721,0.15217,-0.62027,-0.48209,-0.44164,-0.00236,0.10678,-0.15047,-0.05463,0.30979,
0.43223,0.17820,0.07737,-0.10347,0.52778,0.46573,0.01003,-0.14490,0.00799,0.45322,-0.43716,-0.73940,-0.65833,-0.17312,-0.29519,0.11042,0.13251,-0.15767,0.24095,-0.22392,-0.77919,
-0.22962,0.64718,0.33707,-0.21448,-0.55684,-0.32812,0.10090,-0.14751,0.14663,0.16208,0.56973,0.06786,-0.00074,-0.25930,0.18397,0.07911,0.41682,0.36428,-0.75274,-0.14078,-0.17174,
-0.44437,-0.07901,0.16290,-0.24345,-0.34929,-0.33698,0.11954,0.42538,0.04976,-0.14047,0.17419,-0.26204,-0.10430,0.26469,0.23981,-0.52338,0.17645,-0.18697,0.06444,-0.15862,0.06053,
0.23911,0.33974,0.20588,0.09738,-0.25123,-0.26872,-0.08243,0.32643,0.04512,0.36797,0.05329,0.08587,-0.41543,-0.04252,0.03779,-0.34132,-0.00035,-0.78047,0.23275,0.54387,-0.25859,
0.14609,-0.07509,0.10506,-0.29395,0.25397,0.11308,0.11807,0.01383,-0.19751,0.14565,0.22760,0.12152,0.04494,0.05181,0.14736,0.17785,0.11042,-0.23106,-0.10536,0.57059,0.32225,
-0.02764,-0.32515,0.04126,0.32137,0.18957,-0.12952,-0.19573,0.19924,-0.25652,0.15693,0.33704,0.03503,-0.39201,-0.16114,-0.21006,0.14916,-0.41671,-0.05558,-0.06143,-0.05322,-0.17913,
-0.38019,-0.43091,0.01190,-0.42327,0.24783,-0.01751,0.21868,0.39022,0.30976,0.34196,-0.44010,-0.27991,-0.28598,-0.09090,-0.22895,0.18682,0.07966,0.07327,0.04071,-0.09721,-0.80760,
0.07083,0.24781,-0.04392,-0.51043,0.54971,0.15969,0.46521,0.92379,0.15475,-0.29795,-0.00322,-0.22999,-0.11803,0.16420,-0.22404,-0.42265,-0.05875,-0.29435,-0.01587,0.21126,0.38584,
-0.09656,-0.44168,0.02302,0.12681,0.07228,-0.99590,-0.20758,0.33297,0.50776,0.05495,-0.10499,-0.19281,-0.13870,0.52407,-0.24128,-0.05997,0.14289,0.08857,-0.05656,-0.27820,-0.28378,
-0.38162,-0.23264,-0.47645,0.26181,0.60234,0.19053,-0.14693,0.20872,-0.11889,-0.00830,0.14910,0.34594,0.41847,0.34197,-0.20212,0.10584,0.11112,-0.07178,-0.06280,0.05824,0.05836,
-0.05574,0.13765,0.37218,0.04107,0.08723,0.19375,-0.07760,0.20366,0.11193,-0.01962,-0.28044,-0.04948,-0.16508,0.00842,-0.38429,-0.10006,-0.21004,-0.20162,-0.04666,-0.02838,-0.32702,
-0.20255,0.27563,-0.01194,-0.16178,0.03494,-0.15132,0.31535,-0.26595,0.09665,0.26332,0.38376,0.39140,-0.02647,0.04672,-0.12355,0.40924,0.24320,0.07605,0.27986,0.07463,-0.12154,
-0.56837,0.23574,0.13593,-0.27451,-0.30136,-0.11863,-0.24240,-0.14242,-0.25308,-0.23759,-0.09111,0.03582,-0.42685,0.00937,-0.08068,0.12139,0.03975,0.02436,0.11641,-0.32098,0.04670,
-0.05826,-0.32156,-0.41728,-0.26982,-0.11692,-0.16564,-0.18352,0.28868,0.30705,0.47137,0.42661,-0.40106,-0.23583,-0.02688,-0.28543,-0.46980,-0.06829,-0.15558,0.18805,-0.13837,0.03348,
-0.32774,0.10718,-0.04295,-0.05355,0.35023,0.10164,0.12163,0.45214,0.18238,0.02089,-0.01247,-0.03139,0.05934,0.19045,0.23136,0.20477,-0.04904,-0.07435,0.25896,0.04383,-0.36636,
-0.14940,-0.11221,0.20708,0.53911,-0.07170,-0.68683,-0.13608,-0.14598,-0.29837,-0.21495,0.23225,-0.25276,0.21721,-0.10944,-0.05464,-0.10615,0.07230,-0.29515,0.14290,-0.11931,0.02303,
-0.01152,0.42131,-0.23262,0.08668,-0.19962,-0.16713,0.50190,0.55420,0.09823,-0.42390,-0.43575,0.24857,0.21589,0.22034,-0.26835,-0.09734,-0.03310,0.04481,-0.46915,-0.20279,-0.09068,
-0.07500,0.47655,0.01067,-0.12302,0.15338,0.05106,-0.04837,0.00939,0.04135,0.14152,0.31900,-0.08073,0.13237,0.15260,0.05302,-0.59704,-0.21561,-0.24305,0.62079,0.46168,0.28722,
-0.21971,0.13415,-0.40063,-0.13301,-0.10856,0.01487,0.45295,0.14977,-0.14758,0.00703,-0.09537,-0.01497,0.06037,-0.29969,-0.03828,-0.07095,-0.20800,0.08615,-0.29417,0.23464,0.23171,
-0.19824,-0.00328,0.25887,0.63262,0.09134,0.05531,0.25304,-0.09427,-0.50625,-0.37040,-0.13052,0.35383,0.06994,0.10166,-0.52558,-0.31977,-0.04983,0.30473,0.77452,0.21251,-0.04667,
0.01977,-0.11728,-0.44736,-0.84260,-0.16562,0.27667,0.15803,0.22845,0.01055,-0.12091,-0.38984,-0.34356,-0.32638,0.12775,-0.21436,0.51111,0.51466,0.05187,0.01272,-0.34269,-0.29966,
-0.15640,-0.08355,0.35034,0.75819,0.20176,0.17959,-0.10543,-0.15015,-0.34014,-0.60138,-0.01353,0.46915,-0.35344,-0.49283,-0.88200,-0.22089,-0.14931,0.05513,-0.12528,-0.02005,-0.00197,
0.34349,0.25716,-0.31670,-0.17817,0.00163,-0.14937,-0.30641,0.44287,0.16587,0.08514,-0.14265,-0.38644,-0.25696,-0.33287,-0.03517,0.73710,0.20864,0.35149,0.41385,0.47888,0.27951,
-0.22151,-0.32550,-0.07680,0.15738,0.56747,0.17886,-0.38732,-0.12904,-0.40511,-0.28772,0.22697,0.08719,-0.35390,0.01237,-0.15541,-0.24320,-0.10179,-0.16016,-0.06707,0.25703,0.25498,
0.00388,-0.14985,-0.14521,-0.32479,-0.22048,-0.10264,0.20256,-0.26404,-0.12691,0.07473,0.04732,-0.11084,0.12596,-0.03597,0.28913,0.43249,0.65419,0.26115,0.28560,0.07988,0.22614,
-0.30673,0.00258,0.04450,-0.03506,-0.01576,-0.01014,-0.16499,-0.00456,-0.07007,-0.19417,0.11427,-0.05965,0.36565,-0.33213,0.61054,0.61055,0.44499,0.10551,-0.75334,-0.23226,0.09528,
0.08873,-0.60117,0.21408,0.02092,-0.01395,-0.09771,-0.32895,-0.04745,-0.08969,0.39205,-0.01100,0.31901,-0.13137,-0.10840,-0.06996,0.03765,-0.09875,0.13325,0.51552,-0.28321,0.45362,
0.08187,0.22772,0.79658,-0.02270,0.16945,0.12163,0.20712,0.48426,0.49142,-0.16586,0.08528,0.32377,0.12050,0.02846,0.07207,-0.07010,-0.19688,-0.00034,0.00243,-0.26926,-0.09715,
-0.53553,-0.46501,-0.54449,-0.11206,-0.50959,-0.21818,-0.39796,-0.28757,-0.32782,-0.07456,-0.16605,-0.23675,0.21881,0.10976,0.02865,0.30966,-0.36570,-0.02677,0.30713,-0.20596,0.38972,
0.24494,-0.61210,-0.02473,0.43074,0.00119,0.26455,0.15884,-0.23566,-0.43395,-0.24858,0.25342,-0.07549,-0.36155,-0.08399,0.13014,0.12313,-0.23887,-0.13128,0.13637,0.24762,-0.46518,
0.35396,-0.07952,0.10820,-0.46500,-0.37018,0.19727,0.26896,0.07120,0.72927,0.40965,0.05639,-0.36369,-0.18466,0.02028,-0.33278,-0.03366,0.25856,0.66306,-0.37125,-0.12307,-0.04712,
0.15309,-0.08803,-0.26639,0.57327,-0.09818,-0.33053,-0.24778,-0.12447,0.18031,-0.64268,-0.23563,0.06753,0.26243,0.42146,0.00423,0.23397,-0.32654,-0.16916,