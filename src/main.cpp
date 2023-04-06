
#include <iostream>
#include <vector>

#define ALGEBRA_DEFAULT_NAMES
#include "algbera.h"

using namespace Algebra;

int main()
{
    std::vector<std::vector<Vec2>> points = {
        {{1447.6632, 652.4185}, {1855.779, 653.6937}, {2260.2432, 653.9705}, {2661.6667, 651.54346}, {3064.6223, 646.02716}, {3471.6267, 639.5849}, {3881.254, 632.47833}, {1464.3779, 1062.6641}, {1868.8302, 1063.05}, {2267.5527, 1061.9054}, {2663.9385, 1059.0275}, {3060.6692, 1055.5146}, {3463.0896, 1049.4722}, {3869.5818, 1043.6792}, {1479.5461, 1462.8992}, {1879.4222, 1460.6489}, {2273.8083, 1458.2185}, {2666.064, 1455.7513}, {3058.276, 1452.1133}, {3456.0764, 1449.0414}, {3859.107, 1445.2458}, {1490.8859, 1855.3342}, {1888.0289, 1850.9668}, {2279.3936, 1847.3595}, {2668.0698, 1844.0776}, {3057.708, 1842.0598}, {3452.0283, 1840.2754}, {3851.1074, 1839.253}, {1500.0402, 2243.298}, {1894.5195, 2236.4607}, {2283.9185, 2231.2527}, {2670.608, 2227.747}, {3058.3245, 2226.5789}, {3449.8108, 2226.9683}, {3845.4277, 2227.5166}},
        {{517.5, 935.0}, {952.3259, 960.80835}, {1353.4779, 988.25745}, {1720.868, 1013.289}, {2056.7158, 1036.1849}, {2364.4983, 1056.2407}, {2649.7966, 1073.4001}, {519.0, 1339.0}, {951.9582, 1345.7369}, {1353.3053, 1357.2115}, {1719.8691, 1365.8746}, {2053.3228, 1373.4413}, {2359.5503, 1379.8296}, {2642.948, 1385.9207}, {520.8562, 1736.2976}, {953.15674, 1729.3018}, {1351.6271, 1721.7445}, {1716.6978, 1713.2767}, {2048.6711, 1706.4999}, {2353.8354, 1700.0667}, {2636.5544, 1694.7336}, {522.5, 2134.5}, {950.0, 2109.5}, {1347.4999, 2083.166}, {1711.517, 2059.093}, {2043.4258, 2037.552}, {2347.6533, 2018.3014}, {2629.5972, 2001.8385}, {525.50006, 2525.9368}, {949.1145, 2486.271}, {1343.0144, 2444.0735}, {1705.2065, 2404.4453}, {2036.6317, 2368.421}, {2341.5706, 2336.5312}, {2623.3325, 2308.1328}},
        {{2148.0916, 879.2464}, {2476.483, 860.26807}, {2824.0078, 837.79663}, {3196.0, 813.0}, {3594.5, 784.5}, {4023.0, 754.0}, {4480.0, 724.0}, {2162.8083, 1234.5663}, {2488.2415, 1224.3367}, {2831.8015, 1213.9867}, {3199.5833, 1201.5311}, {3592.002, 1188.2374}, {4020.0, 1171.5}, {4472.9927, 1157.5973}, {2176.5352, 1580.5315}, {2499.8037, 1580.784}, {2841.0986, 1580.2937}, {3204.5566, 1580.6044}, {3594.8335, 1580.3271}, {4018.4553, 1580.2104}, {4469.4556, 1580.5819}, {2189.24, 1923.2104}, {2510.081, 1931.8861}, {2850.3513, 1942.4669}, {3211.447, 1953.6062}, {3600.046, 1968.098}, {4019.588, 1982.677}, {4466.475, 1998.2012}, {2200.541, 2263.3433}, {2520.7988, 2281.3823}, {2859.5999, 2301.3213}, {3220.7454, 2325.2927}, {3607.4387, 2351.7654}, {4023.3381, 2380.8245}, {4463.568, 2407.9307}},
        {{2267.3486, 501.52032}, {2601.579, 512.0435}, {2949.8335, 519.6726}, {3314.0754, 526.3494}, {3696.3484, 532.50415}, {4094.984, 541.01184}, {4507.657, 552.4436}, {2278.767, 848.9458}, {2602.008, 864.076}, {2936.6255, 879.26215}, {3287.4907, 893.0994}, {3656.5146, 907.1811}, {4043.4426, 921.5486}, {4444.1743, 939.41534}, {2289.0361, 1170.9359}, {2601.5605, 1191.7053}, {2925.8076, 1211.945}, {3263.7056, 1233.4333}, {3620.1577, 1254.4513}, {3994.4976, 1277.2743}, {4383.3467, 1301.2339}, {2298.346, 1473.4574}, {2601.2136, 1497.6573}, {2915.7317, 1524.3895}, {3243.7744, 1550.9318}, {3587.7356, 1579.4844}, {3950.2, 1609.2067}, {4326.876, 1639.8906}, {2306.5862, 1758.3337}, {2601.558, 1787.6313}, {2906.9106, 1817.2616}, {3225.3342, 1850.2877}, {3559.5242, 1883.7618}, {3909.9592, 1920.2585}, {4273.6963, 1957.3223}},
        {{653.0, 627.5}, {1071.0, 639.5}, {1462.959, 654.88477}, {1823.9983, 669.3163}, {2159.543, 681.5778}, {2469.6497, 691.9491}, {2760.272, 699.5764}, {682.5, 1012.5}, {1087.5, 1010.5}, {1467.2025, 1011.54987}, {1817.5365, 1012.1715}, {2140.9727, 1012.4046}, {2441.9387, 1011.5769}, {2723.6726, 1009.71967}, {706.6491, 1369.8121}, {1104.0856, 1355.6582}, {1470.7479, 1344.1996}, {1809.63, 1332.222}, {2123.2417, 1321.0902}, {2415.498, 1310.1544}, {2689.216, 1299.1498}, {732.5795, 1707.96}, {1116.1271, 1680.0656}, {1471.4089, 1654.6464}, {1801.0315, 1632.0446}, {2106.1099, 1610.6213}, {2390.1226, 1590.5137}, {2657.5, 1573.0}, {756.5, 2020.5}, {1123.0356, 1984.2482}, {1470.9807, 1948.3533}, {1791.4796, 1914.9584}, {2088.5078, 1884.195}, {2366.8066, 1856.6226}, {2627.1553, 1830.9043}},
        {{1643.3267, 610.52246}, {1992.3859, 615.26935}, {2339.1716, 619.42114}, {2683.7305, 621.1982}, {3028.9492, 620.6974}, {3376.4438, 619.6484}, {3726.5354, 616.9469}, {1612.501, 910.454}, {1973.4331, 916.67346}, {2330.0603, 919.7521}, {2684.8098, 921.2707}, {3039.7397, 921.9379}, {3398.4424, 920.54584}, {3760.2578, 918.63525}, {1578.7035, 1230.4514}, {1952.0189, 1234.0334}, {2320.4312, 1237.2756}, {2685.7651, 1239.0144}, {3052.1863, 1238.9408}, {3422.0664, 1239.5789}, {3796.8389, 1238.9928}, {1540.5781, 1571.4427}, {1927.7131, 1573.7181}, {2308.9219, 1575.0308}, {2686.7483, 1576.4598}, {3065.563, 1578.7148}, {3448.9846, 1579.5826}, {3837.3362, 1581.6082}, {1497.6715, 1938.5983}, {1900.0507, 1937.8732}, {2295.701, 1938.2294}, {2689.264, 1939.3522}, {3082.0818, 1942.0643}, {3480.2048, 1945.6782}, {3882.9414, 1949.2864}},
        {{1492.9849, 893.0098}, {1863.8544, 901.1292}, {2228.626, 907.6775}, {2586.5596, 913.1616}, {2942.8213, 916.0796}, {3298.3352, 918.71027}, {3655.5137, 919.3756}, {1523.8474, 1243.7654}, {1882.6993, 1248.5027}, {2233.7527, 1252.2767}, {2579.914, 1255.0486}, {2923.6096, 1258.1886}, {3267.3096, 1259.865}, {3612.3748, 1261.6499}, {1550.6909, 1569.226}, {1898.1613, 1572.2391}, {2237.9841, 1573.9983}, {2573.2478, 1575.654}, {2906.4307, 1577.0311}, {3238.8105, 1578.9474}, {3573.1167, 1580.7388}, {1574.7703, 1878.1948}, {1911.0554, 1876.7805}, {2241.4226, 1876.286}, {2566.6663, 1876.5955}, {2890.228, 1878.1984}, {3213.2688, 1879.385}, {3537.914, 1881.6525}, {1595.4395, 2169.32}, {1922.4526, 2165.586}, {2243.93, 2163.2192}, {2561.1958, 2161.708}, {2875.8875, 2162.284}, {3190.3853, 2163.9648}, {3505.8691, 2165.9001}},
        {{2592.8984, 926.7849}, {2827.941, 896.0024}, {3085.5872, 860.27826}, {3370.4136, 821.29504}, {3687.6492, 776.51953}, {4040.188, 728.4584}, {4430.459, 676.78894}, {2595.6487, 1210.311}, {2829.4763, 1191.5374}, {3086.1338, 1171.5045}, {3369.7573, 1147.5476}, {3686.5945, 1121.4299}, {4039.8987, 1091.9359}, {4432.3086, 1060.7366}, {2598.1433, 1490.6445}, {2831.5498, 1485.6505}, {3088.5, 1478.7373}, {3369.9297, 1471.7289}, {3686.6338, 1462.7498}, {4040.7573, 1454.1433}, {4434.4814, 1444.3029}, {2600.4563, 1771.3242}, {2835.0696, 1777.9448}, {3090.4343, 1785.7148}, {3374.6592, 1794.4032}, {3690.0999, 1803.409}, {4042.5308, 1814.5946}, {4437.2666, 1826.1063}, {2603.543, 2051.3523}, {2837.5496, 2070.673}, {3094.0825, 2092.2354}, {3377.8008, 2117.5056}, {3694.617, 2144.0261}, {4048.2905, 2176.2817}, {4440.4297, 2208.0364}},
        {{391.72955, 772.5915}, {692.3987, 760.1828}, {1005.9653, 746.49384}, {1332.0732, 734.3265}, {1668.3597, 723.9059}, {2013.0369, 712.7255}, {2366.4822, 699.7475}, {391.3061, 1112.4678}, {692.9277, 1104.0728}, {1007.4017, 1096.3835}, {1335.3064, 1090.2109}, {1672.1143, 1084.4639}, {2017.079, 1077.8525}, {2368.1855, 1070.5308}, {391.6167, 1451.3008}, {693.2751, 1448.5477}, {1008.94165, 1446.818}, {1337.6116, 1444.1857}, {1675.5042, 1441.8905}, {2018.6368, 1440.1841}, {2369.7502, 1437.1976}, {390.26508, 1790.8647}, {693.6477, 1793.4462}, {1009.97375, 1796.2478}, {1338.4789, 1797.5643}, {1676.4084, 1798.9711}, {2020.3525, 1800.6768}, {2370.8174, 1803.448}, {391.77618, 2129.5776}, {694.48334, 2137.9993}, {1009.4914, 2145.7769}, {1337.2283, 2152.5105}, {1674.4664, 2157.6436}, {2019.9639, 2163.555}, {2371.5352, 2170.4617}},
        {{672.24866, 672.0428}, {1076.0, 710.5}, {1439.5818, 747.56494}, {1760.6849, 781.92096}, {2051.974, 811.8042}, {2312.012, 837.9621}, {2549.3179, 860.83514}, {679.5, 1068.5}, {1081.0, 1086.5}, {1441.0, 1101.0}, {1763.8546, 1116.6719}, {2050.6497, 1129.126}, {2309.1606, 1140.0269}, {2543.7642, 1149.1569}, {686.2511, 1461.2339}, {1085.5137, 1454.9293}, {1445.3401, 1450.0736}, {1763.9265, 1444.2859}, {2048.3647, 1440.4735}, {2304.5273, 1436.5563}, {2538.5056, 1432.5309}, {693.0, 1847.5}, {1088.4738, 1818.8412}, {1444.11, 1792.41}, {1761.4308, 1768.6797}, {2045.4962, 1747.5656}, {2300.4795, 1729.0266}, {2532.9514, 1713.0629}, {697.5, 2229.0}, {1090.1661, 2177.7227}, {1443.4012, 2132.515}, {1758.7885, 2089.4075}, {2040.8118, 2052.4807}, {2295.687, 2020.0275}, {2527.5232, 1990.4308}},
        {{2521.061, 757.37744}, {2882.1216, 770.4984}, {3232.0564, 779.9663}, {3570.3523, 789.40875}, {3899.498, 796.94446}, {4215.631, 807.2652}, {4514.6777, 817.84326}, {2519.1873, 1139.6821}, {2877.9827, 1143.0249}, {3223.5386, 1147.4141}, {3560.1677, 1149.0765}, {3885.1257, 1151.7323}, {4201.7007, 1154.8739}, {4502.5664, 1156.8029}, {2517.961, 1511.4163}, {2873.5142, 1509.5099}, {3217.267, 1505.1577}, {3550.587, 1503.3713}, {3875.2764, 1499.943}, {4189.843, 1497.1439}, {4489.731, 1495.829}, {2517.4944, 1881.1243}, {2870.6897, 1869.058}, {3212.4607, 1860.6355}, {3544.3975, 1851.8429}, {3867.4033, 1843.5945}, {4178.4824, 1836.1581}, {4478.233, 1828.4125}, {2515.9185, 2244.216}, {2868.634, 2227.8796}, {3209.1006, 2210.632}, {3539.4995, 2198.5515}, {3859.639, 2184.3623}, {4169.738, 2170.7007}, {4466.3027, 2156.2847}},
    };

    int size = 1, layout_x = 7, layout_y = 5;

    std::vector<Vec2> world_points;
    world_points.reserve(layout_x * layout_y);

    for (double i = 0; i < layout_y; i++)
        for (double j = 0; j < layout_x; j++)
            world_points.emplace_back(Vec2{j * size, i * size });
}