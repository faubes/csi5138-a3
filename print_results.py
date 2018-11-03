results_rnn = [
[50, 20, 32, 4432708, 0.6376804670715333, 0.643160000038147, 0.6270833188056946, 0.66212, 249.60792541503906],
[50, 50, 32, 4437298, 0.6171711052322387, 0.6618400000190735, 0.6811975311279297, 0.61252, 360.634868144989],
[50, 100, 32, 4448948, 0.6219186178207398, 0.6533200000381469, 0.6363481485748291, 0.62264, 615.509791135788],
[50, 200, 32, 4487248, 0.6079819486999511, 0.658879999961853, 0.5973224577713012, 0.67672, 1110.7866868972778],
[50, 500, 32, 4722148, 0.6931873051452637, 0.4968, 0.6931473014450074, 0.5, 3458.7809681892395],
[50, 20, 32, 4432708, 0.6606566069984436, 0.5970000001907348, 0.6544740601921082, 0.60184, 209.78257942199707],
[50, 50, 32, 4437298, 0.6329588753509522, 0.6450800001716613, 0.6286846777534485, 0.64664, 318.57941579818726],
[50, 100, 32, 4448948, 0.626706328086853, 0.6443599999809265, 0.6517039742469788, 0.61884, 562.7595283985138],
[50, 200, 32, 4487248, 0.6222157908630371, 0.6452399998855591, 0.6249059831047058, 0.64444, 1023.2206013202667],
[50, 500, 32, 4722148, 0.7037422802734375, 0.5334800001335144, 0.6821173525619507, 0.55776, 3257.578360557556],
[50, 20, 32, 4432708, 0.6172188492012024, 0.6644, 0.6112565525054932, 0.68132, 428.7461807727814],
[50, 50, 32, 4437298, 0.5930276132202148, 0.679119999961853, 0.5828856637573242, 0.70236, 621.2842557430267],
[50, 100, 32, 4448948, 0.5826911168289185, 0.6848800000190735, 0.5933006447410584, 0.6706, 1053.0123896598816],
[50, 200, 32, 4487248, 0.5720118319511414, 0.691680000038147, 0.6346003621482849, 0.6212, 1970.6013939380646],
[50, 500, 32, 4722148, 0.6931803389167785, 0.4983199999809265, 0.6931474530792237, 0.5, 6293.416866540909],
[50, 20, 32, 4432708, 0.6078237443733215, 0.6732400000762939, 0.5906654968070983, 0.69964, 347.2185971736908],
[50, 50, 32, 4437298, 0.595384086151123, 0.6835600001525879, 0.6174233937263489, 0.67136, 530.9194002151489],
[50, 100, 32, 4448948, 0.6165049298667907, 0.6558800000953674, 0.6364465594673157, 0.62336, 949.3022027015686],
[50, 200, 32, 4487248, 0.5673750203132629, 0.70012, 0.6097954645729065, 0.66496, 1810.306406736374],
[50, 500, 32, 4722148, 0.6931765112876892, 0.49704000007629395, 0.6931499082946777, 0.50004, 5876.735167503357],
[50, 20, 32, 4432708, 0.5847420871162414, 0.6939200000190735, 0.6313716492462158, 0.61832, 655.5884897708893],
[50, 50, 32, 4437298, 0.5760439958000183, 0.6929600000190735, 0.5980267438316346, 0.66228, 936.9619038105011],
[50, 100, 32, 4448948, 0.5585314970970153, 0.7087999999618531, 0.588857556552887, 0.68064, 1643.6594483852386],
[50, 200, 32, 4487248, 0.5340089868927002, 0.723439999961853, 0.6241217101669312, 0.66868, 3121.122697353363],
[50, 500, 32, 4722148, 0.6935476118850707, 0.49783999998092654, 0.6936946862983704, 0.5, 10129.571161031723],
[50, 20, 32, 4432708, 0.6150188718795776, 0.6628399999809265, 0.6381556827354431, 0.64404, 539.9765031337738],
[50, 50, 32, 4437298, 0.5909323853111267, 0.688840000076294, 0.6093732368850708, 0.66308, 822.2102127075195],
[50, 100, 32, 4448948, 0.5593225076675415, 0.7099999999427795, 0.5677569313621521, 0.7086, 1489.2499856948853],
[50, 200, 32, 4487248, 0.5196418633460999, 0.7345599999046326, 0.6190253990554809, 0.6676, 2878.357229232788],
[50, 500, 32, 4722148, 0.6851869830703735, 0.566599999923706, 0.6735065095901489, 0.58284, 9603.225801706314]]

NUM_EPOCHS = [3, 6, 10]
BATCH_SIZE = [128, 256]
STATE_DIM = [20, 50, 100, 200, 500]

print('===================RNN=============')
epoch_count = -1
batch_size = 128
for i, item in enumerate(results_rnn):
    if i % 10 == 0:
        epoch_count = epoch_count + 1
    if i % 10 <= 5:
        batch_size = 128
    else:
        batch_size = 256
    print("{} & {} & {} & {} & {:0.4f} & {:0.4f} & {:0.4f} & {:0.4f} & {:0.1f} \\\\".format(item[1],
    batch_size, NUM_EPOCHS[epoch_count], item[3], item[4], item[5], item[6], item[7], item[8]))


print('============LSTM=========')
results_lstm = [
[100, 20, 64, 8869654, 0.6191608553218841, 0.655800000038147, 0.8289668787002563, 0.58472, 798.7499170303345],
[100, 50, 64, 8892094, 0.5870043593788147, 0.682600000038147, 0.473286950340271, 0.77688, 802.1751718521118],
[100, 100, 64, 8945494, 0.6070517388343811, 0.666799999961853, 0.49610917198181154, 0.77296, 806.5077629089355],
[100, 200, 64, 9112294, 0.619145244216919, 0.657, 0.4726732300281525, 0.77492, 813.4629828929901],
[100, 20, 64, 8869654, 0.623666639251709, 0.654800000038147, 0.5537520566749573, 0.72052, 578.2075669765472],
[100, 50, 64, 8892094, 0.6315702415847778, 0.6414000000190735, 0.9549839791297913, 0.50356, 589.7982671260834],
[100, 100, 64, 8945494, 0.6372687135314942, 0.633359999961853, 0.5486601175689697, 0.73252, 595.4609899520874],
[100, 200, 64, 9112294, 0.6517064552116394, 0.6222399999809265, 0.5976166742706299, 0.6752, 597.8804500102997],
[100, 20, 64, 8869654, 0.6513225451278687, 0.61284, 0.5756455237197876, 0.71384, 498.7652509212494],
[100, 50, 64, 8892094, 0.650560165309906, 0.6180400000190734, 0.6040221597099305, 0.68164, 506.0256769657135],
[100, 100, 64, 8945494, 0.6599629478263855, 0.6046000000953674, 0.5901470097541809, 0.69544, 507.25475311279297],
[100, 200, 64, 9112294, 0.6681201704788208, 0.598719999961853, 0.5610254308891296, 0.7182, 509.6307210922241],
[100, 20, 64, 8869654, 0.5193104151344299, 0.745760000038147, 0.4804352688026428, 0.7666, 1099.147863149643],
[100, 50, 64, 8892094, 0.4329339340209961, 0.801759999961853, 0.4261534342288971, 0.80776, 1125.4672000408173],
[100, 100, 64, 8945494, 0.4715856994533539, 0.78064, 0.40420226961135863, 0.8178, 1146.4571330547333],
[100, 200, 64, 9112294, 0.42837264387130736, 0.8039600000190735, 0.38140702835083007, 0.83216, 1147.1901280879974],
[100, 20, 64, 8869654, 0.511504328289032, 0.7545199999809266, 0.4500410025691986, 0.79304, 693.3546130657196],
[100, 50, 64, 8892094, 0.5260707847595215, 0.742839999961853, 0.5670076033210755, 0.71868, 717.5651621818542],
[100, 100, 64, 8945494, 0.5272998892116547, 0.7426000000190734, 0.4495073623275757, 0.79276, 678.8292741775513],
[100, 200, 64, 9112294, 0.5342636950874329, 0.7387199999809265, 0.5073842482566834, 0.7414, 712.8684568405151],
[100, 20, 64, 8869654, 0.5828859311294555, 0.701640000038147, 0.6261196500015259, 0.65, 549.0766999721527],
[100, 50, 64, 8892094, 0.5862616704368592, 0.69448, 0.5394505178833008, 0.73328, 551.2523529529572],
[100, 100, 64, 8945494, 0.5846680870628357, 0.6980000001525879, 0.6367003026771545, 0.65224, 549.9430391788483],
[100, 200, 64, 9112294, 0.6113826164627075, 0.6738800000953674, 0.6161589710617066, 0.66432, 553.7131018638611],
[100, 20, 64, 8869654, 0.4025820939064026, 0.821400000038147, 0.46643792744636536, 0.7816, 1443.2095820903778],
[100, 50, 64, 8892094, 0.3884964138221741, 0.8261599999618531, 0.3715617958831787, 0.83376, 1492.47123503685],
[100, 100, 64, 8945494, 0.3906030861186981, 0.825519999961853, 0.3959442036151886, 0.83624, 1556.698194026947],
[100, 200, 64, 9112294, 0.3710170631122589, 0.834839999961853, 0.34527984260082245, 0.84624, 1598.0276091098785],
[100, 20, 64, 8869654, 0.48165718994140627, 0.7718, 0.4391372120380402, 0.79788, 890.0541830062866],
[100, 50, 64, 8892094, 0.45761226943016053, 0.788599999961853, 0.41104258297920226, 0.81288, 924.6841051578522],
[100, 100, 64, 8945494, 0.43913166749954224, 0.7930400000190735, 0.39526512179374695, 0.82188, 933.8808209896088],
[100, 200, 64, 9112294, 0.47569110125541686, 0.773480000038147, 0.44111298584938047, 0.79332, 936.5242688655853],
[100, 20, 64, 8869654, 0.5107581512928009, 0.7552400001716614, 0.47149983547210694, 0.7844, 645.635046005249],
[100, 50, 64, 8892094, 0.5102337877655029, 0.7523600000190734, 0.5132112271976471, 0.7446, 664.1746110916138],
[100, 100, 64, 8945494, 0.5387547321891785, 0.7307999999046325, 0.5355081140995026, 0.72268, 658.5263390541077],
[100, 200, 64, 9112294, 0.5646911519622803, 0.713439999961853, 0.5495663492584228, 0.73832, 672.7692630290985],
[100, 20, 64, 8869654, 0.3828129069328308, 0.8309199999618531, 0.36056628391742707, 0.84032, 1881.4364349842072],
[100, 50, 64, 8892094, 0.3651727882671356, 0.8384800000190735, 0.3851416187763214, 0.83012, 1933.1145780086517],
[100, 100, 64, 8945494, 0.35854545486450196, 0.8431199999809265, 0.3457882954978943, 0.85264, 1931.5100429058075],
[100, 200, 64, 9112294, 0.3423980367565155, 0.8503600000190735, 0.3429523465871811, 0.84552, 1984.0002179145813],
[100, 20, 64, 8869654, 0.4245243717956543, 0.8067199999618531, 0.3983573951911926, 0.82208, 1046.544270992279],
[100, 50, 64, 8892094, 0.3982924139404297, 0.819520000038147, 0.3759589603805542, 0.82948, 1092.719614982605],
[100, 100, 64, 8945494, 0.38025016710281373, 0.8279600000190734, 0.40751560057640074, 0.80896, 1108.241466999054],
[100, 200, 64, 9112294, 0.38037813965797423, 0.827760000038147, 0.37382604162693023, 0.83632, 1111.9015009403229],
[100, 20, 64, 8869654, 0.5210826204109192, 0.7468000001525879, 0.5072802479171753, 0.7542, 731.4258360862732],
[100, 50, 64, 8892094, 0.4826498876094818, 0.7708000001335144, 0.4381257931613922, 0.79864, 747.2112848758698],
[100, 100, 64, 8945494, 0.4579147246456146, 0.7844800000190735, 0.4220376288986206, 0.80152, 739.3524148464203],
[100, 200, 64, 9112294, 0.47922230768203733, 0.7759199998664856, 0.41699990666389464, 0.8102, 755.3816978931427],
[100, 20, 64, 8869654, 0.37490771572113035, 0.832200000038147, 0.35856684294223784, 0.83916, 2306.485328912735],
[100, 50, 64, 8892094, 0.3366111166715622, 0.852959999961853, 0.3599025763893127, 0.83788, 2306.6222529411316],
[100, 100, 64, 8945494, 0.3279677852344513, 0.857040000038147, 0.3400669117259979, 0.85108, 2324.2139241695404],
[100, 200, 64, 9112294, 0.314098610162735, 0.8632399999618531, 0.3223337998437881, 0.86044, 2371.2839291095734],
[100, 20, 64, 8869654, 0.3849697910308838, 0.829719999961853, 0.3714127429103851, 0.83604, 1197.9322979450226],
[100, 50, 64, 8892094, 0.35745866830825807, 0.842000000038147, 0.40971987406730653, 0.80652, 1252.2555439472198],
[100, 100, 64, 8945494, 0.35634353694438936, 0.8419199999618531, 0.3700818202114105, 0.83356, 1277.655641078949],
[100, 200, 64, 9112294, 0.35558819856643675, 0.8443999999809265, 0.4413900222206116, 0.78516, 1282.376652956009],
[100, 20, 64, 8869654, 0.48159809851646423, 0.7737599999809265, 0.49391325139045716, 0.76792, 803.4152660369873],
[100, 50, 64, 8892094, 0.4352585868167877, 0.798839999923706, 0.4231928563785553, 0.80916, 827.1004779338837],
[100, 100, 64, 8945494, 0.41460476416587827, 0.8091600001335144, 0.47354807779312136, 0.77128, 831.9727680683136],
[100, 200, 64, 9112294, 0.4071693735408783, 0.813079999961853, 0.37275516700744626, 0.83388, 848.0242478847504]]

NUM_EPOCHS = [1, 2, 3, 4, 5]
BATCH_SIZE = [64, 128, 256]
STATE_DIM = [20, 50, 100, 200]

epoch_count = -1
batch_count = -1
for i, item in enumerate(results_lstm):
    if i % 4 == 0:
        batch_count = batch_count + 1
    if batch_count == 3:
        batch_count = 0
    if i % 12 == 0:
        epoch_count = epoch_count + 1
    print("{} & {} & {} & {} & {:0.4f} & {:0.4f} & {:0.4f} & {:0.4f} & {:0.1f} \\\\".format(item[1],
    BATCH_SIZE[batch_count], NUM_EPOCHS[epoch_count], item[3], item[4], item[5], item[6], item[7], item[8]))
