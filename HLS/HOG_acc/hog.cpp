#include "stdint.h"
#include "ap_int.h"
#include "hls_math.h"
#include "ap_fixed.h"
#include <string.h>

static const int MAX_HEIGHT = 600;
static const int MAX_WIDTH  = 800;
static const int SLIDE_HEIGHT = 128;

static const int pixelPerCell = 8;
static const int cellPerBlock = 2;

static const int cellPerHeight_MAX = MAX_HEIGHT / pixelPerCell;
static const int cellPerWidth_MAX  = MAX_WIDTH  / pixelPerCell;

static const int angles = 9;
static const int anglesMax = 180;

static const int BlocksPerWindowX = 4;
static const int BlocksPerWindowY = 8;


static const int WINDOW_HEIGHT = 128;
static const int WINDOW_WIDTH  = 064;

static const int NRCELLS_X 	= MAX_WIDTH 	/ pixelPerCell;		//100
static const int NRCELLS_Y 	= WINDOW_HEIGHT / pixelPerCell;		//16


struct pixelValue{
	uint8_t bin = 0;
	uint16_t mag = 0;
};
struct objects
{
    int x,y,scale;
    float score;
};

inline int MODULO(int a, int b) {
	int res = a % b;
	return res < 0 ? res + b : res;
}



/**
 * Berechne fuer ein Window den Gradienten, den Bin und die magnitude,
 *
 */
void computeHOG(uint8_t *image,pixelValue hist[800*128]) {
	int Gy, Gx;
	for (int y = 0; y < 128; y++) {
		for (int x = 0; x < MAX_WIDTH; x++) {

			if (y == 0 || y == 127) {
				Gy = 0;
			} else {
				Gy = (int) image[(y + 1) * MAX_WIDTH + x]- (int) image[(y - 1) * MAX_WIDTH + x];
			}
			if (x==0 || x >= MAX_WIDTH - 2) {
				Gx = 0;
			} else {
				Gx = (int) image[y * MAX_WIDTH + x + 1]- (int) image[y * MAX_WIDTH + x-1];
			}

			int orientation = MODULO(   (int) hls::nearbyint(hls::atan2f(Gy,Gx)*57.29578049),anglesMax);
			int magnitude   =           (int) hls::nearbyint(hls::sqrtf(Gx*Gx+Gy*Gy));
			int binposition = 0;
			if (orientation < 20) {
				binposition = 0;
			} else if (orientation < 40) {
				binposition = 1;
			} else if (orientation < 60) {
				binposition = 2;
			} else if (orientation < 80) {
				binposition = 3;
			} else if (orientation < 100) {
				binposition = 4;
			} else if (orientation < 120) {
				binposition = 5;
			} else if (orientation < 140) {
				binposition = 6;
			} else if (orientation < 160) {
				binposition = 7;
			} else {
				binposition = 8;
			}
			hist[y*MAX_WIDTH + x].bin = binposition;
			hist[y*MAX_WIDTH + x].mag = magnitude;
		}

	}
}

void classifyHOG(float (*BlockArray)[MAX_WIDTH / 16][angles * 4], int imageSlide,objects *objectList)
{

    const int cellPerWidth = MAX_WIDTH / pixelPerCell;
    const int NumberBlocksX= cellPerWidth / cellPerBlock;
    float weights[] = {0.38659216557681786, -0.17765256024752688, -0.3303109657526789, -0.25685432111625195, 0.6965258327808991,
                -0.08955468271447518, -0.011380397634332682, 0.025097102256305622, 0.46277236012643636, 0.6239231917421081, 0.46487964753040406,
                -0.13799357500253623, -0.016888500558488596, 0.26172621720559996, 0.04821951747212017, 0.023656928749301538, -0.09692267641617822,
                0.38657150312346705, 0.058898306339590846, -0.11991402835984309, -0.36072727757510653, -0.0477902095969468, -0.2899069596259559,
                -0.32274920851082195, 0.08348847343872155, -0.18609957726371237, 0.20603368915678924, 0.5306844868527822, -0.22569740958374251,
                -1.3156962343867646, -0.2626308683241106, 0.08665935834926336, -0.35285209805658946, -0.17594085553912087, -0.26501947087226974, 0.5267531157604476,
                -0.3730070129004375, 0.06039537754387444, -0.27949539109529464, 0.03609077695247769, -0.23006337126339574, -0.3036866281162325, -0.3530879729833021,
                -0.6792968540688507, -0.19821961362097604, -0.02032369263888472, -0.36240567638421983, 0.9067808168296151, 0.6959318719605542, 0.16594258905030163,
                0.6277287010261795, -0.09736088933839021, -0.1721172824349242, -0.18577644467039459, 0.13266893426266385, 0.6009722418913955, -0.36438542195946344,
                -0.17314239205946444, -0.6984952129033746, -0.22844133210071138, -0.4233477672263535, -0.3725564296237725, -0.25074588627016403, 0.2940840031220864,
                0.738439147354931, 0.6543911872768233, 0.5885065020180674, 0.11222033857156306, 0.0023691344874631455, 0.11650009225025111, 0.12949801484864878,
                -0.26667476715229066, -0.11878485451939501, -0.957294860597313, 0.050770826668031385, 0.3214676268829064, 0.08905728062365176, -0.04998044159693689,
                0.30603461866164755, 0.20692019556646818, 0.10636622712084819, -0.8683807371772824, -0.6727806254273542, -0.6933160438734528, -0.20766926341414343,
                -0.11541085392279932, 0.08607086735388811, -0.15301218020549026, -0.42005012925764723, -0.004613804722518225, -0.045468989431366534, 0.11370348850897878,
                0.24097254794205933, 0.6027939900463791, -0.7299282637344802, 0.05513034305631216, 0.6007888470426176, 1.106059078370329, 0.9748657271856439, 0.018232914503309283,
                -0.608775387474421, -0.2776838229330741, -0.04188293065973812, -0.2245122910675267, 0.5077090804562541, -0.1621508231050685, 0.04736229689929851, 0.7472863031418847,
                -0.18676221367996268, -0.03513929274036071, -0.23154025701618475, -0.14158471549976967, 0.09653762635994981, 0.32503754228639864, 0.09584914897907892,
                -0.08525097469772173, 0.008030607315296527, -0.5772909482650174, -0.19204684498874075, -0.14367286518421854, -0.22202652658871394, 0.6203029591774539,
                0.08148150444447162, -0.37721092404242623, -0.038915560480204274, 0.3226002008452323, 0.3646406167309299, 0.06115128340402619, -0.1685034100852258,
                -0.2902914558819232, -0.37079574837941826, -0.1266900651564857, 0.32789888348515106, 0.025086043920012784, -0.5088203592968257, 0.5085090036999764,
                0.22029186195735437, -0.4328935866203492, -0.0711042607080329, 0.5244277567553483, -0.01919674898111124, -0.17644921323627644, 0.04659358508093908,
                0.5976429866210384, 0.06756608675687345, -0.027021385078669913, -0.22744484461179842, -0.35344771117441826, -0.025174928031647864, 0.06464091537904505,
                0.3043492406409905, -0.2742678247616235, 0.1081856440517651, 0.10028305642625604, 0.20258510264976806, 0.3829259971282978, 0.4634716501834759,
                0.7208928502929963, 0.25199194237249545, -0.09142576719552345, 0.06426650721428655, 0.17056742666698144, -0.15626557830117307, 0.38777398571553645,
                -0.27296199398531146, -0.17721801231601653, 0.2758591049405363, -0.5217638231496513, -0.4063684504660831, -0.13598576497366865, 0.2068654470273788,
                -0.3601655159320223, 0.24005355960083316, 0.5551316923364656, 0.26567048115195485, -0.01810778603814834, -0.739848279054075, -0.4626583334053222,
                -0.6143425651127888, -0.13872379335429513, -0.22961858245122413, 0.059685587156782106, 0.46880274737316474, 1.059599567115242, 0.4387734325214341,
                -0.10954135864439205, 0.05905604435790401, 0.013348842735977912, 0.07267669984262896, 0.22214112318252077, -0.11847040335572828, 0.20490024750729588,
                0.8030137573443803, -0.11765146299762687, -0.13012319011645665, 0.6777709265728838, 0.5567631554530815, -0.056182710246877696, 0.11539031183217038,
                0.10905840190139011, 0.2634441317236037, -0.09031516738143884, -0.19546351155735991, -0.7954912610865605, -0.4822749188135612, -0.5758104543970132,
                -0.47471335010383675, -0.9077191528555603, -0.592743166681835, 0.5273965795341291, -0.18983564818442056, -0.42813781992267325, -0.07352747808935066,
                0.3524672007277965, -0.25431267367694715, -0.35246435259807873, 0.10050440464678313, -0.2450875174337002, -0.17070173701670854, 0.41957135458148526,
                -0.7194257068486617, 0.4945713287928385, 0.2334585563583073, 1.0563828593645164, 0.3020798180178488, 0.18023806737304457, 0.11265764887489395,
                -0.08900820376739584, -0.06821410587984536, -0.14749590233952858, 0.41661292109191755, 0.571905281356774, 0.16070737504046625, 0.7146515911560747,
                -0.0857227093001645, -0.167491089735465, -0.08669240258248666, -0.14477872903856015, -0.7265688092792176, 0.00031676910138294333, -0.07843476670401849,
                -0.7112723582352386, -0.13563704478925862, -0.7341540669522834, 0.2584090256442566, -0.0712232947203848, -0.19956881322991402, -0.3728835090292982,
                0.22351193482565124, 0.5226381607807369, -0.3050408527669503, -0.6789211768380148, 0.4213290374105716, 0.06465236034890885, -0.1755728764197496, -0.6541537688376013,
                0.18311205482989412, 0.04548592740254355, 0.5923201016827391, -0.3869975913127298, -0.38532632674325296, 0.19688706903464218, -0.3084971880108348, -0.31586469193313277,
                -0.37568982400871553, 1.0174185696092866, 0.36372987969576864, 0.060163970840654536, -0.0838201413353418, 0.4913378361517715, -0.3486681878490926, -0.4641992027458635,
                -0.8558903273961802, -0.25111457477079885, -0.02375955588108556, 0.26784181506885346, 0.5998394137292686, 0.5206887135623824, -0.6293902414682452, -0.07313513749335318,
                -0.2560221490896103, -0.1310959194435548, -0.17269936287714927, 0.37905110170299433, -0.239110143281265, -0.03458578342623932, 0.4454580610995445, 0.3461559327256872,
                -0.5783365826673811, 0.42289399862796523, 0.2824677513642387, 0.014561543121601317, -0.189050653879833, -0.14302967601947425, 0.23390957496037917, 0.027975418479257157,
                0.06953296326320062, -0.7267757375820803, 0.6150739695002191, 0.4121530096140498, -0.08901868636941028, -0.23786160678666932, -0.1962873283810631, -0.026231856424422332,
                -0.07840252709399284, -0.6963802837810162, 0.17958682125102418, 0.9625864181461602, -0.39052850118727206, -0.6998109936282028, -0.4826150484723445, -0.5361249062793494,
                0.04248849381387593, 0.2949581244408216, 0.004629758085130579, -0.04831829999594251, 0.36794655814013355, 0.30088768014968065, -0.06879466679624928,
                0.08668514628752344, 0.05871857445929085, 0.37957029820688576, 0.04058805203742326, 0.22023889519872164, 0.9097986046575812, 0.07643182252918095,
                0.005558685466727718, 0.21830210073614223, -0.633185042413599, -0.15268989828582447, 0.09152166347372553, -0.2219167305512901, 0.20931472145757943,
                0.059472503858800964, 0.4397725744180193, 0.17757681656113594, 0.1305851908390622, -0.4146244751189253, 0.13647121254954195, 0.05360655199329226,
                0.7112344856537063, -0.35377983045644074, -0.3671447720650173, 0.16791648547077206, 0.16184931810553793, 0.10717040757157398, 0.24011027314216205,
                0.060202205697317344, -0.1934978666771946, 0.026406028357270753, -0.3650042462217295, -0.6011233324535541, -0.39694355658530167, -0.3669427684184419,
                -0.41731218353235333, 0.023209295045000664, -0.06909342229636488, 0.44273640964955024, 0.4422448461967855, -0.5083504958595002, -0.07083254809428949,
                0.0444307256279786, 0.7845982774273225, 0.19989867910305123, -0.2820774699755978, 0.0062533423881652, 0.13471495152254315, 0.6568124785328108,
                -0.05551497341891367, -0.1994750964312193, 0.3138413277907501, -0.1926788203571225, 0.025261044307470865, -0.5023027470532583, 0.1574930458153871,
                0.35606319485869903, 0.7358415041300977, -0.12868821234117295, -0.32058281335651595, 0.322455846657825, 0.04225791954588331, -0.15010285433226211,
                -0.5653986068060692, 0.3738222700203492, 0.24087911580241678, 0.1451255555858648, -0.5787926666620788, -0.23830108224969657, -0.2867126595515831,
                -0.6630569264410304, -0.22517703150349022, -0.475368564935703, 0.0625703217316971, 0.07400342707745654, 0.3438866003174763, -0.25526063520554554,
                0.09794160023774807, 0.01695301066424967, -0.3116082599075207, -0.3973929863348957, 0.15043911988823816, 0.16593199387809743, 0.23750902369538301,
                0.48395060237779763, -0.3602957516465553, -0.08831244827575548, -0.30202336125882695, -0.32754281360112947, -0.3287405916291138, 0.49847344095320284,
                0.28446865149547707, 0.8513991198833819, 0.7674605893528383, 0.3809746926543342, -0.6403061821995207, -0.4963414216762561, -0.4674282051211463,
                -0.6006624532568438, -0.7007165980762106, 0.42881781679303493, 0.36670724547708816, 0.4107031085576953, 0.017452999848843468, -0.0994707868919551,
                -0.2411536542223997, 0.2640959360198458, -0.12328574064120834, -0.2658631980518548, -0.2318390131941899, 0.045612576160840294, 0.2412401649948528,
                0.10274149282944044, 0.2458977728343853, 0.39532086605799555, -0.10977178012155031, -0.2102681156215848, 0.19590313997565523, -0.2236562575076507,
                0.116517027625705, 0.18804222569063797, 0.10623041295785614, -0.7690221711186964, 0.613974860390826, 0.48140109446645596, -0.12384856954427824,
                -0.18877445669597015, -0.2082406459784279, -0.6611955572282794, 0.46744650026427514, 0.08620874081957038, 0.3283370424229572, 0.6326742923266547,
                0.3212133568903937, -0.29804560710073363, -0.2872244922651279, -0.3895000684024511, -0.11212699354511826, -0.1738266563750288, -0.17151456915597807,
                -0.2092094166523716, 0.36578907598364474, 0.10057316088831753, -0.3041072167705896, -0.06779890532649788, -0.20322242395563336, -0.032559272669213164,
                -0.00896758021649713, -0.06276159948874185, 0.03361189965919348, -0.20597268873960187, 0.24492208626420964, -0.2608217561819948, -0.12166653143386891,
                0.32913311669764067, -0.31333350310529706, -0.511724544253641, 0.023316479175625363, 0.3519148998974642, 0.045318970970693165, 0.26641314672055577,
                -0.22953119237962485, 0.11743790098749986, 0.08887898577469432, 0.6259500578098084, 0.3258726045208121, -0.662627753879984, 0.02527272842967876,
                0.5071818048476552, 1.2231903754096816, -0.16784765752972788, -0.5818404064113719, -0.08815057674490132, 0.20351883992338768, 0.39741885455230713,
                0.16067594000700902, -0.6235968122828804, -0.21517213988119394, 0.021613230137824506, -0.12412498268382369, -0.010608299737106076, 0.1940931339720721,
                -0.5635125734954288, 0.15192780843041642, -0.8272311570002625, -0.6682353454428671, 0.6464224741057657, 0.7727299582715905, 0.35925990833907173,
                -0.44865863030676945, -0.5490854843326479, -0.23228441399143399, 0.29929062357341985, -0.283679853578808, -0.08394293699790571, -0.037202452883505314,
                0.42751818957349225, -0.12067674893074988, 0.02141465119649365, 0.09205832074123162, -0.03153942364886258, 0.6549884137349389, -0.1949647628309976,
                -0.7170889730264168, -0.3591994383252387, 0.15259501666548167, -0.18852697272256072, -0.12347121374598183, 0.25856772371379433, 0.03587891731620547,
                -0.019156047464371134, -0.3610502547149615, 0.11351794455525836, 0.006681209313457436, -0.620549914803045, -0.19888735981905958, -0.15128786571769778,
                    -0.15277444082636313, 0.17937898767772345, 0.8428640834014625, 0.5089955570081139, 0.14176788604395418, 0.03868251800814997, 0.029291192765245642,
                    -0.5549021227389642, -0.406576803346096, 0.13108039407962094, -0.06613649208758905, -0.4143498037718849, 0.2019751345407882, 0.02989547809827982,
					-0.42296759085442304, 0.2637080438839031, -0.2512321379978882, 0.41265859375587116, 0.06343340653403262, 0.08557617312053226, 0.1347387053378757,
					0.16737526074055709, -0.5539459253156116, -0.029983598157325927, -0.25301703474766285, 0.1483729185734293, -0.1533198342562332, -0.14820498627735196,
					-0.6477471694884146, -0.06433642994295435, 0.47198939428709563, 0.36193615226035725, -0.09456340191525364, 0.06829930034367279, 0.2526124696094945,
					0.6962844325488365, 0.07781933417087687, -0.4539903551374485, 0.24775349878692118, 0.31992546559822616, -0.09276845896772602, 0.1168467615847111,
					0.5072330370945589, -0.06153555735236875, 0.425229878534242, -0.2204441435237215, 0.09581867910234672, -0.25578040899687876, -0.4005124524509887,
					0.10533966717608707, 0.10630141175979405, 0.26417751887748786, -0.24765660414008311, 0.07154877691015965, 0.0287461642753982, 0.5954521919401836,
					-0.004378864280255922, 0.33888659889256867, -0.18732706153998568, 0.3556362959577664, -0.1567147788612495, -0.06447240089260664, -0.022054769451726106,
					0.10937382588536114, -0.06861869377483253, 0.18557559754061737, -0.14916845019977365, 0.2789453603320337, 0.24691912564573107, -0.5050603059539317,
					-0.2616176253723971, -0.1534714595679585, -0.17165383671003576, -0.27903530184975217, -0.519125063492832, -0.18074408294006702, -0.5263207462177043,
					-0.08375744211593501, 0.3438709196847568, -0.33566829729130354, -0.13889849724359524, 0.34675939381173043, 1.0714766403807647, 0.46299738893394027,
					-0.13876641428970704, -0.5383060586487726, -0.5766755285315381, 0.23008731760363343, -0.055717528237525786, -0.0640850689055057, 0.1968408560858362,
					-0.23156016681601432, 0.5004963630490623, -0.9516981875003373, 0.21445418031853966, 0.2172887244104133, -0.0157842121747, -0.359966224931564,
					0.29986817911543523, -0.12851917942555222, 0.25543021611519, 0.5417082264809001, 0.10780149864880811, -0.37331095897287847, 0.34131255752853495,
					0.3196465864454005, -0.20764432720578163, -0.18277406744131539, 0.004883523311512244, 0.07579884984513055, -0.08735069763086184, -0.7040240218883611,
					-0.4926675449477078, -0.27135588453852055, 0.18143232058550304, -0.03350229004755301, -0.18528628169510777, -0.14716522475178254, 0.06498357351450482,
					0.13703163249565975, -0.3820100101028507, 0.05079970895081482, 0.012158817910717892, -0.15570184918038518, -0.32332295911830505, 0.43533228496072757,
					-0.07361679127215255, 0.18327264816165895, 0.22489383026519802, -0.15077290734294158, -1.0652759010512842, -0.1364885847483344, 0.4874398445676545,
					-0.5174958999777367, -0.2552049758834423, -0.10206608096949728, 0.05548871749485591, 0.1940475753009795, -0.3907911103206717, -0.2724728602460117,
					0.6559276577995437, 0.0851464557664384, 0.2956247543797033, 0.3628962332558559, 0.271621121775254, 0.4122179248708608, 0.4073266604757124,
					0.26744904015720855, -0.37445986678204957, -0.49026849913622184, 0.4155816954980501, 0.1206797681049672, 0.3815009942033665, -0.046769932440224246,
					0.006623349586886957, 0.11873845084864763, 0.2969841908826503, 0.35700014239277333, 0.1424192563549666, -0.14206933840535302, -0.5162034310012192,
					-0.01483428253779864, -0.057001588103342014, 0.15133323805202073, 0.65342332310487, 0.30725955018207873, -0.3286875230766806, -0.140670803882386,
					-0.18826638827020928, 0.08980375592931775, 0.08751523839198437, 0.03170430074828234, -0.4780905877011777, 0.4939838654327149, -0.30563940649357335,
					0.02242132671586554, -0.19883224083409656, 0.10634207094591676, 0.10090834655812661, -0.12076530363990684, -0.3905492714917001, -0.23317168535073418,
					-0.25182901292877646, 0.22377247767878664, 0.13174290244996129, -0.08724915060954978, 0.184482358308178, -0.018755009481445345, 0.420233065032769,
					0.023870824515311785, -0.24030519570419334, 0.0070455750948298845, -0.18607131849161787, -0.4026559363258481, 0.451649477716899, 0.3161942735857426,
					-0.10959069168891905, 0.46920913640698086, -0.09992398369524878, -0.23731278522978635, -0.12420591251762565, 0.16174820601180687, 0.05484168980114606,
					-0.0029854343638303867, -0.09302569175396457, -0.005812628339265864, 0.5739443257879806, -0.20277091082172455, -0.6033265948146478, -0.044816772956346534, 0.13734789524227345, -0.2788068062548434, 0.8652443965414969, -0.1389424381991476, -0.0541155944185761, -0.3658988078980402, -0.25602879904476633, -0.04059762213535433, 0.03860567853272342, -0.41307298943843657, 0.5654007641552559, 0.3084167083550476, -0.015899089498987943, -0.23915989254705972, -0.0875785326731337, -0.3293186448450582, -0.1494523347141757, 0.13122326662078787, -0.3963222697597505, 0.16568721069763934, 0.07831922071880147, -0.23642551642955556, -0.5323549326325336, -0.19303437641895083,
                -0.6334049634734248, 0.3780328448217177, 0.43260697017547695, -0.5566757339736071, -0.2891021281769048, 0.32713705318666614, -0.038193181968280886,
                -0.24371619949403647, -0.2027757050975611, -0.48595475330680715, 0.23813433735249184, 0.5091249351072491, -0.1514564211472638, 0.5114392934656641,
                -0.2639271989329006, 0.2936938161871326, 0.2402499412504562, 0.5921375453792335, -0.3737050088729943, -0.3841575806427036, 0.18699332286949258,
                0.673911997581038, -0.2503866754310118, 0.19983369982499755, -0.34634467551803655, -0.2626538957997282, 0.31662341258829874, 0.0475934740823908,
                0.38104586641246785, 0.37478695343705903, -0.3365802337448691, -0.3282183205434758, 0.13977671606878786, -0.6203691752520047, -0.07846603094030001,
                0.1933055487926107, 0.16466316003217576, 0.5905010685177188, 0.1250575661689905, -0.11948608817289302, 0.3677203631684704, 0.10539005172817094,
                0.36931347586531627, 0.05089557789735721, 0.22559237501364335, -0.8312782971169738, -0.12080601825028825, 0.8781705364215876, 0.694775301464113,
                -0.4801651118807197, -0.2424573807866038, -0.28742176123453994, 0.04177337694799814, -0.3879341363479031, -0.5009235570205873, -0.43316933832015325,
                0.38049677082810435, -0.19075485965277103, -0.11192823549408933, 0.16263777105485183, 0.5138999264689841, -0.21580058813309191, -0.2449390261786392,
                -0.2979318655867946, 0.22532036826235796, -0.24769352465795821, -0.5777783777549301, 0.1173835292341911, -0.060953574578703, -0.1953079130846984,
                -0.1433899463846076, 0.8106716607806174, -0.3282045305334586, -0.04652576713931102, 0.07536233069574302, 0.3295952488704599, -0.3452894022164758,
                0.13846195986606466, 0.12141413707959248, -0.4061596384411074, 0.009143955551931118, -0.5920864768419014, 0.2706022190100127, -0.06141000271907021,
                -0.21726618556124142, 0.4454432675010792, 0.10857144719148787, -0.15964583579674213, 0.12423812400278529, 0.8038530328514839, 0.060537573212214826,
                0.09046827905190348, -0.2253383020210772, -0.10005176860728507, 0.09856575207904597, 0.48611174746802804, -0.3584263471223105, -0.01325509252678209,
                0.6971804979385972, -0.40760266259537004, -0.24885469567478322, -0.2605536719139078, 0.13353158725109826, -0.20326932045913795, 0.24844424655785788,
                0.3915257792057921, -0.14042082564747588, 0.5667325186691664, -0.3290366385249719, -0.6588420849856279, -0.3365403512964945, 0.3484841491537928,
                0.7565468520878211, 0.06711100816904197, 0.5227995460076654, -0.5129281955586522, 0.30639920093677403, -0.03141941775319665, -0.1458761770201808,
                -0.09641647941959627, -0.25501950505723325, -0.2044369856404736, -0.39481326938285843, -0.25429374090360657, -0.03228333907780984, 0.5377293651389393,
                0.2674114979189373, -0.6357779401790307, 0.006324906182619523, -0.08494570961653002, 0.11814617406082174, 0.6044734224333176, 0.6679117474552442,
                -0.27580868551738313, -0.6332073384097773, -0.6846310906681444, -0.4581314479061825, 0.698584690892995, -0.1586701787247949, 0.5044270555606236,
                0.25263068756261764, 0.18699189913677014, -0.17137842042705131, -0.4026847717505515, -0.3236880046551494, -0.28807848502499894, 0.45836626388242524,
                -0.016278585141204106, 0.5789567813906399, 0.08208990355620485, 0.2949909375804105, -0.7109674684774722, -0.23019951920190784, -0.13537119710665677,
                -0.42914090427122, 0.303534294097867, 0.2063390729075698, 0.3834245265790629, -0.24591456120191638, 0.35415760305084, -0.7164778174778823,
                -0.8173541376823829,-0.44844110615956667, -0.0240498018119463, 0.7235674923736649, 0.5817730966494842, -0.18482622906595175, -0.2233000411099731,
                -0.3064101362975363,-0.2814578882069383,-0.16285183172585616, -0.544469801160021, -0.18226924450893497, 0.2788420676911773, 0.018922123835275258,
                0.0739860316663632,0.07065234024803524,-0.6902806209853045,-0.5010870484704111, -0.2927078393048006, -0.326918031417909, 0.1683853417504982,
				0.3428635090179478, 0.024462916565227526, -0.12661264477171202,0.06187407354667334, 0.13596792075556613, 0.01531043803353832, -0.06973805422766027,
				-0.4415316701540975, 0.15455256372505302, 0.9887808355219472,0.03947994381744994, 0.21454037345562, 0.32857172294545467, 0.3211383941770314,
				-0.10053664248528793, -0.2064610312687075, -0.14284719744513108,0.3887049951100296, 0.5790392189675492, 0.40812397791981003, 0.10264718846587557,
				-0.4719442205068188, -0.19143523285196803, -0.4115664426108996,
                -0.3517167741418775, -0.4358296643710798, -0.06705425939379328, 0.3264283565970326, 0.23919252497690788, -0.0791984141698027, -0.3398231747564084,
                0.17219768388980528, 0.2311766699727065, 1.1876406782003703, 0.4267000577590127, 0.17999065896722968, -0.093684314707436, -0.14284627132598487,
                -0.2312595297536336, -0.24496149316403787, -0.14335774680543095, -0.45192377092566116, -0.24388203910483808, -0.06467887761579415, 0.4767150472457722,
                0.5849220560944761, 0.3637006201833131, -0.011478750992527579, 0.015422923843131306, -0.10505098084822916, 0.10892727817398232, 0.2029561555614998,
                0.3220432780485802, -0.3551995436512169, -0.5684807788386359, -0.10539617182870824, 0.08867198694621177, -0.06905204069593793, 0.0331968743435752,
                -0.46694571814322605, 0.47689759809765614, -0.07327568285400397, -0.06739412489437845, 0.058820562516744265, -0.24714009343713522, 0.19833967079098813,
                -0.10098893665152384, -0.049898926738008775, 0.25042976552420426, 0.23345723716923272, 0.4438302773535178, 0.22298240778191264, 0.24294353033845764,
                -0.2543161707642297, -0.028637227008459807, -0.4641855622057082, -0.795099369614495, -0.31072784585271973, -0.08956947236728742, 0.058534566543603214,
                0.4067527786730969, -0.06228436040552326, -0.037792412147927154, -0.0934310147705186, -0.12594812695913166, 0.1982990860130456, 0.17660596046178773,
                0.5923124473340803, -0.22612161904097552, -0.44488650196055773, -0.12913719883292152, 0.2846534006019315, 0.168290525705937, 0.2589013105404436,
                0.2691450971937443, 0.0009351120600270674, -0.10077233815599504, 0.08459644775187936, 0.0333040669246478, -0.29196451650929445, 0.09460578143230548,
                0.6355581047098927, 0.48691160734918154, 0.01798363699252153, -0.07525724029596796, 0.07132262367786267, -0.1733799804579701, -0.7495458385212086,
                0.028226197096655616, 0.8206456595564736, -0.5168741114473728, -0.19921475786401505, -0.24607949356371284, -0.16919560785334958, 0.2564252836622212,
                0.19094625297134663, -0.2494760982453287, -0.1712576408005428, -0.18388994096888056, 0.35008571742335937, -0.4243083118814052, -0.5627463734596684,
                0.04396987789372812, 0.42941137923192163, -0.010913282917748228, -0.23975201701784238, -0.1606422483286012, 0.11390413958296734, 0.07658813005726947,
                -0.14097003777031303, 0.5032304855572188, 0.2618210775990554, -0.008643789797710843, -0.4775779115443496, -0.17111325093634722, 0.42244923394397405,
                0.07806587607730034, -0.015638050361754998, -0.0411380439914027, 0.40074433392423153, 0.12544246103484558, -0.26564051508868597, -0.1823513059717589,
                0.2127896369196073, 0.10137131530732345, 0.04940662850856945, 0.04336734241960816, -0.036648014914207834, -0.08467254338055777, 0.022761852624525878,
                0.06780534329533304, -0.20669032164489456, -0.1976992600518429, -0.46280010138560174, 0.4411604912493368, 0.003699246511845048, -0.20995982689081846,
                0.10436758561679248, 0.09261801709480035, 0.2874271330457698, -0.0454153615701069, -0.023487057896151877, -0.2337693647972385, -0.6610154250305802,
                0.28380718926021226, -0.1763745770864788, -0.0055705344254337995, 0.020098174334023774, 0.21741364547592693, 0.2059786139339789, 0.24097664332544425,
                0.27226035662631487, -0.04274291651116124, 0.10162045091755825, -0.24271814700874572, -0.3292354488881593, -0.06073790002482817, -0.35259257604765426,
                -0.38899936156592685, -0.029402251156178852, -0.5940585798163583, 0.4695012886191279, -0.2782930307825758, -0.11955920952982205, 0.3218518706153704,
                0.35370175774752505, 0.14609372615265795, 0.2517148999026915, -0.05805861047603968, -0.6452233001883708, 0.10128792578061234, -0.3357270242650388,
                -0.2073987497314942, 0.04596546369922206, 0.2454984254108476, 0.7313924023370342, 0.1425645717684907, -0.13441071786058473, -0.3767433892508108, 0.05601490209182387};

    int counter = 0;

    for (int windowX = 0; windowX < NumberBlocksX - BlocksPerWindowX; windowX++)
    {
        float sum = -0.17;
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                for (int i = 0; i < 36; i++)
                {
                    sum += BlockArray[y][x + windowX][i] * weights[y * 144 + x * 36 + i];
                }
            }
        }
        if (sum > 1.0){
            objectList[counter].score = sum;
            objectList[counter].x = windowX * 16;
            objectList[counter].y = imageSlide;
            counter++;
        }
    }
}

void BlockSort(pixelValue hist[MAX_WIDTH*SLIDE_HEIGHT],float (*BlockArray)[MAX_WIDTH/16][angles*4]) {
    int binsum[angles];
    memset((void *)binsum,0,sizeof(int)*angles);
    BlockSortLoop:
    for (int y = 0; y < SLIDE_HEIGHT; y+=16) {
		for (int x = 0; x < MAX_WIDTH; x+=16) {
#pragma HLS loop_tripcount avg=0 max=0
			ComputeCell1:
			for(int i=0;i<8;i++){//(0,0) bis (8,8)
				for(int j=0;j<8;j++){
                    pixelValue current_val = hist[(y + i) * MAX_WIDTH + x + j];
                    binsum[current_val.bin] +=current_val.mag;
                }
			}
            for (int i = 0; i < 9; i++){//normiere, setzte auf null, schreibe wert
                BlockArray[y / 16][x / 16][i] = binsum[i];// / 64.;
                binsum[i] = 0;
            }
            ComputeCell2:
            for(int i=0;i<8;i++){//(0,8) bis (8,16)
				for(int j=8;j<16;j++){
                    pixelValue current_val = hist[(y + i) * MAX_WIDTH + x + j];
                    binsum[current_val.bin] +=current_val.mag;
                }
			}
            for (int i = 0; i < 9; i++){
                BlockArray[y / 16][(x /16)][i+9] = binsum[i];// / 64.;
                binsum[i] = 0;
            }
            ComputeCell3:
            for(int i=8;i<16;i++){//(8,0) bis (16,8)
				for(int j=0;j<8;j++){
                    pixelValue current_val = hist[(y + i) * MAX_WIDTH + x + j];
                    binsum[current_val.bin] +=current_val.mag;
                }
			}
            for (int i = 0; i < 9; i++){
                BlockArray[(y / 16)][x / 16][i+18] = binsum[i];// / 64.;
                binsum[i] = 0;
            }
            ComputeCell4:
            for(int i=8;i<16;i++){//(8,8) bis (16,16)
				for(int j=8;j<16;j++){
                    pixelValue current_val = hist[(y + i) * MAX_WIDTH + x + j];
                    binsum[current_val.bin] +=current_val.mag;
                }
			}
            for (int i = 0; i < 9; i++){
                BlockArray[(y / 16)][(x / 16)][i+27] = binsum[i];// / 64.;
                binsum[i] = 0;
            }
		}
	}

}

void normBlock_L1(float (*BlockArray)[MAX_WIDTH / 16][angles * 4]) {
	for (int i = 0; i < SLIDE_HEIGHT / 16; i++) {
		for (int j = 0; j < MAX_WIDTH / 16; j++) {
#pragma HLS loop_tripcount avg=0 max=0
			float sum = 0;
			for (int k = 0; k < angles * 4; k++) {
				sum += hls::abs(BlockArray[i][j][k]);
			}
			for (int k = 0; k < angles * 4; k++) {
				BlockArray[i][j][k] = BlockArray[i][j][k] / sum;
			}
		}
	}
}

void hog_acc(uint8_t *picture, objects *objectList){
    /*
    Copy parts of the data from Master interface to locale block ram.
    These data are then processed:
        Compute the gradient magnitude.
        Sort the gradient & magnitude per block
        Normalize the blocks
        Calculate the SVM score.
    This will be repeated until the complete data from the iamge is read.
    It is advantageous that the height of the image is a multiple of 128
    */
	for (int i=0;i<(MAX_HEIGHT/128);i++){
		uint8_t imageBuffer[MAX_WIDTH*SLIDE_HEIGHT];
		pixelValue hist[MAX_WIDTH*SLIDE_HEIGHT];
		float BlockArray[SLIDE_HEIGHT/16][MAX_WIDTH/16][angles*4];
		memcpy(imageBuffer,picture+i*MAX_WIDTH,MAX_WIDTH*128*sizeof(uint8_t));
		computeHOG(imageBuffer,hist);
		BlockSort(hist,BlockArray);
		normBlock_L1(BlockArray);
		classifyHOG(BlockArray, i,objectList);
	}
}

void top_level(uint8_t* picture,objects *objectList){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE m_axi depth=32 port=objectList
#pragma HLS INTERFACE m_axi depth=8 port=picture
	hog_acc(picture,objectList);

}

