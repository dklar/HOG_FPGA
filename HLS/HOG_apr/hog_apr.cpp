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


struct pixelValue{
	uint8_t bin = 0;
	uint16_t mag = 0;
};
struct objects
{
    int x,y,scale;
    float score;
};

typedef ap_fixed<16, 4> fix16;

int atan2_apr(int y, int x) {
	if (x == 0 && y == 0) {
		return 0;
	}
	if (x == 0) {
		// +/- pi/2
		return 4;
	}
	if (y == 0 && x < 0) {
		// +/- pi
		return 8;
	}
	float q = (((float) y) / (float) (x));

	if (q < 0) {
		if (q >= -0.3639) { //-0..-20°=340°
			return 8;
		}
		if (q >= -0.839) { //-20...-40°=320
			return 7;
		}
		if (q >= -1.732) { //-40..-60°=320
			return 6;
		}
		if (q >= -5.671) { //-60..-80°=320
			return 5;
		} else {
			//-100°=320
			//half of bin
			return 4;
		}
	} else {

		if (q <= 0.3639) {
			//0..20
			return 0;
		}
		if (q <= 0.839) {
			//20..40
			return 1;
		}
		if (q <= 1.732) {
			//40..60
			return 2;
		}
		if (q <= 5.671) {
			//60..80
			return 3;
		} else {
			//half of bin
			return 4;
		}
	}
}

/**
 * @brief SVM score calculation.
 * 
 * @param BlockArray HOG values
 * @param imageSlide image slide for the current position in the image
 * @param objectList List for writing back detected objects
 * @param scale Scale factor of the image pyramid 
 */
void classifyHOG_apr(fix16 (*BlockArray)[MAX_WIDTH / 16][angles * 4], int imageSlide,objects *objectList,int scale)
{

    const int cellPerWidth = MAX_WIDTH / pixelPerCell;
    const int NumberBlocksX= cellPerWidth / cellPerBlock;
	const int limit = NumberBlocksX - BlocksPerWindowX;

	fix16 Intercept = -0.277;
	fix16 weights[] = {
    0.15061,-0.20561,-0.10163,-0.14403,0.19473,-0.08734,-0.24039,-0.25564,0.08457,0.36947,-0.05390,0.20649,-0.01303,0.45472,-0.02749,-0.24547,-0.18111,0.11844,0.12782,0.01664,-0.17207,
    -0.17627,0.07035,-0.04300,0.06975,-0.12144,0.37833,0.14161,-0.56742,-0.33982,-0.38881,0.02782,-0.07950,-0.14770,0.18130,0.72184,-0.38000,0.07637,-0.14601,-0.08344,-0.05547,-0.22044,
    -0.25144,-0.41544,0.01831,-0.21176,0.20102,0.25869,0.67906,0.62725,0.14415,0.28656,-0.25994,-0.24768,0.06255,0.02820,-0.36748,-0.19115,-0.26138,-0.28978,-0.24855,-0.32726,0.15861,
    0.03096,0.23729,0.28550,0.24516,-0.03944,-0.02135,0.20046,0.36416,-0.16344,-0.20553,-0.15893,0.28224,0.05471,0.36603,0.18732,0.67779,0.16401,-0.02762,-0.42013,-0.45084,-0.27834,
    -0.28720,-0.00184,-0.05780,0.28143,-0.35335,-0.09157,-0.06391,0.13131,-0.25694,0.14464,-0.15389,0.08866,0.48076,0.44160,0.20401,-0.13376,0.21584,-0.41280,-0.45226,-0.52122,-0.16046,
    0.02111,0.09096,0.37880,0.21958,0.05910,-0.07997,-0.39157,-0.00270,-0.09378,0.10730,-0.26975,-0.05690,0.29329,-0.09263,-0.16304,-0.17994,0.00844,-0.12897,-0.02429,-0.07713,0.45436,
    0.20842,0.27925,0.00467,-0.42610,-0.52203,-0.14553,-0.04626,-0.35423,0.23710,0.15086,-0.04346,-0.27850,-0.11501,0.38754,0.00248,0.06437,0.03442,0.70349,-0.01881,0.07163,-0.02426,
    -0.22356,-0.51733,-0.16107,-0.11250,-0.12637,0.59468,-0.05524,-0.15345,0.43118,0.02020,0.02260,0.12573,0.05908,0.16399,0.19277,0.04157,-0.07942,0.23395,-0.13868,-0.32628,-0.16790,
    -0.34938,-0.33748,-0.03287,0.19630,0.57111,0.76942,0.31399,0.30375,-0.42569,-0.57085,-0.25585,-0.31212,-0.32095,0.03749,0.46881,1.02457,0.40816,0.31363,0.04806,0.18779,-0.11784,
    -0.54192,0.25266,0.36688,0.40857,0.16239,0.15959,0.34200,0.13361,-0.37649,-0.91048,0.04758,0.25972,0.36890,-0.09775,-0.19897,-0.04124,-0.20666,-0.41490,-0.83093,-0.00181,-0.12350,
    -0.24326,-0.46679,-0.09433,0.09589,0.01681,-0.39245,-0.18670,0.47895,0.47463,0.29624,-0.31533,0.14360,0.00931,0.22141,0.41607,-0.33976,0.16809,0.27587,-0.22486,-0.04234,0.53767,
    0.60031,0.33885,0.15610,-0.75500,0.11175,0.16269,-0.11693,-0.61298,-0.13583,-0.05323,-0.00537,-0.29931,-0.87642,-0.17756,-0.34468,-0.17831,-0.07495,-0.16798,0.28318,0.27215,-0.31653,
    -0.03165,0.00180,0.11173,-0.48382,0.21129,-0.06078,0.52308,-0.29042,-0.08423,0.26731,-0.03818,0.00699,-0.24770,-0.40917,0.01404,-0.10611,-0.22628,0.26215,-0.72884,-0.00891,-0.16323,
    0.04899,0.25358,0.10733,0.40549,0.15591,-0.12071,0.20064,-0.14178,0.11236,-0.03662,-0.08727,-0.07374,0.23346,0.03154,0.11459,-0.01679,0.43656,0.45554,-0.26024,0.25936,-0.42192,
    -0.06730,-0.08524,0.05621,0.28535,0.65342,0.19554,-0.29537,-0.36509,-0.24600,-0.34438,-0.20751,0.05593,-0.06606,0.23209,0.08011,-0.19281,-0.41702,-0.14640,-0.10145,-0.15571,-0.07888,
    -0.13398,0.52477,0.29432,-0.03701,-0.29970,0.03003,0.01977,0.28589,-0.20321,0.10609,0.09034,-0.03765,-0.23231,-0.48166,0.12497,0.01405,-0.00276,-0.08392,-0.32610,0.33277,-0.07657,
    0.16183,0.03023,0.04917,0.58044,0.29158,-0.50042,0.16151,0.41716,0.22528,0.08827,-0.03275,0.03031,-0.28248,-0.06440,-0.21791,-0.24336,0.00229,0.18047,-0.07494,0.10416,0.03155,
    -0.14887,-0.08804,-0.40550,-0.51060,0.07286,0.03047,-0.24818,-0.35993,-0.09781,0.34755,0.16061,-0.12724,0.28584,0.28517,0.22053,-0.27460,-0.40289,-0.22043,0.18408,0.77551,0.46806,
    -0.66208,-0.01800,-0.14252,-0.17918,-0.21854,-0.10444,0.14792,-0.03886,-0.43014,0.09054,0.45242,0.09505,-0.29814,-0.40875,-0.22355,0.17812,0.49863,0.39534,-0.25537,-0.19830,0.05335,
    -0.51140,-0.07249,-0.07475,0.04318,0.47229,0.14670,-0.02334,-0.15635,0.14532,-0.44056,0.17488,-0.27462,0.30511,0.48641,-0.02011,-0.00975,0.07081,-0.11006,-0.20908,-0.49847,-0.26718,
    0.20621,0.26726,-0.07913,0.30009,0.18053,0.15382,-0.13132,-0.19727,0.11259,-0.11135,0.20701,0.03820,-0.41658,-0.07070,0.10078,0.02533,-0.30398,-0.10642,-0.10758,-0.06905,0.06586,
    0.00600,0.42179,0.16618,0.05644,-0.26175,-0.09805,0.13550,0.12049,0.02455,-0.29799,-0.14326,0.14258,-0.11303,-0.20884,0.07735,0.15876,-0.06612,-0.02191,0.05371,0.52068,0.16004,
    0.14457,-0.06113,-0.06517,0.06802,-0.20135,-0.11288,0.05443,0.26112,0.53092,0.07352,0.24866,-0.12122,0.15583,0.08337,-0.40760,-0.43088,0.06348,0.29296,-0.30276,-0.02320,0.04421,
    0.29193,0.00818,-0.10986,0.49606,0.18800,0.09269,0.20676,0.49858,0.14835,-0.14732,-0.39255,-0.81161,-0.59078,-0.26908,-0.17609,0.04167,0.38418,0.04761,-0.09426,-0.12132,-0.49113,
    -0.29468,0.20861,0.11970,-0.39876,-0.51486,-0.17617,-0.13875,-0.02885,-0.17142,-0.10563,0.16641,0.39566,-0.29917,-0.47238,0.00445,0.14234,0.27861,0.15078,0.13399,-0.28596,-0.25886,
    -0.08182,0.30080,-0.02163,0.12925,-0.24667,-0.35877,0.11010,0.51346,0.47190,0.02668,0.43639,-0.10089,-0.07690,0.29355,-0.12767,-0.18511,0.02103,-0.24042,-0.19460,-0.39227,0.03134,
    0.14912,0.22485,0.42643,-0.18469,-0.21335,-0.01793,-0.07313,0.22889,-0.02137,0.06345,-0.16291,-0.29852,-0.62934,-0.09357,0.05287,-0.01932,0.05007,-0.13338,0.26845,0.63566,-0.19010,
    -0.58931,-0.00569,0.20760,-0.08547,0.57405,0.29075,-0.01138,0.03307,0.20709,0.02372,0.50727,-0.00658,0.06195,0.32517,0.24052,-0.12602,-0.06914,0.13853,0.15352,0.08118,0.31882,
    -0.18266,-0.14605,0.21579,0.23375,0.00905,-0.39921,0.13071,0.13289,-0.03781,-0.15431,0.26713,0.10011,0.13044,0.03641,-0.00193,-0.10720,-0.34150,-0.46871,-0.22639,-0.26516,-0.10351,
    -0.32943,-0.05940,-0.35913,-0.44365,0.10816,0.11617,-0.00936,0.27891,0.40703,0.65446,-0.09919,-0.43323,-0.24003,-0.16397,0.09566,0.07637,0.20645,-0.15749,-0.13737,-0.02991,-0.60966,
    -0.14596,-0.02431,-0.15543,-0.36722,0.09034,-0.02738,0.20796,0.41362,-0.12467,0.40979,0.13986,0.45794,0.12285,-0.04368,-0.12271,-0.35731,-0.02682,-0.34338,-0.69400,-0.35977,-0.09580,
    0.15665,-0.32232,0.05140,-0.02558,-0.06492,-0.66363,0.21039,0.21137,0.41275,0.31768,0.00031,-0.14851,-0.09921,-0.11464,-0.03597,-0.36547,0.10003,-0.07141,-0.05687,0.16305,-0.07601,
    -0.35207,-0.04872,-0.14556,0.40774,0.31475,0.47697,0.54694,0.34266,0.02010,-0.03953,-0.16871,-0.06126,-0.05739,0.25351,0.15888,-0.03199,0.16820,-0.38947,-0.15067,0.14320,-0.19638,
    -0.10653,0.04062,0.38182,-0.29394,0.03662,0.08070,0.01702,0.18690,0.28779,0.02859,-0.13039,-0.00270,-0.26646,0.24240,-0.22116,-0.22987,-0.19927,-0.16697,-0.08593,0.03427,0.09025,
    -0.06307,0.14658,-0.06994,-0.08160,0.02640,0.14281,-0.13023,-0.03030,-0.14793,0.13766,0.08536,0.19843,-0.00153,0.01022,0.01325,0.56548,0.25720,0.00958,0.07236,-0.16010,-0.07393,
    -0.09191,-0.04373,0.12530,-0.26480,-0.03632,-0.04771,-0.45205,0.06751,-0.02064,-0.13759,-0.22002,0.10632,0.11628,0.28730,-0.16975,-0.16026,0.13469,0.03926,-0.14201,-0.05112,-0.12142,
    0.29757,0.18516,0.07361,-0.26536,-0.43267,-0.10271,0.00684,0.04872,0.16270,0.31249,0.28133,-0.11200,-0.19961,-0.30906,-0.31662,-0.18837,0.00696,-0.17716,-0.23390,0.32704,-0.07771,
    -0.36227,-0.21626,-0.30273,-0.04051,0.31339,0.45018,0.31290,0.10049,-0.09166,0.21943,0.11854,-0.10925,-0.03131,0.11052,-0.03589,0.07520,-0.11690,-0.09860,-0.15463,0.12612,-0.21317,
    -0.08545,-0.05939,-0.08082,-0.03977,-0.06608,0.15571,0.40719,-0.30750,-0.24599,-0.39935,-0.02439,0.02710,0.25368,0.22386,0.14483,0.05384,-0.05795,-0.28935,0.16080,0.19549,0.38870,
    -0.10474,-0.08749,-0.31065,0.04668,0.07316,-0.22841,0.11908,0.03424,0.20779,-0.09982,0.00679,0.15967,-0.20571,-0.47201,-0.33090,-0.09902,-0.13393,0.13444,-0.26548,-0.25157,-0.05952,
    -0.08272,0.95364,0.34405,0.10028,-0.06727,0.18999,0.17795,-0.09180,-0.04128,-0.10997,0.16280,-0.13532,-0.30875,-0.02821,0.07572,-0.03610,-0.07586,0.01200,0.32248,0.52892,0.30525,
    -0.43220,-0.29697,-0.12674,-0.11908,0.32083,0.18181,-0.00200,-0.27043,-0.21426,-0.09474,0.04169,-0.20662,0.09165,0.00830,0.11143,-0.08142,-0.32678,-0.07101,0.22733,0.29215,-0.08420,
    0.18198,0.26595,0.25444,-0.05491,0.26780,-0.02195,-0.14210,0.08659,-0.19947,-0.26252,-0.15424,0.04475,-0.07190,0.30827,0.05048,-0.50238,-0.18053,0.04793,0.23361,0.33925,0.01546,
    -0.00413,-0.45915,-0.35239,-0.47182,-0.16145,0.44889,-0.30472,0.00225,0.09260,-0.22201,-0.38504,-0.21501,-0.22184,-0.02860,-0.02520,0.75408,-0.02458,0.14477,-0.24554,0.06405,-0.25778,
    0.01304,0.36213,0.21060,0.55997,-0.01901,0.11666,-0.35189,-0.04290,-0.48853,0.08970,0.29589,0.26144,0.04527,-0.23390,-0.40229,-0.04941,0.24285,-0.36349,0.17021,0.39827,0.43848,
    -0.14537,-0.07425,-0.35775,-0.39872,-0.02763,0.05717,-0.01448,0.04824,0.25318,-0.00531,0.16651,-0.24214,-0.32379,0.00242,-0.28020,-0.08105,0.25842,0.37101,0.36010,-0.04202,-0.17224,
    -0.48635,0.06277,-0.05232,0.10491,0.15703,0.33872,0.22254,-0.27217,-0.12258,-0.11055,0.25332,0.23127,0.15156,0.17089,0.33239,-0.26404,-0.03557,-0.25429,-0.29501,-0.30487,-0.12740,
    -0.31089,-0.35073,-0.00445,-0.17294,-0.11791,-0.15263,-0.27723,0.10177,0.25559,0.15128,0.30017,0.05138,0.06214,-0.28811,0.16254,0.18888,0.45565,0.31401,-0.07378,-0.05957,-0.08782,
    0.28390,0.18803,0.03068,0.03604,0.54997,0.22911,-0.07103,-0.24963,-0.19509,0.00547,0.07460,0.07179,-0.00057,0.03624,0.29547,-0.14266,-0.26967,-0.34310,-0.05041,-0.17195,-0.12933,
    -0.10928,0.10303,0.01006,-0.02048,0.10774,-0.01338,-0.44393,0.11551,0.02271,0.15203,0.07276,0.21622,-0.31621,-0.38448,0.03270,0.41474,0.22027,0.04806,-0.03155,0.31918,0.09351,
    -0.09233,0.09898,0.43914,-0.04123,0.06489,-0.06991,-0.06090,-0.34402,0.03891,0.13354,0.13767,0.01621,0.23984,0.10163,0.01560,-0.09772,-0.04034,0.01300,-0.12675,-0.00813,-0.00105,
    -0.48172,-0.23557,-0.32899,-0.11371,0.30259,0.01102,-0.29775,-0.38425,-0.23002,-0.03520,0.22928,-0.17529,0.22704,0.06825,-0.20619,-0.07644,-0.01136,0.09134,-0.09683,-0.22659,-0.11945,
    0.06035,-0.06768,-0.34614,0.36296,-0.04446,0.13432,-0.48388,-0.12781,-0.29079,0.03867,0.57512,0.13945,-0.02086,0.02222,0.11141,-0.05064,0.11688,-0.31853,0.24792,0.24054,-0.00482,
    0.08242,-0.17084,-0.15152,-0.36671,-0.01655,0.05739,0.14437,0.33486,0.18020,0.11963,-0.06385,-0.15868,0.35744,0.08016,-0.22290,-0.04115,0.45233,0.12410,-0.31995,0.13341,-0.35013,
    -0.46431,-0.12802,0.17499,0.19542,0.10341,0.11448,-0.18916,-0.09603,-0.09633,-0.23894,-0.01583,0.02400,-0.00281,-0.00409,0.14047,0.09977,-0.24355,-0.09462};

    uint16_t counter = 0;
    /*
     * Calculate the score of one window 
     * Go through the detection window and calcuate the score
     */
    SVM_Loop:
	for (int windowX = 0; windowX < limit; windowX++) {
		fix16 sum = Intercept;
		for (int y = 0; y < 8; y++) {
			for (int x = 0; x < 4; x++) {
				for (int i = 0; i < 36; i++) {
					fix16 BlockArrayVal = (fix16) BlockArray[y][x + windowX][i];
					fix16 product;
//#pragma HLS RESOURCE variable=product core=Mul_LUT
//#pragma HLS RESOURCE variable=product core=FMul_nodsp
					product = BlockArrayVal * weights[y * 144 + x * 36 + i];
					sum += product;
				}
			}
		}
		// if the score is big enough, a object is detected, add them to the list with information needed to backconvert information
		if (sum > 1.0) {
			objectList[counter].score = sum;
			objectList[counter].x = windowX * 16;
			objectList[counter].y = imageSlide;
			objectList[counter].scale = scale;
			counter++;
		}
	}
}

/**
 * @brief Calculate the HOG features for an image slide
 * 
 * @param image input image
 * @param hist adress to write the output data
 * @param w width of the image slide
 * @param h height of the image slide
 */
void computeHOG_apr(uint8_t *image,pixelValue hist[MAX_WIDTH*SLIDE_HEIGHT],int w,int h) {
	int Gy, Gx;
	ComputeHOG_loop:
	for (int y = 0; y < SLIDE_HEIGHT; y++) {
		for (int x = 0; x < w; x++) {
#pragma HLS loop_tripcount avg=0 max=0
			/*
				Gradient calculation
			*/
			if (y == 0 || y == 127) {
				Gy = 0;
			} else {
				Gy = (int) image[(y + 1) * w + x]- (int) image[(y - 1) * w + x];
			}
			if (x==0 || x >= w - 2) {
				Gx = 0;
			} else {
				Gx = (int) image[y * w + x + 1]- (int) image[y * w + x-1];
			}
			/*
				magnitude and orientation calculation
			*/
			int magnitude   = hls::abs(Gx)+hls::abs(Gy);
			int binposition = atan2_apr(Gy, Gx);
			hist[y*w + x].bin = binposition;
			hist[y*w + x].mag = magnitude;
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
            for (int i = 0; i < 9; i++){
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

void BlockSort(pixelValue hist[MAX_WIDTH*SLIDE_HEIGHT],fix16 (*BlockArray)[MAX_WIDTH/16][angles*4]) {
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
            for (int i = 0; i < 9; i++){
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

void BlockSort(pixelValue hist[MAX_WIDTH*SLIDE_HEIGHT],int (*BlockArray)[MAX_WIDTH/16][angles*4]) {
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
            for (int i = 0; i < 9; i++){
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

void normBlock_L1(fix16 (*BlockArray)[MAX_WIDTH / 16][angles * 4]) {
	for (int i = 0; i < SLIDE_HEIGHT / 16; i++) {
		for (int j = 0; j < MAX_WIDTH / 16; j++) {
#pragma HLS loop_tripcount avg=0 max=0
			ap_fixed<32,16> sum = 0;
			for (int k = 0; k < angles * 4; k++) {
				sum += hls::abs(BlockArray[i][j][k]);
			}
			for (int k = 0; k < angles * 4; k++) {
				BlockArray[i][j][k] = BlockArray[i][j][k] / sum;
			}
		}
	}
}

void normBlock_L1_sqrt(float (*BlockArray)[MAX_WIDTH / 16][angles * 4]) {
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

void normBlock_L2(float (*BlockArray)[MAX_WIDTH / 16][angles * 4]) {
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

void normBlock_L1(int (*BlockArray)[MAX_WIDTH / 16][angles * 4]) {
	NormLoop:
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


/**
 * @brief Calculated (the HOG) and classify (SVM) the image.
 * 
 * @param picture pointer to the picture data
 * @param objectList pointer to the at the beginning empty List of detected objects
 * @param scale Current scale factor for downsizeing the image. Used to determine the correct position 
 * @param w Current width of the image data. By adapting the image pyramid the value is shrinking. 
 * @param h Current height of the image data. By adapting the image pyramid the value is shrinking.
 */
void HOG_apr(uint8_t *picture, objects *objectList,int scale,int w,int h){
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
	pictureSlideLoop:
	for (int i=0;i<(h/SLIDE_HEIGHT);i++){
		#pragma HLS loop_tripcount avg=0 max=0
		uint8_t imageBuffer[MAX_WIDTH*SLIDE_HEIGHT];
		pixelValue hist[MAX_WIDTH * SLIDE_HEIGHT];
		fix16 BlockArray[SLIDE_HEIGHT/16][MAX_WIDTH/16][angles*4];
		memcpy(imageBuffer, picture+i*w*128, w*128*sizeof(uint8_t));
		computeHOG_apr(imageBuffer,hist,w,h);
		BlockSort(hist,BlockArray);
        normBlock_L1(BlockArray);
		classifyHOG_apr(BlockArray, i,objectList,scale);
	}

}

/**
 * @brief Top level functions for synthesis.
 * The module copies the data from the master interface to the local Block RAM, 
 * via a axi lite interface the information of the current state of the image pyramid is passed.
 * 
 * @param picture adress of the picture
 * @param objectList List for detected objects
 * @param scale State of the image pyramid (shrinking factor)
 * @param w State of the image pyramid (current width of the picture)
 * @param h State of the image pyramid (current height of the picture)
 */
void top_level(uint8_t* picture,objects *objectList,int scale,int w,int h){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE m_axi depth=32 port=objectList
#pragma HLS INTERFACE m_axi depth=8 port=picture
#pragma HLS INTERFACE s_axilite port=scale
#pragma HLS INTERFACE s_axilite port=w
#pragma HLS INTERFACE s_axilite port=h
	HOG_apr(picture,objectList,scale,w,h);
}


