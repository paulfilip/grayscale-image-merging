// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>

using namespace std;

vector<Point> selected_points_image1;	//vector pentru stocarea punctelor selectate din prima imagine
vector<Point> selected_points_image2;	//vector pentru stocarea punctelor selectate din a doua imagine
Vec3b colour[200];						//vector pentru stocarea a 200 de culori generate aleator
boolean can_select = true;				//variabila booleana pentru inceputul procesarii si selectia punctelor corespondente


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}


/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Proiect LIPIRE DE IMAGINI
//CALACEAN Ionut & FILIP Paul
//grupa 30239, UTCN - Calculatoare

//functie isInside: verifica daca un pixel se regaseste intr-o imagine
bool isInside(Mat img, int i, int j)
{
	return (i >= 0 && i < img.rows && j >= 0 && j < img.cols);
}

//functie care genereaza aleatoriu 200 de culori
void initRandomColours()
{
	//generare 200 culori random
	default_random_engine gen;
	uniform_int_distribution<int> d(0, 255);

	for (int i = 0; i < 200; i++)
	{
		colour[i] = Vec3b(d(gen), d(gen), d(gen));
	}
}

//procesare pereche puncte corespondente, functie de callback pentru mouse
void procesare_pereche(int event, int x, int y, int flags, void* param)
{
	Mat src = *((Mat*)param);
	
	static int point_selected = 0;		//variabila statica pentru stocarea nr. de puncte selectate

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		if (can_select)

		{
			Vec3b pixel = src.at<Vec3b>(y, x);
			printf("Selected position (x,y): %d, %d\n",
				x, y);


			if (point_selected % 2 == 0) //se selecteaza punct din prima imagine
			{
				line(src, Point(x + 5, y), Point(x - 5, y), colour[point_selected]);
				line(src, Point(x, y + 5), Point(x, y - 5), colour[point_selected]);
				selected_points_image1.push_back(Point(y, x));
				imshow("Source 1", src);
			}
			else //selectie punct corespondent din a doua imagine
			{
				line(src, Point(x + 5, y), Point(x - 5, y), colour[point_selected - 1]);
				line(src, Point(x, y + 5), Point(x, y - 5), colour[point_selected - 1]);
				selected_points_image2.push_back(Point(y, x));
				imshow("Source 2", src);
			}

			point_selected++;
		}
	}

	if (event == CV_EVENT_RBUTTONDOWN)
	{
		if (point_selected % 2 == 1 || point_selected < 6)
		{
			cout << "Perechi incomplete de puncte sau numar prea mic de puncte alese!" << endl;		//eroare de selectie
		}
		else
		{
			can_select = false;
			point_selected = 0;				//se poate incepe procesarea pe datele curente, resetare pentru utilizare ulterioara
		}
	}
}

//calculare mapping function, utilizand image registration si metoda least squares
std::vector<Mat> image_registration()
{
	int K = selected_points_image1.size(); //cate puncte dintr-o imagine am ales
	int N = 3;

	//construire matricea A
	/*
		|1	x1	y1|
		|1	x2	y2|
		|.	.	. |
		|1	xk	yk|
	*/
	Mat A = Mat(K, N, CV_32F);
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < N; j++)
		{
			switch (j)
			{
			case 0: A.at<float>(i, j) = 1.0f; break;
			case 1: A.at<float>(i, j) = selected_points_image1.at(i).x; break;
			case 2: A.at<float>(i, j) = selected_points_image1.at(i).y; break;
			default: break;
			}

		}
	}

	//Matricea X contine coordonata X a punctelor corespondente din a doua imagine
	//Matricea Y contine coordonata Y a punctelor corespondente din a doua imagine
	Mat X = Mat(K, 1, CV_32F);
	Mat Y = Mat(K, 1, CV_32F);

	for (int i = 0; i < K; i++)
	{
		X.at<float>(i, 0) = selected_points_image2.at(i).x;
		Y.at<float>(i, 0) = selected_points_image2.at(i).y;
	}

	//Matricile a si b reprezinta matricile cautate, specifica functia de mapare - a pentru x si b pentru y
	Mat a = Mat(N, 1, CV_32F);
	Mat b = Mat(N, 1, CV_32F);

	//matricea A este patratica, daca este inversabila, se poate inmulti cu A^(-1) si se afla solutiile
	if (K == N)
	{
		a = A.inv() * X;
		b = A.inv() * Y;

	}
	else
	{
		//in cazul in care nr. de puncte corespondente este mai mic decat N(3) => avem nevoie de mai multe perechi de puncte
		if (K < N)
		{
			cout << "More corresponding points need to be found !";
			exit(1);
		}
		else
		//aplicam metoda Least-Squares
		{
			Mat transpose = A.t();

			a = ((transpose * A).inv()) * transpose * X;
			b = ((transpose * A).inv()) * transpose * Y;

		}
	}

	for (int i = 0; i < N; i++)
	{
		cout << "a(" << i << ") = " << a.at<float>(i, 0) << endl;
	}

	for (int i = 0; i < N; i++)
	{
		cout << "b(" << i << ") = " << b.at<float>(i, 0) << endl;
	}

	//Matricile a si b se returneaza intr-un vector de matrici cu 2 elemente. Pe pozitia 0 => a, pe pozitia 1 => b
	std::vector<Mat> result;
	result.push_back(a);
	result.push_back(b);
	return result;
}

//efectuare interpolare bilinara
float billinear_interpolation(Mat src, float x, float y)
{
	int i = (int)x;
	int j = (int)y;

	float miu = x - i;
	float lambda = y - j;

	float result = lambda * (miu * src.at<uchar>(i+1,j+1) + (1-miu)*src.at<uchar>(i+1,j)) + (1 - lambda)*(miu * src.at<uchar>(i,j+1) + (1-miu) * src.at<uchar>(i,j));

	return result;


}

//mapare inversa de la imag2 la imag1
//Metoda prin rezolvarea sistemului de 2 ecuatii cu 2 necunoscute
/*
	a1 + a2*x + a3*y = u;
	b1 + b2*x + b3*y = v;
*/
Point backward_transformation(float u, float v, Mat a, Mat b)
{
	Point result;
	float a1 = a.at<float>(0, 0);
	float a2 = a.at<float>(1, 0);
	float a3 = a.at<float>(2, 0);

	float b1 = b.at<float>(0, 0);
	float b2 = b.at<float>(1, 0);
	float b3 = b.at<float>(2, 0);

	float y = (u*b2 - a1 * b2 - a2 * v + a2 * b1) / (a3*b2 - a2 * b3);
	float x = (u - a1 - a3 * y) / a2;

	result.x = x;
	result.y = y;

	return result;
}

//functie principala
void test_project()
{

	char fname_image1[MAX_PATH];
	char fname_image2[MAX_PATH];
	initRandomColours();

	while (openFileDlg(fname_image1))
	{
		//curatarea vectori puncte selectate si permitere selectare puncte
		selected_points_image1.clear();
		selected_points_image2.clear();
		can_select = true;

		Mat src1 = imread(fname_image1, CV_LOAD_IMAGE_COLOR);
		Mat src1_bw = imread(fname_image1, CV_LOAD_IMAGE_GRAYSCALE);
		
		int height1 = src1.rows;
		int width1 = src1.cols;

		Mat src2;
		Mat src2_bw;
		while (openFileDlg(fname_image2))
		{
			src2 = imread(fname_image2, CV_LOAD_IMAGE_COLOR);
			src2_bw = imread(fname_image2, CV_LOAD_IMAGE_GRAYSCALE);
			break;
		}

		
		int height2 = src2.rows;
		int width2 = src2.cols;

		setMouseCallback("Source 1", procesare_pereche, &src1);
		setMouseCallback("Source 2", procesare_pereche, &src2);

		imshow("Source 1", src1);
		imshow("Source 2", src2);

		setMouseCallback("Source 1", procesare_pereche, &src1);
		setMouseCallback("Source 2", procesare_pereche, &src2);

		waitKey();

		std::vector<Mat> solutions;	//solutiile a si b

		if (!can_select)
		{
			cout << "Acum incepre procesarea" << endl;
			for (int i = 0; i < selected_points_image1.size(); i++)
			{
				cout << "(" << selected_points_image1.at(i).x << ";" << selected_points_image1.at(i).y << ") -> ("<< selected_points_image2.at(i).x << ";" << selected_points_image2.at(i).y << ")" << endl;

			}
			//Algoritm image registration
			solutions = image_registration();
		}
		
		Mat transformed_first_image = Mat(height1, width1, CV_32FC2);
		Mat a = solutions.at(0);
		Mat b = solutions.at(1);
		float max_w = FLT_MIN,max_h= FLT_MIN, min_h = FLT_MAX,min_w=FLT_MAX;

		//transformare pixeli imagine 1 -> coordonate pixeli imagine destinatie
		for (int i = 0; i < height1; i++)
		{
			for (int j = 0; j < width1; j++)
			{
				Vec2f transformed_points = Vec2f((a.at<float>(0, 0) + i * a.at<float>(1, 0) + j * a.at<float>(2, 0)), (b.at<float>(0, 0) + i * b.at<float>(1, 0) + j * b.at<float>(2, 0)));
				transformed_first_image.at<Vec2f>(i, j) = transformed_points;
				if (transformed_points[0] > max_h)
					max_h = transformed_points[0];
				if (transformed_points[0] < min_h)
					min_h = transformed_points[0];
				if (transformed_points[1] > max_w)
					max_w = transformed_points[1];
				if (transformed_points[1] < min_w)
					min_w = transformed_points[1];

			}
		}

		printf("max_w = %f, max_h = %f, min_h = %f, min_w = %f ", max_w, max_h, min_h, min_w);

		//calculare dimensiuni fereastra rezultat
		int new_window_width=width2, new_window_height=height2;

		if (max_h > height2)
		{
			new_window_height =(int) max_h;
		}
		if (max_w > width2)
		{
			new_window_width = (int)max_w;
		}
		if (min_w < 0)
		{
			new_window_width -= (int)min_w;
		}
		if (min_h < 0)
		{
			new_window_height -= (int)min_h;
		}

		Mat dst = Mat::zeros(new_window_height, new_window_width, CV_8UC1);

		int deplas_h = min_h < 0 ? (int)-min_h : 0;
		int deplas_w = min_w < 0 ? (int)-min_w : 0;
		printf("\ndeplas H = %d , deplas w = %d", deplas_h, deplas_w);


		//imaginea 2 se regaseste in totalitate in imaginea rezultat, deplasata cu depl_h pe inaltime si depl_w pe latime
		for (int i = 0; i < height2; i++)
		{
			for (int j = 0; j < width2; j++)
			{
				dst.at<uchar>(i + deplas_h, j + deplas_w) = src2_bw.at<uchar>(i, j);

			}
		}

		//Metoda 1 pentru Bacward Warping: inversare puncte corespondente, se re-calculeaza vectorii coloana a si b
		std::vector<Mat> reverse_solutions;
		std::vector<Point> aux_list = selected_points_image2;
		selected_points_image2 = selected_points_image1;
		selected_points_image1 = aux_list;
		
		image_registration();

		reverse_solutions = image_registration();
		Mat reverse_a = reverse_solutions.at(0);
		Mat reverse_b = reverse_solutions.at(1);
		
		//bacward mapping + interploare bilineara
		/*for (int i = 0; i < new_window_height; i++)
		{
			for (int j = 0; j < new_window_width; j++)
			{
				if (i > deplas_h && i < deplas_h + height2 && j > deplas_w && j < deplas_w + width2)
				{
					continue;
				}
				Vec2f transformed_points = Vec2f((reverse_a.at<float>(0, 0) + (i-deplas_h) * reverse_a.at<float>(1, 0) + (j-deplas_w) * reverse_a.at<float>(2, 0)), (reverse_b.at<float>(0, 0) + (i-deplas_h) * reverse_b.at<float>(1, 0) + (j-deplas_w) * reverse_b.at<float>(2, 0)));
				if (transformed_points[0] >= 0 && transformed_points[0] < height1 && transformed_points[1] >= 0 && transformed_points[1] < width1)
				{
					if (isInside(src1_bw, transformed_points[0] + 1, transformed_points[1] + 1))
						dst.at<uchar>(i, j) = billinear_interpolation(src1_bw, transformed_points[0], transformed_points[1]);
				}
				else
				{
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
		*/
		//Metoda 2 Backward Warping : se utilizeaza functia backward_transformation(calculata) + interpolare bilineara
		for (int i = 0; i < new_window_height; i++)
		{
			for (int j = 0; j < new_window_width; j++)
			{
				if (i > deplas_h && i < deplas_h + height2 && j > deplas_w && j < deplas_w + width2)
				{
					continue;
				}
				int color = 0;
				Point backward = backward_transformation(i - deplas_h, j - deplas_w , a, b);
				//testam sa fie in prima imagine
				if (backward.x >= 0 && backward.x < height1 && backward.y >= 0 && backward.y < width1)
				{
					if (isInside(src1_bw, backward.x + 1, backward.y + 1))
						color = (int)billinear_interpolation(src1_bw, backward.x, backward.y);
				}
				dst.at<uchar>(i, j) = color;
			}
		}
		printf("\nwindow height = %d , window width = %d", new_window_height, new_window_width);


		imshow("REZULTAT", dst);

		waitKey();
		destroyAllWindows();


	}

}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Lipire imagini\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				test_project();
				break;
		}
	}
	while (op!=0);
	return 0;
}