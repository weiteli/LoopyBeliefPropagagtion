//
//  main.cpp
//  OpencvLoopBelievePropagation
//
//  Created by Wei-Te Li on 14/9/27.
//  Copyright (c) 2014年 wade. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>


using namespace std;
using namespace cv;

const int LABELS = 24;
const int Iteration = 20;
const int LAMBDA = 10;
const int SMOOTHNESS_TRUC = 2;

enum DIRECTION {LEFT, RIGHT, UP, DOWN, DATA};

double countTime(){
	const clock_t begin_time = clock();
	return (double)begin_time;
}

struct Pixel{
    unsigned int msg[5][LABELS];
    int best_assignment;
};


struct MRF2D{
    vector<Pixel> grid;
    int width, height;
};


void initial(string left_img, string right_img, MRF2D &mrf);
unsigned int dataCost(Mat &left_img, Mat &right_img, int x, int y, int label);
unsigned int smoothnessCost(int i, int j);
void BP(MRF2D &mrf, DIRECTION direction);
void SendMsg(MRF2D &mrf, int x, int y, DIRECTION direction);
unsigned int MAP(MRF2D &mrf);

int main(){
    double t1 = countTime();
    MRF2D mrf;
    string s1 = "/Users/wei-teli/Downloads/LoopyBP/tsukuba-imL.png";
    string s2 = "/Users/wei-teli/Downloads/LoopyBP/tsukuba-imR.png";
    
    initial(s1, s2, mrf);
    
    for (int i=0; i<Iteration; i++) {
        BP(mrf, RIGHT);
        BP(mrf, LEFT);
        BP(mrf, UP);
        BP(mrf, DOWN);
        
        unsigned int energy = MAP(mrf);
        
        cout << "Iteration: " << i << "  Energy: " << energy << endl;
    }
    
    Mat output = Mat::zeros(mrf.height, mrf.width, CV_8U);
    
    for (int i = LABELS; i < mrf.height - LABELS; i++){
        for (int j =LABELS; j < mrf.width - LABELS; j++){
            output.at<uchar>(i,j) = mrf.grid[i*mrf.width+j].best_assignment* (256/LABELS);
        }
    }
    double t2 = countTime();
    cout << (t2-t1) / CLOCKS_PER_SEC << endl;
    
    imshow("Output", output);
    waitKey();
    return 0;
}


unsigned int datacost(Mat &left_img, Mat &right_img, int x, int y, int label){
    const int wradius = 2;
    int sum = 0;
    for (int dy = -wradius; dy <= wradius; dy++){
        for (int dx = -wradius; dx <= wradius; dx++) {
            int a = left_img.at<uchar>(y+dy, x+dx);
            int b = right_img.at<uchar>(y+dy, x+dx-label);
            sum += abs(a-b);
        }
    }
    unsigned int avg = sum / ((wradius*2+1)*(wradius*2+1));
    return avg;
}

unsigned int smoothnessCost(int i, int j){
    int d = i-j;
    return LAMBDA*min(abs(d), SMOOTHNESS_TRUC);
}

void initial(string left, string right, MRF2D &mrf){
    Mat left_img = imread(left, 0);
    Mat right_img = imread(right, 0);
    
    assert(left_img.channels() == 1);
    assert(right_img.channels() == 1);
    
    mrf.width = left_img.cols;
    mrf.height = left_img.rows;
    
    int total = mrf.width * mrf.height;
    mrf.grid.resize(total);
    
    for (int i=0; i<total; i++){
        for (int j=0; j<5; j++){
            for (int k=0; k<LABELS; k++){
                mrf.grid[i].msg[j][k] = 0;
            }
        }
    }
    
    int border = LABELS;
    
    for (int y = border; y < mrf.height-border; y++){
        for (int x = border; x < mrf.width-border; x++){
            for (int i=0; i<LABELS; i++){
                mrf.grid[y*left_img.cols+x].msg[DATA][i] = datacost(left_img, right_img, x, y, i);
            }
        }
    }
}

void BP(MRF2D &mrf, DIRECTION direction){
    int width = mrf.width;
    int height = mrf.height;
    
    switch (direction) {
        case RIGHT:
            for (int y=0; y<height; y++){
                for (int x = 0; x<width-1; x++){
                    SendMsg(mrf, x, y, direction);
                }
            }
            break;
        
        case LEFT:
            for (int y=0; y<height; y++){
                for (int x = width-1; x>0; x--){
                    SendMsg(mrf, x, y, direction);
                }
            }
            break;
        
        case UP:
            for (int x = 0; x < width; x++) {
                for (int y=height-1; y >0; y--) {
                    SendMsg(mrf, x, y, direction);
                }
            }
            break;
            
        case DOWN:
            for (int x=0; x<width; x++){
                for (int y=0; y<height-1; y++){
                    SendMsg(mrf, x, y, direction);
                }
            }
            break;
        
        default:
            assert(0);
            break;
    }
}

void SendMsg(MRF2D &mrf, int x, int y, DIRECTION direction){
    unsigned int new_msg[LABELS];
    int width = mrf.width;
    
    //collect msg and pass to the next node
    for (int i=0; i<LABELS; i++){
        unsigned int min_val = UINT_MAX;
        for (int j=0; j<LABELS; j++){
            unsigned int p = 0;
            p += smoothnessCost(i, j);
            p += mrf.grid[y*width+x].msg[DATA][j];
            
            if (direction != LEFT){
                p+= mrf.grid[y*width+x].msg[LEFT][j];
            }
            if (direction != RIGHT){
                p+= mrf.grid[y*width+x].msg[RIGHT][j];
            }
            if (direction != UP){
                p+= mrf.grid[y*width+x].msg[UP][j];
            }
            if (direction != DOWN){
                p+= mrf.grid[y*width+x].msg[DOWN][j];
            }
            min_val = min(min_val, p);
        }
        new_msg[i] = min_val;
    }
    
    //update each node's msg
    
    for (int i=0; i<LABELS; i++){
        switch (direction) {
            case LEFT:
                mrf.grid[y*width+x-1].msg[RIGHT][i] = new_msg[i];
                break;

            case RIGHT:
                mrf.grid[y*width+x+1].msg[LEFT][i] = new_msg[i];
                break;
            
            case UP:
                mrf.grid[(y-1)*width+x].msg[DOWN][i] = new_msg[i];
                break;
            
            case DOWN:
                mrf.grid[(y+1)*width+x].msg[UP][i] = new_msg[i];
                break;

            default:
                assert(0);
                break;
        }
    }
}

unsigned int MAP(MRF2D &mrf){
    for (int i=0; i<mrf.grid.size(); i++){
        unsigned int best = UINT_MAX;
        for (int j=0; j<LABELS; j++){
            unsigned int cost = 0;
            cost += mrf.grid[i].msg[LEFT][j];
            cost += mrf.grid[i].msg[RIGHT][j];
            cost += mrf.grid[i].msg[UP][j];
            cost += mrf.grid[i].msg[DOWN][j];
            
            if (cost < best){
                best = cost;
                mrf.grid[i].best_assignment = j;
            }
        }
    }
    
    int width = mrf.width;
    int height = mrf.height;
    
    unsigned int energy = 0;
    
    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            int cur_label = mrf.grid[y*width+x].best_assignment;
            energy += mrf.grid[y*width+x].msg[DATA][cur_label];
            if (x-1 >= 0) energy += smoothnessCost(cur_label, mrf.grid[y*width+x-1].best_assignment);
            if (x+1 < width) energy +=smoothnessCost(cur_label, mrf.grid[y*width+x+1].best_assignment);
            if (y-1 >= 0) energy += smoothnessCost(cur_label, mrf.grid[(y-1)*width+x].best_assignment);
            if (y+1 < height) energy += smoothnessCost(cur_label, mrf.grid[(y+1)*width+x].best_assignment);            
        }
    }
    
    return energy;
}












