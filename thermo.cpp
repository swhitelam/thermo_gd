
#include <random>
std::mt19937_64 rng_engine; // default-constructed engine
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
std::normal_distribution<double> normal01(0.0, 1.0);

#include <cmath>
#include <ctime>
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>

//minimal std imports
using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;

char st[256];

//thermodynamic parameters
double kt=1.0;
double j2=1.0;
double j4=1.0;

//time parameters (microseconds)
double tau;
double timestep=1e-3;
double trajectory_time=0.2;

//mnist parameters
const int lx=28;
const int number_of_pixels=lx*lx;

const int test_set=10000;
const int number_of_digits=test_set;

//registers
int digit_type[number_of_digits];
double digit[number_of_digits][number_of_pixels];

//testing digit indices for a given class
int test_digits[number_of_digits];
int number_of_test_digits = 0;

//computer parameters
const int number_of_digit_classes=10;
const int number_of_in_plane_bonds=4;
const int number_of_input_spins=number_of_pixels;
const int number_of_hidden_spins=64;
const int number_of_output_spins=number_of_digit_classes;
const int number_of_spins=number_of_input_spins+number_of_hidden_spins+number_of_output_spins;
const int number_of_bonds=number_of_input_spins*number_of_in_plane_bonds/2+number_of_input_spins*number_of_hidden_spins+(number_of_hidden_spins*(number_of_hidden_spins-1))/2+number_of_hidden_spins*number_of_output_spins+(number_of_output_spins*(number_of_output_spins-1))/2+number_of_input_spins*number_of_output_spins;
const int number_of_parameters=number_of_bonds+number_of_spins;

//bond registers
int number_of_neighbors[number_of_spins];
int bond_constituents[number_of_bonds][2];
int neighbors[number_of_spins][number_of_spins];
int bond_number[number_of_spins][number_of_spins];

double spin[number_of_spins];
double delta_spin[number_of_spins];
double grad_potential[number_of_spins];
double parameters[number_of_parameters];
double parameter_gradients[number_of_parameters];

int bond_counter[7]; //to id bonds by class (in order: ii, ih, hh, ho, oo, io)

int chosen_digit;

//for display
int q_display_trajectory=0;
int trajectory_count=0;

int report_index=-1;
const int number_of_report_steps=1000;
double report_time=trajectory_time/(1.0*number_of_report_steps);

//functions void
void initialize(void);
void read_mnist(void);
void langevin_step(void);
void test_computer(void);
void run_trajectory(void);
void read_parameters(void);
void set_interactions(void);
void set_test_input(int s1);
void visualize_outputs(void);
void collect_test_digits(void);
void connection_field_grid(void);
void accumulate_gradients(int i);
void normalize_digits_by_l2(void);
void make_mnist_picture(int digit_id);
void partial_test_computer(int cycle_number);
void write_png(const char* filename, uint8_t* rgb, int width, int height);
void draw_thick_line(uint8_t* img, int width, int height, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b, double thickness);

//functions int
int bias(int i);
int neighbor(int s1, int j);

//functions double
double gauss_rv(double sigma);

int main(void){

//seed the RN generator, read MNIST
initialize();

//read computer parameters
read_parameters();

//plot input-hidden connection weights
connection_field_grid();

//visualize outputs;
visualize_outputs();

//test computer
test_computer();

return 0;

}

double gauss_rv(double sigma){

return sigma * normal01(rng_engine);
    
}


void initialize(void){

//clean up
snprintf(st,sizeof(st),"rm report_*.*");
cout << st << endl;
system(st);

//seed RN generator
rng_engine.seed(static_cast<uint64_t>(time(NULL)));

//read MNIST
read_mnist();

//test set
collect_test_digits();

//set interactions
set_interactions();

//initialize parameters
for(int i=0;i<number_of_parameters;i++){parameters[i]=gauss_rv(0.0);}

//hyperparams
ofstream out_hyper("report_config.txt", ios::out);
out_hyper << std::setprecision(16);
out_hyper << "kt " << kt << "\n";
out_hyper << "j2 " << j2 << "\n";
out_hyper << "j4 " << j4 << "\n";
out_hyper << "timestep " << timestep << "\n";
out_hyper << "trajectory_time " << trajectory_time << "\n";

}


void normalize_digits_by_l2(void){

double max_norm=0.0;
double digit_norm[number_of_digits];

for(int i=0;i<number_of_digits;i++){

double norm=0.0;

for(int j=0;j<number_of_pixels;j++){
norm+=digit[i][j]*digit[i][j];
}

norm=sqrt(norm);
digit_norm[i]=norm;

if(norm>max_norm){max_norm=norm;}

}

if(max_norm<1e-12){max_norm=1.0;}

for(int i=0;i<number_of_digits;i++){

double scale=max_norm/digit_norm[i];

for(int j=0;j<number_of_pixels;j++){
digit[i][j]*=scale;
}

}

}

void read_mnist(void){

int i,j;
int label;
int dummy;

//from training set
double mnist_mu=0.13066;
double mnist_sigma=0.112003;
mnist_sigma=sqrt(mnist_sigma-mnist_mu*mnist_mu);

cout << "reading mnist " << endl;

ifstream infile2("mnist_test.dat", ios::in);
for(j=0;j<test_set;j++){

infile2 >> label;
digit_type[j]=label;

for(i=0;i<number_of_pixels;i++){

infile2 >> dummy;

digit[j][i]=dummy/(255.0);

}
}

cout << "mnist read " << endl;
cout << "scaling and normalizing digits" << endl;

for(i=0;i<test_set;i++){
for(j=0;j<number_of_pixels;j++){

digit[i][j]=(digit[i][j]-mnist_mu)/mnist_sigma;

}}

normalize_digits_by_l2();

}


int bias(int i){

return i+number_of_bonds;

}


void collect_test_digits(){

number_of_test_digits = 0;

for (int i = 0; i < test_set; i++) {
int label = digit_type[i];
if (label >= 0 && label < number_of_digit_classes) {
test_digits[number_of_test_digits++] = i;
}}

/*
cout << "First 11 test digits (index, class):" << endl;
for (int i = 0; i < 11 && i < number_of_test_digits; i++) {
cout << i << ": " << test_digits[i] << ", class " << digit_type[test_digits[i]] << endl;
}

cout << endl;
cout << " total test digits " << number_of_test_digits << endl;
cout << endl;

*/

}

void set_test_input(int s1){

s1 = s1 % number_of_test_digits;
chosen_digit = test_digits[s1];

//set computer
for(int i=0;i<number_of_input_spins;i++){spin[i]=digit[chosen_digit][i];}
for(int i=number_of_input_spins;i<number_of_spins;i++){spin[i]=0.0;}

}


//accumulate parameter gradients during noising
void accumulate_gradients(int i){

double q1=0.0;

if(i>=number_of_bonds){
// if i is a bias

int s=i-number_of_bonds;
q1=(delta_spin[s]+grad_potential[s]*timestep)/(2.0*kt);
parameter_gradients[i]+=q1;

}
else{
//i is a bond

//get s1 and s2
int s1=bond_constituents[i][0];
int s2=bond_constituents[i][1];

q1=(delta_spin[s1]+grad_potential[s1]*timestep)*spin[s2]/(2.0*kt);
q1+=(delta_spin[s2]+grad_potential[s2]*timestep)*spin[s1]/(2.0*kt);
parameter_gradients[i]+=q1;

}
}

void langevin_step(void){

for(int i=number_of_input_spins;i<number_of_spins;i++){

double xi=spin[i];
double xi3=xi*xi*xi;
double force=0.0;

//intrinsic force
force=-2*j2*xi-4*j4*xi3;

//bias
force-=parameters[bias(i)];

//interactions
for(int n=0;n<number_of_neighbors[i];n++){
int j=neighbors[i][n];
if(i!=j){force-=parameters[bond_number[i][j]]*spin[j];}
}


//noise
double noise=sqrt(2.0*timestep*kt)*gauss_rv(1.0);

//set delta_spin
delta_spin[i]=timestep*force+noise;

//safeguard
if(!std::isfinite(delta_spin[i])){cerr << "delta_spin non-finite at i="<<i<<" (force="<<force<<")"<<endl;exit(1);}

}

//update spins
for(int i=number_of_input_spins;i<number_of_spins;i++){spin[i]+=delta_spin[i];}

}


// Minimal PNG writer (uncompressed, 24-bit RGB, 28x28)
#include <zlib.h>
void write_png(const char* filename, uint8_t* rgb, int width, int height) {
    // Use stb_image_write or a minimal PNG writer, but for now use a PPM as a placeholder
    // (replace with PNG writer if needed)
    // Write PPM (for debugging)
    // FILE* f = fopen(filename, "wb");
    // fprintf(f, "P6\n%d %d\n255\n", width, height);
    // fwrite(rgb, 1, width * height * 3, f);
    // fclose(f);

    // Write PNG using libpng or stb_image_write, but here is a minimal zlib-compressed PNG
    // Use system call with ImageMagick's convert if PNG library is not available
    // For now, use stb_image_write if available, otherwise fallback to PPM and convert
    // For now, use a temporary PPM and convert to PNG
    char ppmname[128];
    snprintf(ppmname, sizeof(ppmname), "%s.ppm", filename);
    FILE* f = fopen(ppmname, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(rgb, 1, width * height * 3, f);
    fclose(f);
    char cmd[512];
    // Resize to 560x560 (28*20) using ImageMagick, then remove ppm
    snprintf(cmd, sizeof(cmd), "magick -filter point %s %s; rm %s", ppmname, filename, ppmname);
    system(cmd);
}

void run_trajectory(void){

ofstream out1;
ofstream out2;

if(q_display_trajectory==1){
snprintf(st,sizeof(st),"report_output.dat");
out1.open(st, ios::out | ios::app);
//snprintf(st,sizeof(st),"report_hidden.dat");
//out2.open(st, ios::out | ios::app);
}

tau=0.0;
report_index=-1;

while(tau<trajectory_time){

langevin_step();

if((q_display_trajectory==1) && ((int) (tau/report_time) > report_index)){
  
report_index=(int) (tau/report_time);

out1 << tau << " ";
for(int j=0;j<number_of_digit_classes;j++){out1 << spin[j+number_of_input_spins+number_of_hidden_spins] << " ";}
out1 << endl;

//out2 << tau << " ";
//for(int j=0;j<number_of_hidden_spins;j++){out2 << spin[j+number_of_input_spins] << " ";}
//out2 << endl;

}

tau+=timestep;

}

if(q_display_trajectory==1){

out1.close();
out2.close();

snprintf(st,sizeof(st),"mv report_output.dat report_output_%d_%d_%d.dat",chosen_digit,digit_type[chosen_digit],0);
system(st);
//snprintf(st,sizeof(st),"mv report_hidden.dat report_hidden_%d_%d_%d.dat",chosen_digit,digit_type[chosen_digit],0);
//system(st);

trajectory_count++;

}
}

void test_computer(void){

cout << "calculating test-set classification accuracy" << endl;

snprintf(st,sizeof(st),"report_accuracy.dat");
ofstream out1(st,ios::out);

snprintf(st,sizeof(st),"report_accuracy_log.dat");
ofstream out2(st,ios::out);

// Open top-2 accuracy output file
snprintf(st,sizeof(st),"report_accuracy_top2.dat");
ofstream out3(st,ios::out);

int samples=1;
double signal[number_of_digit_classes];

int digit_count=0;
int n_digits=number_of_test_digits;
int offset=number_of_input_spins+number_of_hidden_spins;

double acc=0.0;
double acc2=0.0;

// confusion matrix: rows = true label, cols = predicted label
int confusion[number_of_digit_classes][number_of_digit_classes];
for(int a=0;a<number_of_digit_classes;a++){
for(int b=0;b<number_of_digit_classes;b++){
confusion[a][b]=0;
}}

//flags
q_display_trajectory=0;

for(int i=0;i<n_digits;i++){

  for(int j=0;j<number_of_digit_classes;j++){signal[j]=0.0;}

  for(int k=0;k<samples;k++){
    set_test_input(i);
    run_trajectory();
    for(int j=0;j<number_of_digit_classes;j++){signal[j]+=spin[offset+j]/(1.0*samples);}
  }

  out2 << "Digit " << chosen_digit << " type " << digit_type[chosen_digit] << " outputs: ";
  for (int j=0; j<number_of_digit_classes; j++){out2 << signal[j] << " ";}
  out2 << endl;

  // find top-1 and top-2
  int pred1=0, pred2=0;
  double b1=signal[0], b2=-1e300;
  for(int j=1;j<number_of_digit_classes;j++){
    double v=signal[j];
    if(v>b1){ b2=b1; pred2=pred1; b1=v; pred1=j; }
    else if(v>b2){ b2=v; pred2=j; }
  }
  if(pred1==digit_type[chosen_digit]){acc+=1.0;}
  if(pred1==digit_type[chosen_digit] || pred2==digit_type[chosen_digit]){acc2+=1.0;}
  
  //confusion matrix
  confusion[digit_type[chosen_digit]][pred1]++;

  digit_count++;

  out1 << digit_count << " " << acc/(1.0*digit_count) << endl;
  out3 << digit_count << " " << acc2/(1.0*digit_count) << endl;

  if((digit_count % 1000)==0){
    cout << digit_count << " top-1 accuracy " << acc/(1.0*digit_count) << " top-2 accuracy " << acc2/(1.0*digit_count) << endl;
  }
}

// write confusion matrix
cout << "writing confusion matrix to report_confusion.dat" << endl;
ofstream out_cm("report_confusion.dat", ios::out);
for(int a=0;a<number_of_digit_classes;a++){
for(int b=0;b<number_of_digit_classes;b++){
out_cm << confusion[a][b] << " ";
}
out_cm << endl;}

}


void set_interactions(void){

int bond_count=0;
int s1,s2;

//int bond_counters[6]; //to id bonds by class (in order: ii, ih, hh, ho, oo, io)
bond_counter[0]=bond_count;

//input-input connections
for(int i=0;i<number_of_input_spins;i++){
for(int j=0;j<number_of_in_plane_bonds/2;j++){

s1=i;
s2=neighbor(s1,j); //left, down (don't double count)

//neighbor lists
neighbors[s1][number_of_neighbors[s1]]=s2;
neighbors[s2][number_of_neighbors[s2]]=s1;
number_of_neighbors[s1]++;
number_of_neighbors[s2]++;

//bonds
bond_number[s1][s2]=bond_count;
bond_number[s2][s1]=bond_count;
bond_constituents[bond_count][0]=s1;
bond_constituents[bond_count][1]=s2;
bond_count++;

}}

//end of ii class
bond_counter[1]=bond_count;

//input-hidden connections
for(int i=0;i<number_of_input_spins;i++){
for(int j=0;j<number_of_hidden_spins;j++){

s1=i;
s2=number_of_input_spins+j;

//neighbor lists
neighbors[s1][number_of_neighbors[s1]]=s2;
neighbors[s2][number_of_neighbors[s2]]=s1;
number_of_neighbors[s1]++;
number_of_neighbors[s2]++;

//bonds
bond_number[s1][s2]=bond_count;
bond_number[s2][s1]=bond_count;
bond_constituents[bond_count][0]=s1;
bond_constituents[bond_count][1]=s2;
bond_count++;

}}

//end of ih class
bond_counter[2]=bond_count;

//hidden-hidden connections
for(int i=0;i<number_of_hidden_spins;i++){
for(int j=0;j<i;j++){

s1=number_of_input_spins+i;
s2=number_of_input_spins+j;

//neighbor lists
neighbors[s1][number_of_neighbors[s1]]=s2;
neighbors[s2][number_of_neighbors[s2]]=s1;
number_of_neighbors[s1]++;
number_of_neighbors[s2]++;

//bonds
bond_number[s1][s2]=bond_count;
bond_number[s2][s1]=bond_count;
bond_constituents[bond_count][0]=s1;
bond_constituents[bond_count][1]=s2;
bond_count++;

}}

//end of hh class
bond_counter[3]=bond_count;

//hidden-output connections
for(int i=0;i<number_of_hidden_spins;i++){
for(int j=0;j<number_of_output_spins;j++){

s1=number_of_input_spins+i;
s2=number_of_input_spins+number_of_hidden_spins+j;

//neighbor lists
neighbors[s1][number_of_neighbors[s1]]=s2;
neighbors[s2][number_of_neighbors[s2]]=s1;
number_of_neighbors[s1]++;
number_of_neighbors[s2]++;

//bonds
bond_number[s1][s2]=bond_count;
bond_number[s2][s1]=bond_count;
bond_constituents[bond_count][0]=s1;
bond_constituents[bond_count][1]=s2;
bond_count++;

}}

//end of ho class
bond_counter[4]=bond_count;

//output-output connections
for(int i=0;i<number_of_output_spins;i++){
for(int j=0;j<i;j++){

s1=number_of_input_spins+number_of_hidden_spins+i;
s2=number_of_input_spins+number_of_hidden_spins+j;

//neighbor lists
neighbors[s1][number_of_neighbors[s1]]=s2;
neighbors[s2][number_of_neighbors[s2]]=s1;
number_of_neighbors[s1]++;
number_of_neighbors[s2]++;

//bonds
bond_number[s1][s2]=bond_count;
bond_number[s2][s1]=bond_count;
bond_constituents[bond_count][0]=s1;
bond_constituents[bond_count][1]=s2;
bond_count++;

}}

//end of oo class
bond_counter[5]=bond_count;

//input-output connections
for(int i=0;i<number_of_input_spins;i++){
for(int j=0;j<number_of_output_spins;j++){

s1=i;
s2=number_of_input_spins+number_of_hidden_spins+j;

//neighbor lists
neighbors[s1][number_of_neighbors[s1]]=s2;
neighbors[s2][number_of_neighbors[s2]]=s1;
number_of_neighbors[s1]++;
number_of_neighbors[s2]++;

//bonds
bond_number[s1][s2]=bond_count;
bond_number[s2][s1]=bond_count;
bond_constituents[bond_count][0]=s1;
bond_constituents[bond_count][1]=s2;
bond_count++;

}}

//end of io class
bond_counter[6]=bond_count;

/*
cout << "bonds ordered as "<< endl;
cout << " ii " << bond_counter[0] << " to " <<  bond_counter[1] << endl;
cout << " ih " << bond_counter[1] << " to " <<  bond_counter[2] << endl;
cout << " hh " << bond_counter[2] << " to " <<  bond_counter[3] << endl;
cout << " ho " << bond_counter[3] << " to " <<  bond_counter[4] << endl;
cout << " oo " << bond_counter[4] << " to " <<  bond_counter[5] << endl;
cout << " io " << bond_counter[5] << " to " <<  bond_counter[6] << endl;
cout << "total bond count " << bond_count << " " << number_of_bonds << endl;

cout << " biases ordered as " << endl;
cout << " i " << bond_counter[6] << " to " << bond_counter[6]+number_of_input_spins << endl;
cout << " h " << bond_counter[6]+number_of_input_spins << " to " << bond_counter[6]+number_of_input_spins+number_of_hidden_spins << endl;
cout << " o " << bond_counter[6]+number_of_input_spins+number_of_hidden_spins << " to " << number_of_parameters << endl;

*/

//cout << "total parameter count " << number_of_parameters << endl;

}


void draw_thick_line(uint8_t* img, int width, int height, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b, double thickness){

int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
int err = dx + dy;

while (true) {
    for (int ty = -int(thickness); ty <= int(thickness); ty++) {
    for (int tx = -int(thickness); tx <= int(thickness); tx++) {
        int px = x0 + tx;
        int py = y0 + ty;
        if (px >= 0 && px < width && py >= 0 && py < height) {
            int idx = 3 * (py * width + px);
            img[idx + 0] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }}

    if (x0 == x1 && y0 == y1) break;
    int e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
}
}

void make_mnist_picture(int digit_id) {

uint8_t rgb[lx * lx * 3];

for (int i = 0; i < number_of_pixels; ++i) {

    double v = digit[digit_id][i];

    // scale to [0,1] using logistic for visual consistency
    double scale = 5.0;
    double t = 1.0 / (1.0 + exp(-scale * v));

    uint8_t r = (uint8_t)(255 * (1.0 - t));
    uint8_t g = (uint8_t)(255 * (1.0 - t));
    uint8_t b = 255;

    rgb[3 * i + 0] = r;
    rgb[3 * i + 1] = g;
    rgb[3 * i + 2] = b;
}

char fname[128];
snprintf(fname, sizeof(fname), "report_digit_%d.png", digit_id);

write_png(fname, rgb, lx, lx);
}


// For fabs, std::max
#include <algorithm>
void connection_field_grid(void){

// image grid: tiles laid out in a near-square grid
const int tile_size=lx;
int tiles_per_row=(int)ceil(sqrt(1.0*number_of_hidden_spins));
if(tiles_per_row<1){tiles_per_row=1;}
int tiles_per_col=(number_of_hidden_spins+tiles_per_row-1)/tiles_per_row;
const int img_w=tiles_per_row*tile_size;
const int img_h=tiles_per_col*tile_size;

std::vector<uint8_t> rgb(img_w*img_h*3,255);

// loop over hidden spins
for(int idx=0;idx<number_of_hidden_spins;idx++){

int h=number_of_input_spins+idx;

// determine min/max J_ij for normalization
double jmin=1e10;
double jmax=-1e10;

for(int ni=0;ni<number_of_neighbors[h];ni++){
int i=neighbors[h][ni];
if(i>=number_of_input_spins){continue;} // only input pixels
int b=bond_number[i][h];
double val=parameters[b];
if(val<jmin){jmin=val;}
if(val>jmax){jmax=val;}
}

double jabs_max=fabs(jmin)>fabs(jmax)?fabs(jmin):fabs(jmax);
if(jabs_max<1e-12){jabs_max=1.0;}

// render connection weights
for(int ni=0;ni<number_of_neighbors[h];ni++){
int i=neighbors[h][ni];
if(i>=number_of_input_spins){continue;}

int b=bond_number[i][h];
double val=parameters[b];

double intensity=fabs(val)/jabs_max;
if(intensity>1.0){intensity=1.0;}

uint8_t r,g,bl;
if(val>=0.0){
    r=(uint8_t)(255*(1.0-intensity));
    g=(uint8_t)(255*(1.0-intensity));
    bl=255;
}
else{
    r=255;
    g=(uint8_t)(255*(1.0-intensity));
    bl=(uint8_t)(255*(1.0-intensity));
}

    int local_row=i/lx;
    int local_col=i%lx;

    int tile_row=idx/tiles_per_row;
    int tile_col=idx%tiles_per_row;

    int row=tile_row*tile_size+local_row;
    int col=tile_col*tile_size+local_col;

    int pix=3*(row*img_w+col);
    rgb[pix+0]=r;
    rgb[pix+1]=g;
    rgb[pix+2]=bl;
}
}

// Upscale and add gridlines for clarity
int scale = 8; // magnification factor for clarity
const int out_w = img_w * scale;
const int out_h = img_h * scale;
std::vector<uint8_t> rgb_big((size_t)out_w*(size_t)out_h*3u, 255);

// Nearest-neighbor upscale: expand each source pixel to a scale x scale block
for(int y=0; y<img_h; ++y){
  for(int x=0; x<img_w; ++x){
    int src = 3*(y*img_w + x);
    uint8_t r = rgb[src+0];
    uint8_t g = rgb[src+1];
    uint8_t b = rgb[src+2];
    int y0 = y * scale;
    int x0 = x * scale;
    for(int dy=0; dy<scale; ++dy){
      int row = (y0 + dy) * out_w;
      for(int dx=0; dx<scale; ++dx){
        int dst = 3*(row + (x0 + dx));
        rgb_big[dst+0] = r;
        rgb_big[dst+1] = g;
        rgb_big[dst+2] = b;
      }
    }
  }
}

// Draw 1px gridlines between tiles (after scaling) for readability
// vertical lines between columns
for(int tc=1; tc<tiles_per_row; ++tc){
  int xline = tc * tile_size * scale;
  for(int y=0; y<out_h; ++y){
    int idx = 3*(y*out_w + xline);
    rgb_big[idx+0] = 0; rgb_big[idx+1] = 0; rgb_big[idx+2] = 0;
  }
}
// horizontal lines between rows
for(int tr=1; tr<tiles_per_col; ++tr){
  int yline = tr * tile_size * scale;
  int base = 3*(yline * out_w);
  for(int x=0; x<out_w; ++x){
    int idx = base + 3*x;
    rgb_big[idx+0] = 0; rgb_big[idx+1] = 0; rgb_big[idx+2] = 0;
  }
}

write_png("report_connection_fields.png", rgb_big.data(), out_w, out_h);

}


void visualize_outputs(void){

q_display_trajectory=1;

int need_c=4;
int need_w=4;

// Store the selected 8 examples so we can generate BOTH pages
std::vector<int> sel_digit;
std::vector<int> sel_truth;
std::vector<int> sel_pred;
std::vector<int> sel_correct;
std::vector< std::vector<double> > sel_outs;

ofstream html("report_visualize_trajectories.html", ios::out);

html << "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n";
html << "<title>Trajectory visualization</title>\n";

html << "<style>\n";
html << "body{font-family:system-ui, sans-serif; margin:16px;}\n";
html << ".grid{display:grid; grid-template-columns:repeat(2,1fr); gap:14px;}\n";
html << ".tile{border:1px solid #ccc; border-radius:8px; padding:10px;}\n";
html << ".ok{background:#e8f5ee;}\n";
html << ".bad{background:#fdecec;}\n";
html << ".hdr{font-size:14px; margin-bottom:8px;}\n";
html << ".row{display:flex; gap:10px; align-items:flex-start;}\n";
html << ".mn{width:140px; image-rendering:pixelated; border:1px solid #ddd; flex:0 0 auto;}\n";
html << ".plot{flex:1 1 auto;}\n";
html << ".note{font-size:12px; color:#333; margin-top:6px;}\n";
html << "</style>\n</head>\n<body>\n";

html << "<h2>8 digits with output trajectories</h2>\n";
html << "<div class=\"grid\">\n";

// Simple palette for 10 traces
const char* col[10] = {
"#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
"#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
};

for(int i=0; (need_c>0 || need_w>0) && i<number_of_test_digits; i++){

    set_test_input(i);

    // cheap run to short-circuit when a bucket is already full
    q_display_trajectory=0;
    run_trajectory();

    int offset=number_of_input_spins+number_of_hidden_spins;
    int truth=digit_type[chosen_digit];

    int pred0=0;
    double best0=spin[offset];
    for(int j=1;j<number_of_digit_classes;j++){
        if(spin[offset+j]>best0){best0=spin[offset+j];pred0=j;}
    }
    int correct0=(pred0==truth);

    if(correct0 && need_c<=0) continue;
    if(!correct0 && need_w<=0) continue;

    // plotted run: produces report_output_<chosen_digit>_<truth>_0.dat
    set_test_input(i);
    q_display_trajectory=1;
    run_trajectory();

    // prediction from the plotted run
    int pred=0;
    double best=spin[offset];
    for(int j=1;j<number_of_digit_classes;j++){
        if(spin[offset+j]>best){best=spin[offset+j];pred=j;}
    }
    int correct=(pred==truth);

    // enforce exact 4 correct + 4 incorrect based on the plotted run
    if(correct && need_c<=0){
        snprintf(st,sizeof(st),
                 "rm -f report_output_%d_%d_%d.dat",
                 chosen_digit, truth, 0);
        system(st);
        continue;
    }
    if(!correct && need_w<=0){
        snprintf(st,sizeof(st),
                 "rm -f report_output_%d_%d_%d.dat",
                 chosen_digit, truth, 0);
        system(st);
        continue;
    }

    // MNIST picture
    make_mnist_picture(chosen_digit);

    // Store this example for the outputs bar page
    {
        sel_digit.push_back(chosen_digit);
        sel_truth.push_back(truth);
        sel_pred.push_back(pred);
        sel_correct.push_back(correct);
        std::vector<double> outs(10,0.0);
        for(int j=0;j<number_of_digit_classes;j++) outs[j]=spin[offset+j];
        sel_outs.push_back(outs);
    }

    // Load trajectory file
    char fname[256];
    snprintf(fname, sizeof(fname), "report_output_%d_%d_%d.dat", chosen_digit, truth, 0);

    std::vector<double> tt;
    std::vector<double> yy[10];

    {
        ifstream in(fname, ios::in);
        if(!in){
            cerr << "could not open " << fname << " for reading" << endl;
        }
        else{
            while(true){
                double t;
                if(!(in >> t)) break;
                tt.push_back(t);
                for(int j=0;j<number_of_digit_classes;j++){
                    double v;
                    in >> v;
                    yy[j].push_back(v);
                }
            }
        }
    }

    // Determine plot ranges
    double tmin=0.0, tmax=1.0;
    if(tt.size()>0){ tmin=tt.front(); tmax=tt.back(); }

    double ymin=0.0, ymax=0.0;
    int have_y=0;
    for(int j=0;j<number_of_digit_classes;j++){
        for(size_t k=0;k<yy[j].size();k++){
            double v=yy[j][k];
            if(!have_y){ ymin=v; ymax=v; have_y=1; }
            else{ if(v<ymin) ymin=v; if(v>ymax) ymax=v; }
        }
    }
    if(!have_y){ ymin=-1.0; ymax=1.0; }
    if(fabs(ymax-ymin)<1e-12){ ymax=ymin+1.0; }

    // SVG geometry
    const int W=520;
    const int H=220;
    const int L=40;
    const int R=12;
    const int T=12;
    const int B=26;

    // Precompute affine maps (avoid C++11 lambdas)
    double dx_t = (tmax - tmin);
    if(dx_t <= 0.0) dx_t = 1.0;
    double dy_y = (ymax - ymin);
    if(fabs(dy_y) < 1e-12) dy_y = 1.0;

    double xscale = (W - L - R) / dx_t;
    double yscale = (H - T - B) / dy_y;

    // write tile
    html << "<div class=\"tile " << (correct?"ok":"bad") << "\">\n";

    html << "<div class=\"hdr\">";
    html << "<b>digit</b> " << chosen_digit
         << " &nbsp; class <b>" << truth
         << "</b> &nbsp; pred <b>" << pred << "</b>";
    html << "</div>\n";

    html << "<div class=\"row\">\n";
    html << "<img class=\"mn\" src=\"report_digit_" << chosen_digit << ".png\">\n";

    html << "<div class=\"plot\">\n";

    // SVG plot
    html << "<svg width=\"" << W << "\" height=\"" << H << "\" viewBox=\"0 0 " << W << " " << H << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

    // axes box
    html << "<rect x=\"" << L << "\" y=\"" << T << "\" width=\"" << (W-L-R) << "\" height=\"" << (H-T-B) << "\" fill=\"white\" stroke=\"black\" stroke-width=\"1\"/>\n";

    // sparse y-axis ticks and labels
    {
     double yticks[3];
int nt=0;
yticks[nt++]=ymin;
yticks[nt++]=0.5*(ymin+ymax);
yticks[nt++]=ymax;

        // draw ticks (left side)
        for(int ti=0; ti<nt; ti++){
            double yv=yticks[ti];
            double yp = T + yscale * (ymax - yv);

            // tick mark
            html << "<line x1=\"" << (L-4) << "\" y1=\"" << yp << "\" x2=\"" << L << "\" y2=\"" << yp << "\" stroke=\"#000\" stroke-width=\"1\"/>\n";

            // label
            html << "<text x=\"" << (L-6) << "\" y=\"" << (yp+4) << "\" font-size=\"11\" text-anchor=\"end\">";
            html << std::setprecision(3) << yv;
            html << "</text>\n";
        }
    }

    // y=0 line if visible
    if(ymin < 0.0 && ymax > 0.0){
        double y0 = T + yscale * (ymax - 0.0);
        html << "<line x1=\"" << L << "\" y1=\"" << y0 << "\" x2=\"" << (W-R) << "\" y2=\"" << y0 << "\" stroke=\"#000\" stroke-width=\"1\" stroke-dasharray=\"4,4\"/>\n";
    }

    // traces
    for(int j=0;j<number_of_digit_classes;j++){

        if(yy[j].size()<2) continue;

        html << "<polyline fill=\"none\" stroke=\"" << col[j] << "\" stroke-width=\"1\" points=\"";

        for(size_t k=0;k<yy[j].size() && k<tt.size();k++){
            double px = L + xscale * (tt[k] - tmin);
            double py = T + yscale * (ymax - yy[j][k]);
            html << px << "," << py;
            if(k+1<yy[j].size() && k+1<tt.size()) html << " ";
        }

        html << "\"/>\n";
    }

    // simple labels
    html << "<text x=\"" << L << "\" y=\"" << (H-6) << "\" font-size=\"12\">t</text>\n";

    // legend (0..9)
    int lx0=L;
    int ly0=H-B+16;
    for(int j=0;j<number_of_digit_classes;j++){
        int x0=lx0 + j*48;
        if(x0+40>W-R) break;
        html << "<rect x=\"" << x0 << "\" y=\"" << (ly0-10) << "\" width=\"10\" height=\"10\" fill=\"" << col[j] << "\"/>\n";
        html << "<text x=\"" << (x0+14) << "\" y=\"" << (ly0-1) << "\" font-size=\"12\">" << j << "</text>\n";
    }

    html << "</svg>\n";

    html << "<div class=\"note\">";
    html << "File: " << fname;
    html << "</div>\n";

    html << "</div>\n"; // plot
    html << "</div>\n"; // row
    html << "</div>\n"; // tile

    if(correct) need_c--;
    else need_w--;
}

html << "</div>\n</body>\n</html>\n";
html.close();

// --- Second page: final-time outputs as signed bars, for the SAME selected digits ---
{
    ofstream out("report_visualize_outputs.html", ios::out);

    out << "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n";
    out << "<title>Outputs visualization</title>\n";

    out << "<style>\n";
    out << "body{font-family:system-ui, sans-serif; margin:16px;}\n";
    out << ".grid{display:grid; grid-template-columns:repeat(4,1fr); gap:14px;}\n";
    out << ".tile{border:1px solid #ccc; border-radius:8px; padding:10px;}\n";
    out << ".ok{background:#e8f5ee;}\n";
    out << ".bad{background:#fdecec;}\n";
    out << ".hdr{font-size:14px; margin-bottom:8px;}\n";
    out << ".row{display:flex; gap:10px; align-items:flex-start;}\n";
    out << ".mn{width:140px; image-rendering:pixelated; border:1px solid #ddd;}\n";

    out << ".bars{flex:1;}\n";
    out << ".barrow{display:flex; align-items:center; gap:6px; margin:3px 0;}\n";
    out << ".lab{width:18px; font-size:12px; text-align:right;}\n";

    out << ".barbox{position:relative; width:220px; height:14px; background:#f3f3f3;}\n";
    out << ".axis{position:absolute; left:110px; top:0; bottom:0; width:1px; background:#000;}\n";
    out << ".barpos{position:absolute; left:110px; top:0; bottom:0; background:#4a90e2;}\n";
    out << ".barneg{position:absolute; right:110px; top:0; bottom:0; background:#d9534f;}\n";

    out << ".val{width:70px; font-size:12px; text-align:right;}\n";
    out << "</style>\n</head>\n<body>\n";

    out << "<h2>8 digits with signed output bars</h2>\n";
    out << "<div class=\"grid\">\n";

    for(size_t idx=0; idx<sel_digit.size(); idx++){

        int dig = sel_digit[idx];
        int truth = sel_truth[idx];
        int pred = sel_pred[idx];
        int correct = sel_correct[idx];

        const std::vector<double>& outs = sel_outs[idx];

        // scale bars using max absolute output
        double vmax=0.0;
        for(int j=0;j<number_of_digit_classes;j++){
            double a=fabs(outs[j]);
            if(a>vmax) vmax=a;
        }
        if(vmax<1e-12) vmax=1.0;

        const double half_width=110.0;

        out << "<div class=\"tile " << (correct?"ok":"bad") << "\">\n";

        out << "<div class=\"hdr\">";
        out << "<b>digit</b> " << dig
            << " &nbsp; class <b>" << truth
            << "</b> &nbsp; pred <b>" << pred << "</b></div>\n";

        out << "<div class=\"row\">\n";
        out << "<img class=\"mn\" src=\"report_digit_" << dig << ".png\">\n";
        out << "<div class=\"bars\">\n";

        for(int j=0;j<number_of_digit_classes;j++){

            double v=outs[j];
            double w=fabs(v)/vmax;
            int px=(int)(half_width*w + 0.5);

            out << "<div class=\"barrow\">";
            out << "<div class=\"lab\">" << j << "</div>";

            out << "<div class=\"barbox\">";
            out << "<div class=\"axis\"></div>";

            if(v>=0.0)
                out << "<div class=\"barpos\" style=\"width:" << px << "px;\"></div>";
            else
                out << "<div class=\"barneg\" style=\"width:" << px << "px;\"></div>";

            out << "</div>";

            out << "<div class=\"val\">" << std::setprecision(6) << v << "</div>";
            out << "</div>\n";
        }

        out << "</div></div></div>\n";
    }

    out << "</div>\n</body>\n</html>\n";
    out.close();
}

q_display_trajectory=0;

cout << "wrote report_visualize_trajectories.html (time series)." << endl;
cout << "wrote report_visualize_outputs.html (signed bars)." << endl;

}
int neighbor(int s1, int j){

//return neighbor j of spin s1
//left-down-right-up on periodic grid of side lx
//consult read_mnist for relation of pixel- and lattice sites

 int x = s1 % lx;
 int y = s1 / lx;
 int nx = x;  // default to self (guards out-of-range j)
 int ny = y;

if(j == 0){nx = (x - 1 + lx) % lx; ny = y;}
if(j == 1){ny = (y + 1) % lx; nx = x;}
if(j == 2){nx = (x + 1) % lx; ny = y;}
if(j == 3){ny = (y - 1 + lx) % lx; nx = x;}

return (ny * lx + nx);

}
#include <cmath>
#include <cstdint>

// write_png(char const* path, const uint8_t* rgb, int w, int h) assumed available
// Globals assumed available (names match your codebase):
// digit, digit_type, number_of_input_spins, number_of_hidden_spins,
// number_of_digit_classes, test_set,
// run_trajectory(), spin[], chosen_digit, q_display_trajectory.


void read_parameters(void){

cout << "computer parameters read in " << endl;

snprintf(st, sizeof(st), "parameters.dat");
ifstream in(st, ios::in);

if(!in){cerr << "could not open " << st << " for reading" << endl;exit(1);}
  
for(int k=0; k<number_of_parameters;k++){
if(!(in >> parameters[k])){cerr << "unexpected EOF while reading " << st << " at index k=" << k << endl;exit(1);
}}

in.close();

}
