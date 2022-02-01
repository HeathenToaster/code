// xy args not same order in every function // should be better but recheck
// fix blob spawning in corners
// add previsualiser, with lines delimiting each zone, so can check if the plot fits before waiting 45min 
// try that someday:  https://sighack.com/post/exporting-high-resolution-images-in-processing
Cell[][] grid;
Cell[][] prev;

float dA = 1; //1.0
float dB = 0.33; //0.5
float feed = 0.055; //0.055
float k = 0.062; //0.062
int NbBlobs = 250000; // test = 10, Large POSTER = 250000

void setup() {
  size(8410, 2380);
  //size(1200, 500);
  frameRate(120); // largely over so not capped at 60 fps/ doesn't really matter but can gain a bit of time if testing in small res. 
  //size(8410, 2380); /// intro: x = sheet width (safe zone is in border), y = bounding box height +100
  //size(8410, 2500); /// matmet
  //size(8410, 4750); /// RES
  grid = new Cell[width][height];
  prev = new Cell[width][height];
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j ++) {
      float a = 1;
      float b = 0;
      grid[i][j] = new Cell(a, b);
      prev[i][j] = new Cell(a, b);}}
  for (int n = 0; n < NbBlobs; n++) { 
    int startx = int(random(border+10, width-border-10)); // poster : int startx = int(random(110, width-110)); /pb: angles are in range of spawn
    int starty = int(random(border+10, height-border-10)); // poster: int starty = int(random(60, height-60));
    for (int i = startx; i < startx+10; i++) {
      for (int j = starty; j < starty+10; j ++) {
        float a = 1;
        float b = 1;
        grid[i][j] = new Cell(a, b);
        prev[i][j] = new Cell(a, b);}}}}

class Cell {
  float a;
  float b;
  Cell(float a_, float b_) {
    a = a_;
    b = b_;}}
    
void erase(int imin, int imax, int jmin, int jmax) {
  for (int i = imin; i < imax; i++) {
    for (int j = jmin; j < jmax; j ++) {
      Cell newspot = grid[i][j];
      newspot.a = 1;
      newspot.b = 0;}}}

void update(int imin, int imax, int jmin, int jmax) {
  for (int i = imin; i < imax; i++) {
    for (int j = jmin; j < jmax; j ++) {
      Cell spot = prev[i][j];
      Cell newspot = grid[i][j];
      float a = spot.a;
      float b = spot.b;
      float laplaceA = 0;
      laplaceA += a*-1;
      laplaceA += prev[i+1][j].a*0.2;
      laplaceA += prev[i-1][j].a*0.2;
      laplaceA += prev[i][j+1].a*0.2;
      laplaceA += prev[i][j-1].a*0.2;
      laplaceA += prev[i-1][j-1].a*0.05;
      laplaceA += prev[i+1][j-1].a*0.05;
      laplaceA += prev[i-1][j+1].a*0.05;
      laplaceA += prev[i+1][j+1].a*0.05;
      float laplaceB = 0;
      laplaceB += b*-1;
      laplaceB += prev[i+1][j].b*0.2;
      laplaceB += prev[i-1][j].b*0.2;
      laplaceB += prev[i][j+1].b*0.2;
      laplaceB += prev[i][j-1].b*0.2;
      laplaceB += prev[i-1][j-1].b*0.05;
      laplaceB += prev[i+1][j-1].b*0.05;
      laplaceB += prev[i-1][j+1].b*0.05;
      laplaceB += prev[i+1][j+1].b*0.05;
      newspot.a = a + (dA*laplaceA - a*b*b + feed*(1-a))*1;
      newspot.b = b + (dB*laplaceB + a*b*b - (k+feed)*b)*1;
      newspot.a = constrain(newspot.a, 0, 1);
      newspot.b = constrain(newspot.b, 0, 1);}}}


void circleeraseq(int x, int y, int r, String q) {
  if (q.equals("NO") || q.equals("SO")) {
      for (int i = y-r; i < y; i++) {
        if (q.equals("NO")) {
          for (int j = x; (j-x)*(j-x) + (i-y)*(i-y) <= r*r; j--) {
                  Cell newspot = grid[i][j];
                  newspot.a = 1;
                  newspot.b = 0;}}
        if (q.equals("SO")) {
          for (int j = x; (j-x)*(j-x) + (i-y)*(i-y) <= r*r; j++) {
                  Cell newspot = grid[i][j];
                  newspot.a = 1;
                  newspot.b = 0;}}}}
  if (q.equals("NE") || q.equals("SE")) {
      for (int i = y; i < y+r; i++) {
        if (q.equals("NE")) {
          for (int j = x; (j-x)*(j-x) + (i-y)*(i-y) <= r*r; j--) {
                  Cell newspot = grid[i][j];
                  newspot.a = 1;
                  newspot.b = 0;}}
        if (q.equals("SE")) {
          for (int j = x; (j-x)*(j-x) + (i-y)*(i-y) <= r*r; j++) {
                  Cell newspot = grid[i][j];
                  newspot.a = 1;
                  newspot.b = 0;}}}}}

void circleupdateq(int x, int y, int r, String q) {
  if (q.equals("NO") || q.equals("SO")) {
      for (int i = y-r; i < y; i++) {
        if (q.equals("NO")) {
          for (int j = x; (j-x)*(j-x) + (i-y)*(i-y) <= r*r; j--) {
            Cell spot = prev[i][j];
            Cell newspot = grid[i][j];
            float a = spot.a;
            float b = spot.b;
            float laplaceA = 0;
            laplaceA += a*-1;
            laplaceA += prev[i+1][j].a*0.2;
            laplaceA += prev[i-1][j].a*0.2;
            laplaceA += prev[i][j+1].a*0.2;
            laplaceA += prev[i][j-1].a*0.2;
            laplaceA += prev[i-1][j-1].a*0.05;
            laplaceA += prev[i+1][j-1].a*0.05;
            laplaceA += prev[i-1][j+1].a*0.05;
            laplaceA += prev[i+1][j+1].a*0.05;
            float laplaceB = 0;
            laplaceB += b*-1;
            laplaceB += prev[i+1][j].b*0.2;
            laplaceB += prev[i-1][j].b*0.2;
            laplaceB += prev[i][j+1].b*0.2;
            laplaceB += prev[i][j-1].b*0.2;
            laplaceB += prev[i-1][j-1].b*0.05;
            laplaceB += prev[i+1][j-1].b*0.05;
            laplaceB += prev[i-1][j+1].b*0.05;
            laplaceB += prev[i+1][j+1].b*0.05;
            newspot.a = a + (dA*laplaceA - a*b*b + feed*(1-a))*1;
            newspot.b = b + (dB*laplaceB + a*b*b - (k+feed)*b)*1;
            newspot.a = constrain(newspot.a, 0, 1);
            newspot.b = constrain(newspot.b, 0, 1);}}
        if (q.equals("SO")) {
          for (int j = x; (j-x)*(j-x) + (i-y)*(i-y) <= r*r; j++) {
            Cell spot = prev[i][j];
            Cell newspot = grid[i][j];
            float a = spot.a;
            float b = spot.b;
            float laplaceA = 0;
            laplaceA += a*-1;
            laplaceA += prev[i+1][j].a*0.2;
            laplaceA += prev[i-1][j].a*0.2;
            laplaceA += prev[i][j+1].a*0.2;
            laplaceA += prev[i][j-1].a*0.2;
            laplaceA += prev[i-1][j-1].a*0.05;
            laplaceA += prev[i+1][j-1].a*0.05;
            laplaceA += prev[i-1][j+1].a*0.05;
            laplaceA += prev[i+1][j+1].a*0.05;
            float laplaceB = 0;
            laplaceB += b*-1;
            laplaceB += prev[i+1][j].b*0.2;
            laplaceB += prev[i-1][j].b*0.2;
            laplaceB += prev[i][j+1].b*0.2;
            laplaceB += prev[i][j-1].b*0.2;
            laplaceB += prev[i-1][j-1].b*0.05;
            laplaceB += prev[i+1][j-1].b*0.05;
            laplaceB += prev[i-1][j+1].b*0.05;
            laplaceB += prev[i+1][j+1].b*0.05;
            newspot.a = a + (dA*laplaceA - a*b*b + feed*(1-a))*1;
            newspot.b = b + (dB*laplaceB + a*b*b - (k+feed)*b)*1;
            newspot.a = constrain(newspot.a, 0, 1);
            newspot.b = constrain(newspot.b, 0, 1);}}}}
  if (q.equals("NE") || q.equals("SE")) {
      for (int i = y; i < y+r; i++) {
        if (q.equals("NE")) {
          for (int j = x; (j-x)*(j-x) + (i-y)*(i-y) <= r*r; j--) {
            Cell spot = prev[i][j];
            Cell newspot = grid[i][j];
            float a = spot.a;
            float b = spot.b;
            float laplaceA = 0;
            laplaceA += a*-1;
            laplaceA += prev[i+1][j].a*0.2;
            laplaceA += prev[i-1][j].a*0.2;
            laplaceA += prev[i][j+1].a*0.2;
            laplaceA += prev[i][j-1].a*0.2;
            laplaceA += prev[i-1][j-1].a*0.05;
            laplaceA += prev[i+1][j-1].a*0.05;
            laplaceA += prev[i-1][j+1].a*0.05;
            laplaceA += prev[i+1][j+1].a*0.05;
            float laplaceB = 0;
            laplaceB += b*-1;
            laplaceB += prev[i+1][j].b*0.2;
            laplaceB += prev[i-1][j].b*0.2;
            laplaceB += prev[i][j+1].b*0.2;
            laplaceB += prev[i][j-1].b*0.2;
            laplaceB += prev[i-1][j-1].b*0.05;
            laplaceB += prev[i+1][j-1].b*0.05;
            laplaceB += prev[i-1][j+1].b*0.05;
            laplaceB += prev[i+1][j+1].b*0.05;
            newspot.a = a + (dA*laplaceA - a*b*b + feed*(1-a))*1;
            newspot.b = b + (dB*laplaceB + a*b*b - (k+feed)*b)*1;
            newspot.a = constrain(newspot.a, 0, 1);
            newspot.b = constrain(newspot.b, 0, 1);}}
        if (q.equals("SE")) {
          for (int j = x; (j-x)*(j-x) + (i-y)*(i-y) <= r*r; j++) {
            Cell spot = prev[i][j];
            Cell newspot = grid[i][j];
            float a = spot.a;
            float b = spot.b;
            float laplaceA = 0;
            laplaceA += a*-1;
            laplaceA += prev[i+1][j].a*0.2;
            laplaceA += prev[i-1][j].a*0.2;
            laplaceA += prev[i][j+1].a*0.2;
            laplaceA += prev[i][j-1].a*0.2;
            laplaceA += prev[i-1][j-1].a*0.05;
            laplaceA += prev[i+1][j-1].a*0.05;
            laplaceA += prev[i-1][j+1].a*0.05;
            laplaceA += prev[i+1][j+1].a*0.05;
            float laplaceB = 0;
            laplaceB += b*-1;
            laplaceB += prev[i+1][j].b*0.2;
            laplaceB += prev[i-1][j].b*0.2;
            laplaceB += prev[i][j+1].b*0.2;
            laplaceB += prev[i][j-1].b*0.2;
            laplaceB += prev[i-1][j-1].b*0.05;
            laplaceB += prev[i+1][j-1].b*0.05;
            laplaceB += prev[i-1][j+1].b*0.05;
            laplaceB += prev[i+1][j+1].b*0.05;
            newspot.a = a + (dA*laplaceA - a*b*b + feed*(1-a))*1;
            newspot.b = b + (dB*laplaceB + a*b*b - (k+feed)*b)*1;
            newspot.a = constrain(newspot.a, 0, 1);
            newspot.b = constrain(newspot.b, 0, 1);}}}}}
         
void eraseroundedbox(int xa, int xb, int ya, int yb, int r) {        
    erase(xa + r, xb - r, ya,     yb    );                       //|  
    erase(xa,     xa + r, ya + r, yb - r);                       //|  
    erase(xb - r, xb,     ya + r, yb - r);                       //|  
    circleeraseq(ya + r, xa + r, r, "NO");                       //|  
    circleeraseq(ya + r, xb - r, r, "NE");                       //|  
    circleeraseq(yb - r, xa + r, r, "SO");                       //|  
    circleeraseq(yb - r, xb - r, r, "SE");}                      //|  
    
void erasetitlebox(int xa, int xb, int ya, int yb, int r) {      //|poster title box
    erase(xa + r, xb - r, ya,     yb    );                       //|  
    erase(xa,     xa + r, ya + r, yb - r);                       //|  
    erase(xb - r, xb,     ya + r, yb - r);                       //|  
    circleeraseq(yb - r, xa + r, r, "SO");                       //|  
    circleeraseq(yb - r, xb - r, r, "SE");}                      //|  
    
void updateroundedbox(int xa, int xb, int ya, int yb, int r) {        
    update(xa + r, xb - r, ya,     yb    );                       //|  
    update(xa,     xa + r, ya + r, yb - r);                       //|  
    update(xb - r, xb,     ya + r, yb - r);                       //|  
    circleupdateq(ya + r, xa + r, r, "NO");                       //|  
    circleupdateq(ya + r, xb - r, r, "NE");                       //|  
    circleupdateq(yb - r, xa + r, r, "SO");                       //|  
    circleupdateq(yb - r, xb - r, r, "SE");}                      //|  

void updateroundedbox2(int xa, int ya, int xb, int yb, int r) {
    update(xa + r, ya,     xb - r, yb    );
    update(xa,     ya + r, xa + r, yb - r);
    update(xb - r, ya + r, xb,     yb - r);
    circleupdateq(ya + r, xa + r, r, "NO");
    circleupdateq(ya + r, xb - r, r, "NE");
    circleupdateq(yb - r, xa + r, r, "SO");
    circleupdateq(yb - r, xb - r, r, "SE");}
    
void updatecircle(int xa, int ya, int r) {
    circleupdateq(ya, xa, r, "NO");
    circleupdateq(ya, xa, r, "NE");
    circleupdateq(ya, xa, r, "SO");
    circleupdateq(ya, xa, r, "SE");}

void swap() {
  Cell[][] temp = prev;
  prev = grid;
  grid = temp;}
  

int c = 0;
int border = 50;
int radius = 50;
void draw() {
  for (int i = 0; i < 1; i++) {
    println(c, frameRate);
    c +=1;
    
    //tests/debug
    //updateroundedbox(border, width - border, border, height-border, 1); 
    //updatecircle(250, 250, 100);
    //updateroundedbox(25, 1025, 25, 425, 200); 
    
    
    //INTRO POSTER
    //updateroundedbox(100, 8410-100, border, 2380-border, 2*radius); 
    //eraseroundedbox(150,  1600, 2380-50-2080-border, 2380-50-border, radius);
    //eraseroundedbox(1650, 4250, 2380-50-2080-border, 2380-50-border, radius);
    //eraseroundedbox(4300, 6250, 2380-50-2080-border, 2380-50-border, radius);
    //eraseroundedbox(6300, 8260, 2380-50-2080-border, 2380-50-border, radius);
    //erasetitlebox(200,  2150, 0, border+100, radius);    
    
    //MatMet POSTER
    //updateroundedbox(100, 8410-100, border, 2500-border, 2*radius); 
    //eraseroundedbox(150,  2550, 2500-50-2200-border, 2500-50-border, radius);
    //eraseroundedbox(2650, 8260, 2500-50-2200-border, 2500-50-border, radius);
    //erasetitlebox(200,  2150, 0, border+100, radius);   
    
    //Res POSTER
    //updateroundedbox(100, 8410-100, border, 4750-border, 2*radius); 
    //eraseroundedbox(150,  2650, 4750-50-4450-border, 4750-50-border, radius);
    //eraseroundedbox(2705, 5205, 4750-50-4450-border, 4750-50-border, radius);
    //eraseroundedbox(5260, 8260, 4750-50-4450-border, 4750-50-border, radius);
    //erasetitlebox(200,  2150, 0, border+100, radius);    
    
    //TITLE OUTLINE 
    updateroundedbox(50,  8410-50, border, 2380-border, 2*radius); 
    eraseroundedbox(100,  8410-100, 50+border, 2380-50-border, radius);


    
    swap();}
    
  loadPixels();
  color white = color(255, 255, 255);
  color centuri_Blue = color(0, 155, 195);
  for (int i = 1; i < width-1; i++) {
    for (int j = 1; j < height-1; j ++) {
      Cell spot = grid[i][j];
      float a = spot.a;
      float b = spot.b;
      int pos = i + j * width;
      float hh = 1;
      float slope = -5;
      float p = 0.2;
      //float grad = (hh / (1 + exp(slope*((a-b)-p))));
      pixels[pos] = lerpColor(centuri_Blue, white, (hh / (1 + exp(slope*((a-b)-p)))));}}

  updatePixels();
  //saveFrame("frames/frame#####.png"); //save each frame to make gif
  if (mousePressed){ //maybe change to key press and mouseclick == add/remove A/B
    int yy = year();
    int mt = month();
    int dd = day();
    int hh = hour();
    int mm = minute();
    int ss = second();
    String time = yy+"_"+mt+"_"+dd+"_"+hh+"_"+mm+"_"+ss;
    saveFrame(time + "turing.png");
    print("screenshot");}}
