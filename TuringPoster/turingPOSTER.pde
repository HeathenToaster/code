Cell[][] grid;
Cell[][] prev;

float dA = 1; //1.0
float dB = 0.2; //0.5
float feed = 0.055; //0.055
float k = 0.062; //0.062

void setup() {
  size(1189, 846);
  grid = new Cell[width][height];
  prev = new Cell[width][height];
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j ++) {
      float a = 1;
      float b = 0;
      grid[i][j] = new Cell(a, b);
      prev[i][j] = new Cell(a, b);
    }
  }
  for (int n = 0; n < 5000; n++) {
    int startx = int(random(2, width-12));
    int starty = int(random(2, height-17));
    for (int i = startx; i < startx+10; i++) {
      for (int j = starty; j < starty+10; j ++) {
        float a = 1;
        float b = 1;
        grid[i][j] = new Cell(a, b);
        prev[i][j] = new Cell(a, b);
      }
    }
  }
}


class Cell {
  float a;
  float b;
  Cell(float a_, float b_) {
    a = a_;
    b = b_;
  }
}



void erase(int xa, int ya, int xb, int yb) {
  for (int i = xa; i < xb; i++) {
    for (int j = ya; j < yb; j ++) {
      Cell newspot = grid[i][j];
      newspot.a = 1;
      newspot.b = 0;
    }
  }
}



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
      newspot.b = constrain(newspot.b, 0, 1);
    }
  }
}


void swap() {
  Cell[][] temp = prev;
  prev = grid;
  grid = temp;
}


int c = 0;
int border = 50;
void draw() {
  for (int i = 0; i < 1; i++) {
    println(c, frameRate);
    c +=1;
    
    update(1,              border+1,       1,               height-6);
    update(width-border-1, width-1,        1,               height-6);
    update(border+1,       width-border-1, 1,               border+1);
    update(border+1,       width-border-1, 3*border+1,      4*border+1);    
    update(border+1,       width-border-1, 11*border+1,     12*border+1);
    update(border+1,       width-border-1, height-border-1, height-6);
    update(926,            976,            4*border+1,      11*border+1);
    update(926,            976,            12*border+1,     height-border-1);
    erase(border+1, border+1, width-border-1, 3*border+1);
    erase(border+1, 4*border+1, 926, 11*border+1);
    erase(976, 4*border+1, width-border-1, 11*border+1);
    erase(border+1, 12*border+1, 926, height-border-1);
    erase(976, 12*border+1, width-border-1, height-border-1);
 
    swap();
    
  }
  loadPixels();
  color white = color(255, 255, 255);
  color centuri_Blue = color(0, 155, 195);
  for (int i = 1; i < width-1; i++) {
    for (int j = 1; j < height-1; j ++) {
      Cell spot = grid[i][j];
      float a = spot.a;
      float b = spot.b;
      int pos = i + j * width;
      //pixels[pos] = color((a-b)*255);
      pixels[pos] = lerpColor(centuri_Blue, white, constrain(exp(a-b)-1, 0, 1)); // add alpha// test other than sq(), exp()
    }
  }
  updatePixels();
  
  if (mousePressed){ //maybe change to key press and mouseclick == add/remove A/B
    int yy = year();
    int mt = month();
    int dd = day();
    int hh = hour();
    int mm = minute();
    int ss = second();
    String time = yy+"_"+mt+"_"+dd+"_"+hh+"_"+mm+"_"+ss;
    saveFrame(time + "turing.png");
    print("screenshot");
  }
}

//formula round_corners(xcorner, ycorner, radius, quadrant)
