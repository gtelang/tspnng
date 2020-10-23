size(5cm);
settings.outformat = "pdf";


for (int i=-3 ;  i<= 2 ; ++i){
      draw((2*i,0)--(2*i+2,0), gray);
      draw((2*i,1)--(2*i+2,1),gray);
}

real l = 3;
draw((-2*l,0)--(-2*l,1),gray);
draw((2*l,0)--(2*l,1),gray);


draw((-2*l,1){(-1,1)}..{(1,1)}(-2*l,0),red);
draw((2*l,1){(1,1)}..{(-1,1)}(2*l,0),red);

for (int i=-2 ;  i<= 2 ; ++i){

      draw((2*i,0)--(2*i,1),red);
      dot((2*i,1));
      dot((2*i,0));
}




dot((2*l,1));
dot((2*l,0));
dot((-2*l,1));
dot((-2*l,0));
