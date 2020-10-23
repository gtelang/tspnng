size(5cm);
settings.outformat = "pdf";

pair p=(0,1);
pair q=(0,-1);

pair ln = (-3.5,1.5);
pair rn = (3.5,1.5);

pair ls = (-3.5,-1.5);
pair rs = (3.5,-1.5);


draw(p--q,1.5+blue);
draw(ln--p--rn--rs--q--ls--cycle, 1+gray);
int dotsz=8;
dot(p,dotsz+black);
dot(q,dotsz+black);
dot(ln,dotsz+black);
dot(rn,dotsz+black);
dot(ls,dotsz+black);
dot(rs,dotsz+black);
