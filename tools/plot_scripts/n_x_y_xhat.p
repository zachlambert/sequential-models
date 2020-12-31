set xlabel "n"
set ylabel "x"
plot "output.dat" using 1:2 title "x" with lines, \
     "output.dat" using 1:3 title "y", \
     "output.dat" using 1:4 title "x hat" with lines
