set xlabel "n"
plot "output.dat" using 1:2 title "x" with lines, \
     "output.dat" using 1:3 title "y"
