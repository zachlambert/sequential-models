#ifndef CPP_ML_HMM_H
#define CPP_ML_HMM_H

#include <chrono>
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>

class Hmm {
public:
    virtual double p0(double x)=0;
    virtual double fn(double x_prev, double x, int n)=0;
    virtual double gn(double x, double y, int n)=0;
};


class HmmGaussian: public Hmm {
public:
    HmmGaussian(double a, double b, double c, double d);

private:
    double p0(double x);
    double fn(double x_prev, double x, int n);
    double gn(double x, double y, int n);

    double a, b, c, d;
};


#endif
