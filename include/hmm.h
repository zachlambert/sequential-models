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

class HmmSampler {
public:
    HmmSampler(): n(-1), xn(0), yn(0) {}
    void initialise()
    {
        n = 0;
        xn = sample_p0();
        yn = sample_gn(xn, n);
    }
    void step()
    {
        n++;
        xn = sample_fn(xn, n);
        yn = sample_gn(xn, n);
    }
    int n;
    double xn;
    double yn;

protected:
    virtual double sample_p0() = 0;
    virtual double sample_fn(double xn_prev, int n) = 0;
    virtual double sample_gn(double xn, int n) = 0;
};

class HmmGaussian: public Hmm {
public:
    HmmGaussian(double a, double b, double c, double d):
        a(a), b(b), c(c), d(d) {}
    double p0(double x)
    {
        return (1/std::sqrt(2*M_PI)*b)*
            std::exp(-0.5*std::pow(x/b, 2));
    }
    double fn(double x_prev, double x, int n)
    {
        // n unused - fn constant with n
        return (1/std::sqrt(2*M_PI)*b)*
            std::exp(-0.5*std::pow((x - a*x_prev)/b, 2));
    }
    double gn(double x, double y, int n)
    {
        // n unused - gn constant with n
        return (1/std::sqrt(2*M_PI)*d)*
            std::exp(-0.5*std::pow((y - c*x)/d, 2));
    }
private:
    double a, b, c, d;
};

class HmmSamplerGaussian: public HmmSampler {
public:
    HmmSamplerGaussian(double a, double b, double c, double d):
        a(a), b(b), c(c), d(d),
        // Want to give the generator a different seed each time, so use system clock
        generator(std::chrono::system_clock::now().time_since_epoch().count())
    {}
protected:
    virtual double sample_p0()
    {
        distribution = std::normal_distribution<double>(0, b);
        return distribution(generator);
    }
    virtual double sample_fn(double xn_prev, int n)
    {
        distribution = std::normal_distribution<double>(a*xn_prev, b);
        return distribution(generator);
    }
    virtual double sample_gn(double xn, int n)
    {
        distribution = std::normal_distribution<double>(c*xn, d);
        return distribution(generator);
    }
private:
    const double a, b, c, d;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
};

#endif
