#include "hmm.h"

// HmmGaussian

HmmGaussian::HmmGaussian(double a, double b, double c, double d):
    a(a), b(b), c(c), d(d)
{
}

double HmmGaussian::p0(double x)
{
    return (1/std::sqrt(2*M_PI)*b)*
        std::exp(-0.5*std::pow(x/b, 2));
}

double HmmGaussian::fn(double x_prev, double x, int n)
{
    // n unused - fn constant with n
    return (1/std::sqrt(2*M_PI)*b)*
        std::exp(-0.5*std::pow((x - a*x_prev)/b, 2));
}

double HmmGaussian::gn(double x, double y, int n)
{
    // n unused - gn constant with n
    return (1/std::sqrt(2*M_PI)*d)*
        std::exp(-0.5*std::pow((y - c*x)/d, 2));
}


// HmmSampler

HmmSampler::HmmSampler():
    n(-1), xn(0), yn(0)
{
}

void HmmSampler::initialise()
{
    n = 0;
    xn = sample_p0();
    yn = sample_gn(xn, n);
}

void HmmSampler::step()
{
    n++;
    xn = sample_fn(xn, n);
    yn = sample_gn(xn, n);
}


// HmmSamplerGaussian

HmmSamplerGaussian::HmmSamplerGaussian(double a, double b, double c, double d):
    a(a), b(b), c(c), d(d),
    // Want to give the generator a different seed each time, so use system clock
    generator(std::chrono::system_clock::now().time_since_epoch().count())
{}

double HmmSamplerGaussian::sample_p0()
{
    distribution = std::normal_distribution<double>(0, b);
    return distribution(generator);
}

double HmmSamplerGaussian::sample_fn(double xn_prev, int n)
{
    distribution = std::normal_distribution<double>(a*xn_prev, b);
    return distribution(generator);
}

double HmmSamplerGaussian::sample_gn(double xn, int n)
{
    distribution = std::normal_distribution<double>(c*xn, d);
    return distribution(generator);
}
