#ifndef CPP_ML_PARTICLE_H
#define CPP_ML_PARTICLE_H

#include "hmm.h"
#include <iostream>
#include <memory>
#include <random>


struct Particle {
    double w;
    std::vector<double> x;
};


class ParticleFilter {
public:
    typedef double (*hn_t)(const std::vector<double> &x, int n);

    ParticleFilter(hn_t hn, Hmm *hmm, std::size_t N);

    void initialise(double y0);
    void step(double y_next);
    double estimate()const;

private:
    virtual double sample_q0()=0;
    virtual double sample_qn(double x_prev, int n)=0;
    virtual double q0(double x0)=0;
    virtual double qn(double x_prev, double x, int n)=0;

    hn_t hn;
    std::unique_ptr<Hmm> hmm;
    int n; // Current step
    std::size_t N; // Number of particles
    std::vector<Particle> particles;

    std::uniform_real_distribution<double> J_sampler;

protected:
    std::default_random_engine generator;
};


class ParticleFilterGaussian: public ParticleFilter {
public:
    ParticleFilterGaussian(
        hn_t hn, Hmm *hmm, std::size_t N,
        double a, double b);

private:
    double q0(double x);
    double qn(double x_prev, double x, int n);
    double sample_q0();
    double sample_qn(double xn_prev, int n);

    double a, b;
    std::normal_distribution<double> distribution;
};


class ParticleFilterUniform: public ParticleFilter {
public:
    ParticleFilterUniform(
        hn_t hn, Hmm *hmm, std::size_t N,
        double width);

private:
    double q0(double x);
    double qn(double x_prev, double x, int n);
    double sample_q0();
    double sample_qn(double xn_prev, int n);

    double width;
    std::uniform_real_distribution<double> distribution;
};

#endif
